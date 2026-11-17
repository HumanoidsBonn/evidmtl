# ------------------------------------------------------------------------------
# The original code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# moddified by Pardis Taghavi (taghavi.pardis@gmail.com)
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.modelMulti import GLPDepth, Critc, PatchCritic
from models.optimizer import build_optimizers
import utils.metrics as metrics
from utils.criterion import SiLogLoss, CrossEntropyLoss, SmoothLoss
import utils.logging as logging
import torch.nn as nn

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from tqdm import tqdm
import time
import glob

import wandb
import random

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'mIoU', 'pixel_acc']



def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    
    state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value
    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def freeze_encoder(model):
    #freeze encoder except prompt layers
    for name, param in model.named_parameters():
        if 'prompt' not in name and 'encoder' in name:
            param.requires_grad = False


def mixed_data(input_RGB, depth_gt, seg_gt):

    bs, _, h, w = input_RGB.shape
    depth_gt = depth_gt.unsqueeze(1)
    seg_gt = seg_gt.unsqueeze(1)
    shuffled_idx = torch.randperm(bs)
    shuffled_input_RGB = input_RGB[shuffled_idx]
    shuffled_depth_gt = depth_gt[shuffled_idx]
    shuffled_seg_gt = seg_gt[shuffled_idx]
    
    filled_mask= torch.ones((bs, 1, h, w), device=input_RGB.device, dtype=torch.bool)

    for i in range(bs):
        num = torch.randint(1, 5, (1,)).item()
        size= torch.randint(h//8, h//2, (num,))
        x = torch.randint(0, h, (num,))
        y = torch.randint(0, w, (num,))
        for j in range(num):
            filled_mask[i, :, x[j]:x[j]+size[j], y[j]:y[j]+size[j]] = False

    input_RGB = input_RGB * filled_mask + shuffled_input_RGB * (~filled_mask)
    depth_gt = depth_gt * filled_mask + shuffled_depth_gt * (~filled_mask)
    seg_gt = seg_gt * filled_mask + shuffled_seg_gt * (~filled_mask)
    return input_RGB, depth_gt.squeeze(), seg_gt.squeeze()


def get_gradient(crit, real, fake, epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient= torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient
    

def gradient_penalty(gradient):

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)
    pretrain = args.pretrained.split('.')[0]
    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if 'swin' in args.backbone:
        for i in args.window_size:
            name.append(str(i))
        for i in args.depths:
            name.append(str(i))
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)
    
    # Logging
    exp_name = datetime.now().strftime("%Y_%m_%d_%H-%M-%S_") + exp_name
    wandb_run = wandb.init(
        project="SwinMTL_NYUv2",  # Change project name as needed
        name=exp_name,  # Experiment name
        config=args  # Automatically log all args as config
    )
    log_dir = os.path.join(args.log_dir, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    log_txt = os.path.join(log_dir, 'logs.txt')  
    logging.log_args_to_txt(log_txt, args)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    model = GLPDepth(args=args)
    crit = Critc()

    torch.cuda.empty_cache()

    # CPU-GPU agnostic settings
    if 'gpu' in args.gpu_or_cpu:
        gpu_ids = args.gpu_or_cpu.split(':')[-1]  # Extract GPU list
        gpu_ids = [int(gpu) for gpu in gpu_ids.split(',')] if ',' in gpu_ids else [int(gpu_ids)]  # Parse multiple GPUs
        
        device = torch.device(f'cuda:{gpu_ids[0]}')  # Use first GPU as main device
        cudnn.benchmark = True
    
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)  # Assign remaining GPUs to DataParallel
        else:
            model.to(device)  # Single GPU, move model to device
    else:
        device = torch.device('cpu')  # CPU mode
    
    model.to(device)
    crit.to(device)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs, split='train')
    val_dataset = get_dataset(**dataset_kwargs, split='val')

    #dataset_size = len(train_dataset)
    #subset_size = int(dataset_size * 0.1)  # Use only 50% of data
    #indices = np.random.choice(dataset_size, subset_size, replace=False)
    #sampler = torch.utils.data.SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers, shuffle=True,
                                               pin_memory=True, drop_last=True)
    #subset_val_size = int(len(val_dataset) * 0.05)
    #indices_val = np.random.choice(len(val_dataset), subset_val_size, replace=False)
    #sampler_val = torch.utils.data.SubsetRandomSampler(indices_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True)
    
    # Training settings
    criterion_d = SiLogLoss()
    criterion_seg = CrossEntropyLoss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='SwinLayerDecayOptimizerConstructor',
                paramwise_cfg=dict(num_layers=args.depths, layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))
    optimizer_crit = optim.Adam(crit.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)

    
    start_ep = 1
    if args.resume_from:
        load_model(args.resume_from, model.module, optimizer)
        strlength = len('_model.ckpt')
        resume_ep = int(args.resume_from[-strlength-2:-strlength])
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        start_ep = resume_ep + 1
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        strlength = len('_model.ckpt')
        idx = [ckpt[-strlength-2:-strlength] for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep + 1

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)
    last_checkpoint = None
    best_loss = float('inf')
    best_mIoU = 0
    best_depth_metric = float('inf')  # RMSE (lower is better)

    best_loss_path = os.path.join(log_dir, 'best_loss.pth')
    best_mIoU_path = os.path.join(log_dir, 'best_mIoU.pth')
    best_depth_path = os.path.join(log_dir, 'best_depth.pth')
    # Perform experiment
    for epoch in range(start_ep, args.epochs + 1):
        
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))

        loss_train = train(train_loader, model, crit, criterion_d, criterion_seg, log_txt, optimizer, optimizer_crit,
                           device=device, epoch=epoch, alpha=0.5, args=args, wandb_run=wandb_run)
        writer.add_scalar('Training loss', loss_train, epoch) #total loss
        wandb_run.log({
        "epoch": epoch,
        "train_loss": loss_train
        })
        #save evry 20 epochs
        if args.save_model and epoch % 10 == 0:
            checkpoint_path = os.path.join(log_dir, f'epoch_{epoch:02d}_model.ckpt')
            print('Saving model...')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                checkpoint_path)
            #wandb_run.save(checkpoint_path)
            if last_checkpoint and os.path.exists(last_checkpoint):
                os.remove(last_checkpoint)
                print(f"Deleted previous checkpoint: {last_checkpoint}")
            last_checkpoint = checkpoint_path
            
        if args.save_model and epoch == args.epochs:
            print('Saving final model...')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                os.path.join(log_dir, 'final_model.ckpt'))
            wandb_run.save(os.path.join(log_dir, 'final_model.ckpt'))

        if epoch % args.val_freq == 0:
            print('Validating...')
            results_dict, loss_val = validate(val_loader, model,crit, criterion_d, criterion_seg, device=device, epoch=epoch, args=args, alpha=0.5,
                                              log_dir=log_dir, wandb_run= wandb_run)
            
            if loss_val < best_loss:  # Best model based on total loss
                best_loss = loss_val
                print(f"Saving new best model (val loss: {best_loss:.4f})...")
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            best_loss_path)
                wandb_run.save(best_loss_path)
            if results_dict['mIoU'] > best_mIoU:  # Best model for segmentation
                best_mIoU = results_dict['mIoU']
                print(f"Saving new best model (mIoU: {best_mIoU:.4f})...")
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            best_mIoU_path)
                wandb_run.save(best_mIoU_path)
            if results_dict['rmse'] < best_depth_metric:  # Best model for depth prediction
                best_depth_metric = results_dict['rmse']
                print(f"Saving new best model (depth RMSE: {best_depth_metric:.4f})...")
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                            best_depth_path)
                wandb_run.save(best_depth_path)

            writer.add_scalar('Val loss', loss_val, epoch) #total loss
            wandb_run.log({
                "epoch": epoch,
                "val_loss": loss_val
                })
            result_lines = logging.display_result(results_dict)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)                

            for each_metric, each_results in results_dict.items():
                writer.add_scalar(each_metric, each_results, epoch)
                wandb_run.log({each_metric: each_results, "epoch": epoch})
        torch.cuda.empty_cache()
    wandb_run.finish()

def train(train_loader, model,crit, criterion_d, criterion_seg, log_txt, optimizer,optimizer_crit, device, epoch, alpha, args, wandb_run=None):   

    global global_step
    model.train()
    crit.train()

    depth_loss = logging.AverageMeter()
    seg_loss = logging.AverageMeter()
    loss = logging.AverageMeter()

    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    skip_probability = 0.1 

    for batch_idx, batch in enumerate(train_loader):  
        # if batch_idx > 5:
        #     break    
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step /
                                            iterations/half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step /
                                            iterations/half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale'] if 'swin' in args.backbone else current_lr

        input_RGB = batch['image'].to(device) 
        depth_gt = batch['depth'].to(device)  

        dmin = 1e-3
        dmax = 10.0 
        #depth_gt = torch.clamp(depth_gt, min=0.0, max=1.0)  # Normalize input
        depth_gt = dmin * torch.exp(torch.log(torch.tensor(dmax / dmin)) * depth_gt) 
        #depth_gt = torch.clamp(depth_gt, min=dmin, max=dmax)  # Ensure it stays in range
        seg_gt = batch['seg'].to(device)      
        
        input_RGB, depth_gt, seg_gt = mixed_data(input_RGB, depth_gt, seg_gt)
        
        
        preds = model(input_RGB)
        pred_depth = preds['pred_d'].squeeze()
        pred_seg = preds['pred_seg'].squeeze()
        pred_seg_max = torch.argmax(pred_seg, dim=1).unsqueeze(1).float()
        
        fake_depth_seg = torch.cat((pred_depth.unsqueeze(1), pred_seg_max), dim=1)  
        true_depth_seg = torch.cat((depth_gt.unsqueeze(1), seg_gt.unsqueeze(1)), dim=1)        

        optimizer.zero_grad()  
        optimizer_crit.zero_grad()

        loss_d = criterion_d(pred_depth, depth_gt)
        loss_seg = criterion_seg(pred_seg, seg_gt.to(torch.long))
    
        crit_fake_pred = crit(fake_depth_seg.detach())
        crit_real_pred = crit(true_depth_seg)
        epsilon = torch.rand(len(true_depth_seg ), 1, 1, 1, device=device, requires_grad=True)
        gradient = get_gradient(crit, true_depth_seg, fake_depth_seg.detach(), epsilon)
        gp = gradient_penalty(gradient)
        crit_loss = -(torch.mean(crit_real_pred) - torch.mean(crit_fake_pred)) + 100 * gp
        crit_loss.backward(retain_graph=True)
        optimizer_crit.step()

        crit_fake_pred2 = crit(fake_depth_seg)
        gen_loss = -(torch.mean(crit_fake_pred2))
        total_loss = loss_seg + loss_d + 0.1 * gen_loss

        total_loss.backward()                
        depth_loss.update(loss_d.item(), input_RGB.size(0)) #logging info   
        seg_loss.update(loss_seg.item(), input_RGB.size(0)) #logging info   
        loss.update(total_loss.item(), input_RGB.size(0)) #logging info   
        wandb_run.log({
            "epoch": epoch,
            "train/depth_loss": depth_loss.avg,
            "train/seg_loss": seg_loss.avg,
            "train/total_loss": loss.avg,
            "train/learning_rate": current_lr
            })
        if args.pro_bar:
            logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,   
                                ('Depth Loss: %.4f (%.4f)' %
                                (depth_loss.val, depth_loss.avg)),
                                ('Seg Loss: %.4f (%.4f)' %
                                (seg_loss.val, seg_loss.avg)),
                                ('Total Loss: %.4f (%.4f)' %
                                (loss.val, loss.avg))
                                )

        if batch_idx % args.print_freq == 0:                                  
            result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                'Total Loss: {loss}, LR: {lr}\n'.format(
                    epoch, batch_idx, iterations,
                    loss=loss.avg, lr=current_lr
                )
            result_lines.append(result_line)
            print(result_line)
            
        optimizer.step()

    with open(log_txt, 'a') as txtfile:
        txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        for result_line in result_lines:
            txtfile.write(result_line)   

    return total_loss


def validate(val_loader, model, crit, criterion_d, criterion_seg, device, epoch, args, alpha,log_dir, wandb_run = None):

    depth_loss = logging.AverageMeter()
    seg_loss = logging.AverageMeter()
    loss = logging.AverageMeter()
    
    model.eval()
    crit.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    skip_probability = 0.5

    for batch_idx, batch in enumerate(tqdm(val_loader)):
        
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        seg_gt = batch['seg'].to(device)
        filename = batch['filename'][0]

        #print("depth_gt shape: ", depth_gt.shape)
        #print("input_RGB shape: ", input_RGB.shape)
        with torch.no_grad():
            # if args.shift_window_test:
            #     bs, _, h, w = input_RGB.shape
            #     assert bs == 1
            #     interval_all = w - h
            #     interval = interval_all // (args.shift_size-1)
            #     sliding_images = []
            #     sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device) 
            #     for i in range(args.shift_size):
            #         sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
            #         sliding_masks[..., :, i*interval:i*interval+h] += 1
            #     input_RGB = torch.cat(sliding_images, dim=0)
            # if args.flip_test:
            #     input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
            # s=time.time()
            pred = model(input_RGB)
            #print("time: ", time.time()-s)
        pred_d = pred['pred_d']
        pred_seg = pred['pred_seg']  

        #print("pred_d shape: ", pred_d.shape)

        # if args.flip_test:
        #     batch_s = pred_d.shape[0]//2 #batch_s is 
        #     pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        #     pred_seg = (pred_seg[:batch_s] + torch.flip(pred_seg[batch_s:], [3]))/2.0
            
        # if args.shift_window_test:
        #     pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
        #     pred_s2 = torch.zeros((bs, args.num_classes, h, w), device=pred_seg.device)

        #     for i in range(args.shift_size):
        #         pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
        #         pred_s2[..., :, i*interval:i*interval+h] += pred_seg[i:i+1]
        #     pred_d = pred_s/sliding_masks
        #     pred_seg = pred_s2/sliding_masks

        dmin = 1e-3
        dmax = 10.0
        depth_gt = depth_gt.squeeze()     
        #depth_gt = torch.clamp(depth_gt, min=0.0, max=1.0)  # Normalize input
        depth_gt = dmin * torch.exp(torch.log(torch.tensor(dmax / dmin)) * depth_gt) 
        #depth_gt = torch.clamp(depth_gt, min=dmin, max=dmax)  # Ensure it stays in range
        
        pred_seg = pred_seg                 
        pred_seg_max = torch.argmax(pred_seg, dim=1)

        fake_depth_seg = torch.cat((pred_d, pred_seg_max.unsqueeze(1)), dim=1) 
        
        loss_d   = criterion_d(pred_d.squeeze(), depth_gt)
        loss_seg = criterion_seg(pred_seg, seg_gt.to(torch.long))
        loss_gen = - (torch.mean(crit(fake_depth_seg)))

        total_loss =   loss_seg + loss_d + 0.1 * loss_gen
        
        depth_loss.update(loss_d.item(), input_RGB.size(0))   
        seg_loss.update(loss_seg.item(), input_RGB.size(0))
        loss.update(total_loss.item(), input_RGB.size(0))
        wandb.log({
            "epoch": epoch,
            "val/depth_loss": depth_loss.avg,
            "val/seg_loss": seg_loss.avg,
            "val/total_loss": loss.avg,
            "val/depth_mean": pred_d.mean().item(),
            "val/depth_min": pred_d.min().item(),
            "val/depth_max": pred_d.max().item()
            })

        ####EVALUATION METRICES Depth
        #print("gt shape: ", depth_gt.shape)
        pred_crop, gt_crop = metrics.cropping_img(args, pred_d.squeeze(), depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)   


        ####EVALUATION METRICES Segmentation
        computed_result_seg = metrics.eval_seg(pred_seg_max, seg_gt)  

        save_path = os.path.join(result_dir, filename)
        if save_path.split('.')[-1] == 'jpg':
            save_path = save_path.replace('jpg', 'png')

        if args.save_result:
            if args.dataset == 'kitti':
                pred_d_numpy = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            elif args.dataset == 'cityscapes':
                pred_d_numpy = pred_d.cpu().numpy() 
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d_numpy = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        loss_seg = seg_loss.avg
        total_loss = loss.avg

        if args.pro_bar:
            logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for key in result_metrics.keys():
            #(key)
            if key == 'mIoU':
                result_metrics[key] += computed_result_seg[key]
            elif key == 'pixel_acc':
                result_metrics[key] += computed_result_seg[key]
            else:
                result_metrics[key] += computed_result[key]
            
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics, total_loss


if __name__ == '__main__':
    main()
