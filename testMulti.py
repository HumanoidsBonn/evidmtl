import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from types import SimpleNamespace

from models.modelMulti import GLPDepth
from dataset.base_dataset import get_dataset
from utils.metrics import eval_depth, eval_seg, cropping_img
from utils.criterion import SiLogLoss, CrossEntropyLoss
import cv2
import numpy as np

# Model arguments (same as your training config)
model_args = SimpleNamespace(
    exp_name="swinmtl_nyuv2_swin", gpu_or_cpu="gpu", data_path="/home/rohit/workspace/ros1/kimera_ws/src/swin_mtl_evidential/data/nyuv2_dataset",
    dataset="nyudepthv2", num_classes=14, batch_size=1, workers=6, max_depth=10.0, max_depth_eval=10.0, min_depth_eval=0.001,
    backbone="swin_small_v2", pretrained="",  # Specify model checkpoint path here if needed
    window_size=[7, 7, 7, 7], pretrain_window_size=[23, 23, 23, 11], use_shift=[True, True, True, True], depths=[2, 2, 18, 2],
    drop_path_rate=0.3, use_checkpoint=False, num_deconv=3, num_filters=[256, 128, 64], deconv_kernels=[4, 4, 4],
    shift_window_test=True, shift_size=2, flip_test=True, epochs=500, max_lr=5e-4, min_lr=3e-5, weight_decay=0.05,
    layer_decay=0.9, crop_h=448, crop_w=608, log_dir="", val_freq=1, pro_bar=False, save_freq=1,
    print_freq=100, save_model=True, auto_resume=True,
    ckpt_dir = "/home/user/menon/workspace/software/swinmtl/logs/2025-02-02_01-17-36_nyudepthv2_16_swinv2_small_1k_500k_deconv3_256_4_448_608_00005_3e-05_09_005_500_7_7_7_7_2_2_18_2_swinmtl_nyuv2_swin/final_model.ckpt"
)

def load_model(ckpt, model):
    """Loads model weights from checkpoint."""
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        return
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    state_dict = ckpt_dict['model']
    weights = {key[len('module.'):] if key.startswith('module.') else key: value for key, value in state_dict.items()}
    model.load_state_dict(weights)
    print(f"Model loaded from {ckpt}")

def validate(val_loader, model, criterion_d, criterion_seg, device):
    """Runs validation on the dataset."""
    model.eval()
    depth_loss, seg_loss = 0.0, 0.0
    metrics_results = {key: 0 for key in ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 
                                          'rmse_log', 'log10', 'silog', 'mIoU', 'pixel_acc']}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            input_RGB = batch['image'].to(device)
            depth_gt = batch['depth'].to(device)
            seg_gt = batch['seg'].to(device)

            pred = model(input_RGB)
            pred_depth = pred['pred_d'].squeeze()
            pred_seg = pred['pred_seg']
            
            # Print mean, median, min, max of pred_depth
            pred_depth_np = pred_depth.clone().cpu().numpy()
            print(f"Mean: {np.mean(pred_depth_np):.4f}, Median: {np.median(pred_depth_np):.4f}, Min: {np.min(pred_depth_np):.4f}, Max: {np.max(pred_depth_np):.4f}")
            
            dmin = model_args.min_depth_eval
            dmax = model_args.max_depth_eval #if 'nyudepthv2' in args.dataset else 80.0
            depth_gt = depth_gt.squeeze()     
            depth_gt = torch.clamp(depth_gt, min=0.0, max=1.0)  # Normalize input
            depth_gt = dmin * torch.exp(torch.log(torch.tensor(dmax / dmin)) * depth_gt) # Reverse normalization
            depth_gt = torch.clamp(depth_gt, min=dmin, max=dmax)  # Ensure it stays in range

            # Compute losses
            loss_d = criterion_d(pred_depth, depth_gt)
            loss_seg = criterion_seg(pred_seg, seg_gt.to(torch.long))
            total_loss = loss_d + loss_seg

            depth_loss += loss_d.item()
            seg_loss += loss_seg.item()

            # Compute evaluation metrics
            #pred_depth_np, depth_gt_np = pred_depth.cpu().numpy(), depth_gt.cpu().numpy()
           # pred_seg_np, seg_gt_np = torch.argmax(pred_seg, dim=1).cpu().numpy(), seg_gt.cpu().numpy()
            pred_mask = torch.argmax(pred_seg, dim=1)
            
            pred_crop, gt_crop = cropping_img(model_args, pred_depth.squeeze(), depth_gt)
            depth_metrics = eval_depth(pred_depth, depth_gt)
            seg_metrics = eval_seg(pred_mask, seg_gt, model_args.num_classes)
            
            results_dir = "/home/user/menon/workspace/software/swinmtl/results"
            filename = batch['filename'][0]
            save_path = os.path.join(results_dir, filename)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')

            save_result = False
            if save_result:
              if model_args.dataset == 'kitti':
                  pred_d_numpy = pred_depth.cpu().numpy() * 256.0
                  cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
              elif model_args.dataset == 'cityscapes':
                  pred_d_numpy = pred_depth.cpu().numpy() 
                  cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
              else:
                  single_depth = pred_depth[0, :, :]
                  pred_d_numpy = single_depth.cpu().numpy() * 1000.0
                  cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                              [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            print("Depth Metrics:", depth_metrics)
            print("Seg Metrics:", seg_metrics)
            
            

            for key in metrics_results.keys():
                metrics_results[key] += depth_metrics.get(key, 0) + seg_metrics.get(key, 0)

        # Compute average metrics
        num_samples = len(val_loader)
        depth_loss /= num_samples
        seg_loss /= num_samples
        for key in metrics_results.keys():
            metrics_results[key] /= num_samples

    print(f"\nValidation Depth Loss: {depth_loss:.4f}, Segmentation Loss: {seg_loss:.4f}")
    print("Validation Metrics:", metrics_results)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset_kwargs = {'dataset_name': model_args.dataset, 'data_path': model_args.data_path, 
                      'crop_size': (model_args.crop_h, model_args.crop_w)}
    val_dataset = get_dataset(**dataset_kwargs, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model_args.batch_size, shuffle=False, pin_memory=True)

    # Load model
    model = GLPDepth(args=model_args).to(device)
    if model_args.ckpt_dir:
        load_model(model_args.ckpt_dir, model)
    
    criterion_d = SiLogLoss()
    criterion_seg = CrossEntropyLoss()

    # Run validation
    validate(val_loader, model, criterion_d, criterion_seg, device)

if __name__ == '__main__':
    main()
