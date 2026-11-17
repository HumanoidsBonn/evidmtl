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
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models.modelEvidMulti import GLPDepth, Critc, PatchCritic
from models.optimizer import build_optimizers
from models.model_transfer import transfer_weights
import utils.metrics as metrics
from utils.criterion import EvidentialClassificationLoss, EvidentialCrossEntropyLoss, EvidentialSiLogLoss, NIG_NLL, NIG_Reg, EvidentialKLDivergenceLoss, CombinedSiLogNLLLoss
import utils.logging as logging
from utils.uncertainty import *
import torch.nn as nn

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from tqdm import tqdm
import time
import glob
from types import SimpleNamespace
import yaml

import wandb
# Load YAML file
def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return SimpleNamespace(**config_dict)

# Model arguments (same as your training config)
model_name = "silog_kl_muprior"
model_args = load_config(f"./cfg/model_args/{model_name}.yaml")
print(model_args)

ckpt_dir = "./weights/new_weights/full_nll_reg_best_loss.pth"
model_args.ckpt_dir = ckpt_dir

# model_args = SimpleNamespace(
#     exp_name="swinmtl_nyuv2_swin", gpu_or_cpu="gpu", data_path="/home/rohit/workspace/ros1/kimera_ws/src/swin_mtl_evidential/data/nyuv2_dataset",
#     dataset="nyudepthv2", num_classes=23, batch_size=1, workers=6, max_depth=10.0, max_depth_eval=10.0, min_depth_eval=0.001,
#     backbone="swin_base_v2", pretrained="",  # Specify model checkpoint path here if needed
#     window_size=[7, 7, 7, 7], pretrain_window_size=[23, 23, 23, 11], use_shift=[True, True, True, True], depths=[2, 2, 18, 2],
#     drop_path_rate=0.3, use_checkpoint=False, num_deconv=3, num_filters=[256, 128, 64], deconv_kernels=[4, 4, 4],
#     shift_window_test=True, shift_size=2, flip_test=True, epochs=500, max_lr=5e-4, min_lr=3e-5, weight_decay=0.05,
#     layer_decay=0.9, crop_h=448, crop_w=608, log_dir="", val_freq=1, pro_bar=False, save_freq=1,
#     print_freq=100, save_model=True, auto_resume=True,
#     ckpt_dir = "/home/user/menon/workspace/software/swinmtl/logs/2025-02-02_01-17-36_nyudepthv2_16_swinv2_small_1k_500k_deconv3_256_4_448_608_00005_3e-05_09_005_500_7_7_7_7_2_2_18_2_swinmtl_nyuv2_swin/final_model.ckpt"
# )

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'depth_nll', 'depth_ece', 'mIoU', 'pixel_acc', 'ev_mIOU', 'brier', 'top2_acc', 'ece']

def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name}")


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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset_kwargs = {'dataset_name': model_args.dataset, 'data_path': model_args.data_path, 
                      'crop_size': (model_args.crop_h, model_args.crop_w)}
    val_dataset = get_dataset(**dataset_kwargs, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model_args.batch_size, shuffle=False, pin_memory=True)

    # Load model
    model = GLPDepth(args=model_args).to(device)
    if model_args.ckpt_dir:
        load_model(model_args.ckpt_dir, model)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, #shuffle=False,
                                             pin_memory=True)#, sampler=sampler_val)
    
    result_metrics = validate(val_loader, model, args= model_args, device= "cuda:2")
    # Training settings
    print(result_metrics)
    
global_step = 0    

def validate(val_loader, model, args, device):#, crit, criterion_d, criterion_seg, device, epoch, args, alpha,log_dir, wandb_run = None, writer=None, criterion_d_reg=None, criterion_seg_reg=None):

    
    model.eval()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(tqdm(val_loader)):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        seg_gt = batch['seg'].to(device)
        filename = batch['filename'][0]

        
        with torch.no_grad():
            s=time.time()
            pred = model(input_RGB)
            #print("time: ", time.time()-s)
        pred_d = pred['pred_d'].squeeze()
        pred_seg = pred['pred_seg'].squeeze()
        check_nan_inf(pred_d, "val:pred_d")
        check_nan_inf(pred_seg, "val:pred_seg")
        #print(f"pred_depth shape: {pred_d.shape}")  # Should be [B, 1, H, W]
        #print(f"pred_seg shape: {pred_seg.shape}")  # Should be [B, C, H, W]
        
        #log_evidential_stats(pred_depth=pred_d, pred_seg=pred_seg, wandb_run=wandb_run, writer=writer, global_step=global_step, mode="val")


        dmin = 1e-3
        dmax = 10.0
        depth_gt = depth_gt.squeeze()     
        #depth_gt = torch.clamp(depth_gt, min=0.0, max=1.0)  # Normalize input
        depth_gt = dmin * torch.exp(torch.log(torch.tensor(dmax / dmin)) * depth_gt) 
        #depth_gt = torch.clamp(depth_gt, min=dmin, max=dmax)  # Ensure it stays in range
        check_nan_inf(depth_gt, "val:depth_gt")
        
        #pred_seg = pred_seg                 
        #pred_seg_max = torch.argmax(pred_seg, dim=1)
        pred_seg_max = torch.argmax(pred_seg, dim=1).unsqueeze(1).float()
        pred_mean_depth = pred_d[:, 0:1, :, :]  # Extract only the first channel (μ)
        alpha = pred_d[:, 1:2, :, :]
        beta = pred_d[:, 2:3, :, :]
        gamma = pred_d[:, 3:4, :, :]
        check_nan_inf(alpha, "val:alpha")
        check_nan_inf(beta, "val:beta") 
        check_nan_inf(gamma, "val:gamma")

        #print(f"pred_mean_depth shape: {pred_mean_depth.shape}")  # Should be [B, 1, H, W]
        #print(f"alpha shape: {alpha.shape}")  # Should be [B, 1, H, W]
        #print(f"beta shape: {beta.shape}")  # Should be [B, 1, H, W]
        #print(f"gamma shape: {gamma.shape}")  # Should be [B, 1, H, W]
        #print(f"depth_gt shape: {depth_gt.shape}")  # Should be [B, 1, H, W]
        
        aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty = compute_depth_uncertainties(alpha.squeeze(1), beta.squeeze(1), gamma.squeeze(1))
        check_nan_inf(aleatoric_uncertainty, "val:aleatoric_uncertainty")
        check_nan_inf(epistemic_uncertainty, "val:epistemic_uncertainty")
        check_nan_inf(total_uncertainty, "val:total_uncertainty")
        depth_nll = metrics.compute_negative_log_likelihood( pred_mean_depth.squeeze(1), depth_gt, total_uncertainty)
        depth_ece = metrics.compute_depth_ece(pred_mean_depth.squeeze(1), depth_gt, total_uncertainty,  device=device)
        

        ####EVALUATION METRICES Depth
        pred_crop, gt_crop = metrics.cropping_img(args, pred_mean_depth.squeeze(), depth_gt)
        #print(pred_crop.shape, gt_crop.shape)
        #check_nan_inf(gt_crop, "gt_crop")
        computed_result = metrics.eval_depth(pred_crop, gt_crop)   

        #print(f"pred_seg shape: {pred_seg.shape}")  # Should be [B, C, H, W]
        #print(f"seg_gt shape: {seg_gt.shape}")  # Should be [B, H, W]
        #print(f"pred_seg_max shape: {pred_seg_max.shape}")  # Should be [B, 1, H, W]
        ####EVALUATION METRICES Segmentation
        #print(f"pred_seg_max shape: {pred_seg_max.shape}")  
        #print(f"seg_gt shape: {seg_gt.shape}") 
        computed_result_seg = metrics.eval_seg(pred_seg_max.squeeze(1), seg_gt, args.num_classes, global_step)  
        computed_result_evid_seg = metrics.eval_evid_seg(pred_seg, seg_gt, args.num_classes)


        for key in result_metrics.keys():
            #(key)
            if key == 'mIoU':
                result_metrics[key] += computed_result_seg[key]
            elif key == 'pixel_acc':
                result_metrics[key] += computed_result_seg[key]
            elif key in ['ev_mIOU', 'brier', 'top2_acc', 'ece']:  
                result_metrics[key] += computed_result_evid_seg[key]  # Accumulate evidential metrics

            elif key == 'depth_nll':
                result_metrics[key] += depth_nll["depth_nll"]
            elif key == 'depth_ece':
                result_metrics[key] += depth_ece["depth_ece"]
            else:
                result_metrics[key] += computed_result[key]
            
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    return result_metrics


if __name__ == '__main__':
    main()
