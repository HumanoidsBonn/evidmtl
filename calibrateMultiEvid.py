import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from models.modelEvidMulti import GLPDepth  # Import your model
import utils.metrics as metrics
from utils.criterion import EvidentialClassificationLoss, EvidentialCrossEntropyLoss, EvidentialSiLogLoss, NIG_NLL, NIG_Reg, EvidentialKLDivergenceLoss, EvidentialLossSumOfSquares, CombinedSiLogNLLLoss
import argparse
import utils.logging as logging
from utils.uncertainty import *
import torch.nn as nn

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from tqdm import tqdm
import time
import glob
import yaml
from types import SimpleNamespace
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm


metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'mIoU', 'pixel_acc', 'ev_mIOU', 'brier', 'top2_acc', 'ece']

def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name}")
        
def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return SimpleNamespace(**config_dict)

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

def load_glp_depth_model(args, ckpt):
    model = GLPDepth(args=args).to(device=device)
    
    load_model(ckpt, model)  # Load weights
    return model

# Model arguments (same as your training config)
model_name = "silog_kl"
model_args = load_config(f"/home/user/menon/workspace/software/swinmtl/configs/{model_name}.yaml")
print(model_args)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str, default=".logs / 2025-02-28_19-51-15_alpha2_beta0.1_gamma1.0_ce_silog_kl_reg_linseg_quaddep_0.7_nyudepthv2labels22scannet_mod_8_swinv2_base_22k_500k_deconv3_32_2_448_608_00004_4e-05_09_005_500_30_30_30_15_2_2_18_2/final_model.ckpt")
parser.add_argument("--save_path", type=str, default="./model_zoo/calibrated_silog_kl_muprior_lr1e-5.ckpt")
parser.add_argument("--epochs", type=int, default=10)  # Short fine-tuning (5-20 epochs)
parser.add_argument("--lr", type=float, default=1e-5)  # Lower LR for stability
parser.add_argument("--warmup_epochs", type=int, default=5)  # Gradual reg loss warmup
args = parser.parse_args()

# Load Pretrained Model
model = load_glp_depth_model(model_args, args.pretrained_path)
print("✅ Loaded Pretrained Model from", args.pretrained_path)

# **Freeze Encoder & Segmentation Head** (Only Fine-tune Depth)
for param in model.encoder.parameters():
    param.requires_grad = False  # Freeze shared encoder
for param in model.decoder.parameters():
    param.requires_grad = False  # Freeze shared decoder
for param in model.last_layer_seg.parameters():
    param.requires_grad = False  # Keep segmentation fixed

for param in model.last_layer_depth.parameters():
    param.requires_grad = True  # Fine-tune only depth uncertainty

# Define Loss Functions
criterion_d = NIG_NLL()
criterion_d_reg = NIG_Reg(total_epochs=args.epochs, kl=False, reg= True)
criterion_seg = EvidentialCrossEntropyLoss()  # Keep segmentation loss same
criterion_seg_reg = EvidentialKLDivergenceLoss(total_epochs=args.epochs)

# Optimizer (Only for Depth Head)
optimizer = optim.Adam(model.last_layer_depth.parameters(), lr=args.lr)

dataset_kwargs = {'dataset_name': model_args.dataset, 'data_path': model_args.data_path}
dataset_kwargs['crop_size'] = (model_args.crop_h, model_args.crop_w)

train_dataset = get_dataset(**dataset_kwargs, split='train')
val_dataset = get_dataset(**dataset_kwargs, split='val')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8,
                                            num_workers=model_args.workers, shuffle=True, 
                                            pin_memory=True, drop_last=True)

# Training Loop
for epoch in range(args.epochs):
    total_depth_loss, total_depth_reg_loss, total_seg_loss, total_seg_reg_loss = 0, 0, 0, 0
    for batch_idx, batch in enumerate(tqdm(train_loader)):      
        input_RGB = batch['image'].to(device) 
        depth_gt = batch['depth'].to(device) 
        dmin = 1e-3
        dmax = 10.0
        #depth_gt = torch.clamp(depth_gt, min=0.0, max=1.0)  # Normalize input
        depth_gt = dmin * torch.exp(torch.log(torch.tensor(dmax / dmin)) * depth_gt) 
        #depth_gt = torch.clamp(depth_gt, min=dmin, max=dmax)  # Ensure it stays in range
        seg_gt = batch['seg'].to(device)    
        preds = model(input_RGB)
        pred_depth = preds['pred_d'].squeeze()
        mu, alpha, beta, gamma = torch.chunk(pred_depth, 4, dim=1)
        print("alpha", alpha.mean().item(), "beta", beta.mean().item(), "gamma", gamma.mean().item())
        pred_seg = preds['pred_seg'].squeeze()
        check_nan_inf(pred_depth, "pred_mu (mean depth)")
        check_nan_inf(pred_seg, "pred_seg (segmentation)")
        check_nan_inf(depth_gt, "depth_gt")
        optimizer.zero_grad()


        # Compute Losses
        depth_loss = criterion_d(pred_depth, depth_gt)
        depth_reg_loss = criterion_d_reg(pred_depth, depth_gt)
        seg_loss = criterion_seg(pred_seg, seg_gt)  # Keep segmentation loss
        seg_reg_loss = criterion_seg_reg(pred_seg, seg_gt, epoch)

        # **Warmup Regularization** (gradually increase reg weight over `warmup_epochs`)
        reg_weight = min(1.0, epoch / args.warmup_epochs)
        loss = depth_loss + reg_weight * depth_reg_loss #+ seg_loss  + seg_reg_loss # **Ensure segmentation is not degraded**

        loss.backward()
        optimizer.step()
        mu, alpha, beta, gamma = torch.chunk(pred_depth, 4, dim=1)
        print("Depth Loss:", depth_loss.item(), "Depth Reg Loss:", depth_reg_loss.item(), "Seg Loss:", seg_loss.item(), "Seg Reg Loss:", seg_reg_loss.item())
        print("alpha", alpha.mean().item(), "beta", beta.mean().item(), "gamma", gamma.mean().item())
        #print("Aleatoric Uncertainty:", (beta/(alpha -1 + 1e-6)).item(), "Epistemic Uncertainty:", (beta/(gamma*(alpha -1 + 1e-6))).item())
        total_depth_loss += depth_loss.item()
        total_depth_reg_loss += depth_reg_loss.item()
        total_seg_loss += seg_loss.item()
        total_seg_reg_loss += seg_reg_loss.item()

    print(f"Epoch {epoch+1}/{args.epochs} | NLL: {total_depth_loss:.4f} | Reg: {total_depth_reg_loss:.4f} | Seg Loss: {total_seg_loss:.4f} | Seg Reg: {total_seg_reg_loss:.4f}")

# Save Calibrated Model
torch.save(model.state_dict(), args.save_path)
print("✅ Saved Calibrated Model to", args.save_path)
