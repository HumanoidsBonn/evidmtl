import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_depth_ece, compute_negative_log_likelihood

def compute_depth_uncertainties(alpha, beta, gamma, epsilon=1e-6):
    alpha_safe = torch.clamp(alpha, min=1.00001)  # Alpha should always be > 1
    gamma_safe = torch.clamp(gamma, min=epsilon)   # Avoid division by zero
    # Aleatoric Uncertainty
    aleatoric_uncertainty = beta / (alpha_safe - 1)
    # Epistemic Uncertainty
    epistemic_uncertainty = beta / (gamma_safe * (alpha_safe - 1))
    # Total Uncertainty
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    return aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty

def compute_segmentation_uncertainties(pred):
    alpha = torch.clamp(pred, min=1.0)
    num_classes = alpha.shape[1]
    St = torch.sum(alpha, dim=1, keepdim=True)  # Shape: (batch_size, 1, H, W)
    pred_prob = alpha / St  # Shape: (batch_size, num_classes, H, W)
    epistemic_uncertainty = num_classes / St  # Shape: (B, 1, H, W)
    aleatoric_uncertainty = (alpha * (St - alpha)) / (St * (St + 1))
    total_uncertainty = -torch.sum(pred_prob * torch.log(pred_prob + 1e-6), dim=1, keepdim=True)
    #total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    return aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty, St

def compute_statistics(var):
    mean = var.mean().item()
    median = var.median().item()
    std =  var.std().item()
    min_val = var.min().item()
    max_val = var.max().item()
    


    return mean, median, std, min_val, max_val

def log_evidential_stats(pred_depth, pred_seg, gt_depth, gt_seg, wandb_run, writer, global_step, mode="train"):
    alpha = pred_depth[:, 1:2, :, :]
    beta = pred_depth[:, 2:3, :, :]
    gamma = pred_depth[:, 3:4, :, :]
    mu = pred_depth[:, 0:1, :, :]

    d_alea_unc, d_epi_unc, d_total_unc = compute_depth_uncertainties(alpha, beta, gamma)
    seg_alea_unc, seg_epi_unc, seg_total_unc, seg_S = compute_segmentation_uncertainties(pred_seg)

    d_alea_unc_mean, d_alea_unc_median, d_alea_unc_dev, d_alea_unc_min, d_alea_unc_max = compute_statistics(d_alea_unc)
    d_epi_unc_mean, d_epi_unc_median, d_epi_unc_dev, d_epi_unc_min, d_epi_unc_max = compute_statistics(d_epi_unc)
    d_total_unc_mean, d_total_unc_median, d_total_unc_dev, d_total_unc_min, d_total_unc_max = compute_statistics(d_total_unc)

    seg_alea_unc_mean, seg_alea_unc_median, seg_alea_unc_dev, seg_alea_unc_min, seg_alea_unc_max = compute_statistics(seg_alea_unc)
    seg_epi_unc_mean, seg_epi_unc_median, seg_epi_unc_dev, seg_epi_unc_min, seg_epi_unc_max = compute_statistics(seg_epi_unc)
    seg_total_unc_mean, seg_total_unc_median, seg_total_unc_dev, seg_total_unc_min, seg_total_unc_max = compute_statistics(seg_total_unc)
    seg_S_mean, seg_S_median, seg_S_dev, seg_S_min, seg_S_max = compute_statistics(seg_S)


    # Compute statistics for wandb logging
    alpha_mean, alpha_median, alpha_dev,alpha_min, alpha_max = compute_statistics(alpha)
    beta_mean, beta_median, beta_dev,beta_min, beta_max = compute_statistics(beta)
    gamma_mean, gamma_median, gamma_dev,gamma_min, gamma_max = compute_statistics(gamma)

    wandb_run.log({
        "step": global_step,
        f"{mode}/evid_depth/alpha_mean": alpha_mean, #f"{mode}/evid_depth/alpha_median": alpha_median, f"{mode}/evid_depth/alpha_dev":alpha_dev, f"{mode}/evid_depth/alpha_min": alpha_min, f"{mode}/evid_depth/alpha_max": alpha_max,
        f"{mode}/evid_depth/beta_mean": beta_mean, #f"{mode}/evid_depth/beta_median": beta_median,f"{mode}/evid_depth/beta_dev":beta_dev, f"{mode}/evid_depth/beta_min": beta_min, f"{mode}/evid_depth/beta_max": beta_max,
        f"{mode}/evid_depth/gamma_mean": gamma_mean#, f"{mode}/evid_depth/gamma_median": gamma_median, f"{mode}/evid_depth/gamma_dev": gamma_dev, f"{mode}/evid_depth/gamma_min": gamma_min, f"{mode}/evid_depth/gamma_max": gamma_max
    })

    writer.add_scalars(f"{mode}/evid_depth", {"alpha_mean": alpha_mean, "beta_mean": beta_mean, "gamma_mean": gamma_mean}, global_step)
    wandb_run.log({
    "step": global_step,
    f"{mode}/d_unc/d_alea_unc_mean": d_alea_unc_mean, f"{mode}/d_unc/d_alea_unc_median": d_alea_unc_median, f"{mode}/d_unc/d_alea_unc_dev": d_alea_unc_dev, f"{mode}/d_unc/d_alea_unc_min": d_alea_unc_min, f"{mode}d_unc/d_alea_unc_max": d_alea_unc_max,
    f"{mode}/d_unc/d_epi_unc_mean": d_epi_unc_mean, f"{mode}/d_unc/d_epi_unc_median": d_epi_unc_median, f"{mode}/d_unc/d_epi_unc_dev": d_epi_unc_dev, f"{mode}/d_unc/d_epi_unc_min": d_epi_unc_min, f"{mode}/d_unc/d_epi_unc_max": d_epi_unc_max,
    f"{mode}/d_unc/d_total_unc_mean": d_total_unc_mean, f"{mode}/d_unc/d_total_unc_median": d_total_unc_median, f"{mode}/d_unc/d_total_unc_dev": d_total_unc_dev, f"{mode}/d_unc/d_total_unc_min": d_total_unc_min, f"{mode}/d_unc/d_total_unc_max": d_total_unc_max
    })
    writer.add_scalars(f"{mode}/d_unc", {"alea_unc_mean": d_alea_unc_mean, "d_epi_unc_mean": d_epi_unc_mean, "d_total_unc_mean": d_total_unc_mean}, global_step)

    wandb_run.log({
    "step": global_step,
    f"{mode}/seg_unc/seg_alea_unc_mean": seg_alea_unc_mean, #f"{mode}/seg_unc/seg_alea_unc_median": seg_alea_unc_median, f"{mode}/seg_unc/seg_alea_unc_dev": seg_alea_unc_dev, f"{mode}/seg_unc/seg_alea_unc_min": seg_alea_unc_min, f"{mode}/seg_unc/seg_alea_unc_max": seg_alea_unc_max,
    f"{mode}/seg_unc/seg_epi_unc_mean": seg_epi_unc_mean, #f"{mode}/seg_unc/seg_epi_unc_median": seg_epi_unc_median, f"{mode}/seg_unc/seg_epi_unc_dev": seg_epi_unc_dev, f"{mode}/seg_unc/seg_epi_unc_min": seg_epi_unc_min, f"{mode}/seg_unc/seg_epi_unc_max": seg_epi_unc_max,
    f"{mode}/seg_unc/seg_total_unc_mean": seg_total_unc_mean, #f"{mode}/seg_unc/seg_total_unc_median": seg_total_unc_median, f"{mode}/seg_unc/seg_total_unc_dev": seg_total_unc_dev, f"{mode}/seg_unc/seg_total_unc_min": seg_total_unc_min, f"{mode}/seg_unc/seg_total_unc_max": seg_total_unc_max,
    f"{mode}/seg_unc/seg_S_mean": seg_S_mean#, f"{mode}/seg_unc/seg_S_median": seg_S_median, f"{mode}/seg_unc/seg_S_dev": seg_S_dev, f"{mode}/seg_unc/seg_S_min": seg_S_min, f"{mode}/seg_unc/seg_S_max": seg_S_max
    })
    writer.add_scalars(f"{mode}/seg_unc", {"alea_unc_mean": seg_alea_unc_mean, "epi_unc_mean": seg_epi_unc_mean, "total_unc_mean": seg_total_unc_mean, "S_mean": seg_S_mean}, global_step)

    if mode == "val":
        depth_ece = compute_depth_ece(gt_depth, mu, d_total_unc, device=mu.device)
        depth_nll = compute_negative_log_likelihood(gt_depth, mu, d_total_unc)

        sigma2_alea = beta / (alpha - 1.0)
        sigma2_epi  = beta / (gamma * (alpha - 1.0))
        sigma2_pred = sigma2_alea + sigma2_epi 
        loss_var = torch.abs(torch.log(sigma2_pred + 1e-8) - torch.log((gt_depth - mu).pow(2) + 1e-6)).mean().item()
    
        wandb_run.log({
            "step": global_step,
            f"{mode}/depth/ece": depth_ece,
            f"{mode}/depth/nll": depth_nll,
            f"{mode}/depth/var_loss": loss_var,
        })

    #writer.add_scalars(f"{mode}/depth", {"ece": depth_ece, "nll": depth_nll, "var_loss": loss_var}, global_step)
