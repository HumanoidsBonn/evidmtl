# ------------------------------------------------------------------------------
# The original code is from GLPDepth (https://github.com/vinvino02/GLPDepth)
# moddified by Pardis Taghavi(taghavi.pardis@gmail.com)
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name}")

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        loss = nn.CrossEntropyLoss()(pred, target)
        return loss
    
class SmoothLoss(nn.Module): 
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, depth, img):
        if len(depth.shape)==3:
            depth_=depth.unsqueeze(1)
        elif len(depth.shape)==4:
            depth_=depth
        elif len(depth.shape)==2:
            depth_=depth.unsqueeze(0).unsqueeze(0)
        
        grad_depth_x = torch.abs(depth_[:, :, :, :-1] - depth_[:, :, :, 1:])
        grad_depth_y = torch.abs(depth_[:, :, :-1, :] - depth_[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y= torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()

# ------------------------------------------------------------------------------
# The following code is written by Rohit Menon.
# The EvidentialClassificationLoss is based on Sensoy et al's work "Evidential Deep Learning to Quantify Classification Uncertainty".
# ------------------------------------------------------------------------------
l2_loss = torch.nn.MSELoss(reduction = 'none')

class EvidentialClassificationLoss(nn.Module):
    def __init__(self, kl_coeff=0.2, lambda_dice=1.0, lambda_ce=1.0, total_epochs=500, exp_weight=5):
        super(EvidentialClassificationLoss, self).__init__()
        self.kl_coeff = kl_coeff
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.total_epochs = total_epochs
        self.exp_weight = exp_weight
        print("EvidentialSemanticLoss initialised")

    def evidential_dice_loss(self, alpha, gt):
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S  # Expected probabilities

        n_classes = alpha.shape[1]
        one_hot_gt = F.one_hot(gt.long(), num_classes=n_classes).permute([0,3,1,2]).to(alpha.device)

        # Compute Dice loss
        intersection = torch.sum(one_hot_gt * probs, dim=(1, 2))
        total = torch.sum(one_hot_gt, dim=(1, 2)) + torch.sum(probs, dim=(1, 2))
        dice_loss = 1 - (2 * intersection + 1e-6) / (total + 1e-6)
        return torch.mean(dice_loss)

    def forward(self, alpha, gt, curr_epoch=None):
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha / S
        n_classes = alpha.shape[1]
        inv_prob = 1 - prob
        one_hot_gt = torch.nn.functional.one_hot(gt.long(), num_classes=n_classes).permute([0,3,1,2]).to(alpha.device)
        alpha_tilde = one_hot_gt + (1-one_hot_gt)*alpha
        alpha_tilde = alpha_tilde.permute([0,2,3,1])
        l1 = torch.sum(l2_loss(prob.float(),one_hot_gt.float()),dim =1)
        l2 = torch.sum(prob * inv_prob / (S + 1), dim=1)
        dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(n_classes).to(device=alpha.device).unsqueeze(0).unsqueeze(0).unsqueeze(0),validate_args = True)
        dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde,validate_args = True)
        reg_loss = torch.distributions.kl.kl_divergence(dir_pred, dir_null)

        dice_loss = self.evidential_dice_loss(alpha, gt)

        if curr_epoch is None:
            return self.lambda_ce*torch.mean(l1 + l2) + self.lambda_dice*dice_loss + self.kl_coeff * torch.mean(reg_loss)
        else:
            return self.lambda_ce*torch.mean(l1 + l2) + self.kl_coeff * (1 - math.exp(-self.exp_weight * curr_epoch / self.total_epochs)) * reg_loss

        #background_penalty = torch.sum((one_hot_gt[:, 0, :, :] - prob[:, 2, :, :]) ** 2)
        #final_loss += self.k * background_penalty


class NIG_NLL(nn.Module):
    """Computes Negative Log-Likelihood loss for the Normal-Inverse Gamma (NIG) distribution."""
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, y_true):
        """
        Args:
            y_true (Tensor): Ground truth depth values. Shape: [B, H, W]
            mu (Tensor): Predicted mean depth. Shape: [B, H, W]
            alpha (Tensor): Predicted shape parameter. Shape: [B, H, W]
            beta (Tensor): Predicted scale parameter. Shape: [B, H, W]
            gamma (Tensor): Predicted precision parameter. Shape: [B, H, W]
        
        Returns:
            Tensor: NIG negative log-likelihood loss.
        """
        mu, alpha, beta, gamma = torch.chunk(pred, 4, dim=1)  # Extract components
        twoBlambda = 2 * beta * (1 + gamma)
        twoBlambda = torch.clamp(twoBlambda, min=0.0001)
        
        gamma = torch.clamp(gamma, min=0.0001)
        beta = torch.clamp(beta, min=0.0001)

        # nll = (
        #     0.5 * torch.log(torch.pi / gamma) 
        #     - alpha * torch.log(twoBlambda)  
        #     + (alpha + 0.5) * torch.log(gamma * (y_true - mu) ** 2 + twoBlambda)  
        #     + torch.lgamma(alpha)  
        #     - torch.lgamma(alpha + 0.5)
        # )
        nll = (
        0.5 * (torch.log(torch.tensor(torch.pi)) - torch.log(gamma))  # log(pi) - log(gamma)
        - alpha * torch.log(twoBlambda)  
        + (alpha + 0.5) * torch.log(gamma * (y_true - mu).pow(2) + twoBlambda)  
        + torch.lgamma(alpha)  
        - torch.lgamma(alpha + 0.5)
        )
        
        # nll = (
        # torch.lgamma(alpha - 0.5)  # log(Γ(α - 0.5))
        # - torch.log(torch.tensor(4.0, device=alpha.device))  # -log(4)
        # - torch.lgamma(alpha)  # -log(Γ(α))
        # - torch.log(gamma)  # -log(γ)
        # - 0.5 * torch.log(beta)  # -0.5 log(β)
        # + torch.log(2 * beta * (1 + gamma) + (2 * alpha - 1) * gamma * (y_true - mu).pow(2))  # log(2β(1 + γ) + (2α - 1)γ(y - μ)^2)
        # )

        return (nll).mean()

class NIG_Reg(nn.Module):
    """Computes Regularization for Normal-Inverse Gamma (NIG) predictions using SiLogLoss."""
    def __init__(self, omega=0.01, coeff=0.01, kl=False, reg =False, enable_mu_prior = False, total_epochs=500):
        """
        Args:
            omega (float): Regularization factor (default: 0.01).
        """
        super().__init__()
        self.kl = kl
        self.reg = reg
        self.omega = omega
        self.coeff = coeff
        self.total_epochs = total_epochs
        self.enable_mu_prior = enable_mu_prior
        
        if self.reg == True and self.kl == False:
            print("Evidential Regularization")
        if self.reg == False and self.kl == True:
            print("KL Regularization")      
        if self.reg == True and self.kl == True:
            print("Evidential Regularization + KL Regularization")
            
        if self.enable_mu_prior:
            print("Using mu prior")
        else:
            print("Using mu prediction itself as prior")

    def forward(self, pred, y_true, curr_epoch=None):
        """
        Args:
            y_true (Tensor): Ground truth depth values. Shape: [B, H, W]
            mu (Tensor): Predicted mean depth. Shape: [B, H, W]
            alpha (Tensor): Predicted shape parameter. Shape: [B, H, W]
            beta (Tensor): Predicted scale parameter. Shape: [B, H, W]
            gamma (Tensor): Predicted precision parameter. Shape: [B, H, W]
            kl (bool): If True, applies KL divergence regularization.

        Returns:
            Tensor: Regularization loss.
        """
        mu, alpha, beta, gamma = torch.chunk(pred, 4, dim=1)  # Extract components
        mu = mu.detach() 
        error = torch.abs(mu.squeeze(1) - y_true)  # Compute SiLog Loss
        sigma2_alea = beta / (alpha - 1.0)
        sigma2_epi  = beta / (gamma * (alpha - 1.0))
        sigma2_pred = sigma2_alea + sigma2_epi 
        sq_error = (y_true - mu).square() 
        loss_var = torch.abs(torch.log(sigma2_pred + 1e-8) - torch.log(sq_error + 1e-6)).mean()

        
        if self.kl == False and self.reg == False:
            return torch.zeros_like(error).mean()

        if self.kl == True and self.reg ==False:
            # KL divergence-based regularization
            #alpha_prior = torch.ones_like(mu) * (1.0 + self.omega) 
            #beta_prior = torch.ones_like(mu) * 0.1
            #gamma_prior = torch.ones_like(mu) * 1.0
            alpha_prior = torch.ones_like(mu) * 1.01
            beta_prior =  beta
            gamma_prior = torch.ones_like(mu) * 0.001

            if self.enable_mu_prior == True:
                #weighting_factor = curr_epoch/(0.6*self.total_epochs)
                noise_std = 0.5 
                std_prior = noise_std**torch.randn_like(y_true)
                mu_prior = y_true + std_prior   # Add depth-dependent Gaussian noise (std=0.1 * depth)
                var_prior = std_prior**2
                beta_prior = var_prior*(alpha_prior-1)/(1+ 1/gamma_prior)  # Set beta prior based on noise std
                kl_div = (
                0.5 * (alpha / beta) * ((mu - mu_prior) ** 2) * gamma_prior  # Penalizes mean deviation
                + 0.5 * gamma_prior / gamma  # Penalizes precision shift
                - 0.5  # Constant shift
                + alpha_prior * torch.log(beta / beta_prior)  # Scale ratio regularization
                - (torch.lgamma(alpha) - torch.lgamma(alpha_prior))  # Penalizes gamma function shift
                + (alpha - alpha_prior) * torch.digamma(alpha)  # Penalizes confidence drift
                - (beta - beta_prior) * (alpha / beta)  # Penalizes scale shift
                )
                reg = error * kl_div + 0.1*loss_var
            else:
                mu_prior = mu
                #noise_std = 0.5 
                #std_prior = noise_std**torch.randn_like(y_true)
                #var_prior = std_prior**2
                #beta_prior = var_prior*(alpha_prior-1)/(1+ 1/gamma_prior)
                kl_div = (
                0.5 * (alpha / beta) * ((mu - mu_prior) ** 2) * gamma_prior  # Penalizes mean deviation
                + 0.5 * gamma_prior / gamma  # Penalizes precision shift
                - 0.5  # Constant shift
                + alpha_prior * torch.log(beta / beta_prior)  # Scale ratio regularization
                - (torch.lgamma(alpha) - torch.lgamma(alpha_prior))  # Penalizes gamma function shift
                + (alpha - alpha_prior) * torch.digamma(alpha)  # Penalizes confidence drift
                - (beta - beta_prior) * (alpha / beta)  # Penalizes scale shift
                )
                reg = error * kl_div + 0.1*loss_var
            
        if self.reg == True and self.kl == False:
            # Standard evidential regularization
            evidence = 2 * gamma + alpha
            reg = error * evidence + 0.1*loss_var
            
        if self.reg == True and self.kl == True:
            alpha_prior = torch.ones_like(mu) * (1.01) 
            beta_prior =  beta
            gamma_prior = torch.ones_like(mu) * 0.001
            
            if self.enable_mu_prior == True:
                #weighting_factor = curr_epoch/(0.6*self.total_epochs)
                noise_std = 0.5 
                std_prior = noise_std**torch.randn_like(y_true)
                mu_prior = y_true + std_prior   # Add depth-dependent Gaussian noise (std=0.1 * depth)
                var_prior = std_prior**2
                beta_prior = var_prior*(alpha_prior-1)/(1+ 1/gamma_prior)  # Set beta prior based on noise std
                kl_div = (
                0.5 * (alpha / beta) * ((mu - mu_prior) ** 2) * gamma_prior  # Penalizes mean deviation
                + 0.5 * gamma_prior / gamma  # Penalizes precision shift
                - 0.5  # Constant shift
                + alpha_prior * torch.log(beta / beta_prior)  # Scale ratio regularization
                - (torch.lgamma(alpha) - torch.lgamma(alpha_prior))  # Penalizes gamma function shift
                + (alpha - alpha_prior) * torch.digamma(alpha)  # Penalizes confidence drift
                - (beta - beta_prior) * (alpha / beta)  # Penalizes scale shift
                )
                kl_loss = error * kl_div + 0.1*loss_var
            else:
                mu_prior = mu
                #noise_std = 0.5 
                #std_prior = noise_std**torch.randn_like(y_true)
                #var_prior = std_prior**2
                #beta_prior = var_prior*(alpha_prior-1)/(1+ 1/gamma_prior)
                kl_div = (
                0.5 * (alpha / beta) * ((mu - mu_prior) ** 2) * gamma_prior  # Penalizes mean deviation
                + 0.5 * gamma_prior / gamma  # Penalizes precision shift
                - 0.5  # Constant shift
                + alpha_prior * torch.log(beta / beta_prior)  # Scale ratio regularization
                - (torch.lgamma(alpha) - torch.lgamma(alpha_prior))  # Penalizes gamma function shift
                + (alpha - alpha_prior) * torch.digamma(alpha)  # Penalizes confidence drift
                - (beta - beta_prior) * (alpha / beta)  # Penalizes scale shift
                )
                kl_loss = error * kl_div 
            
            evidence = 2 * gamma + alpha
            reg = error * evidence + kl_loss
            
        if curr_epoch is None:
            return self.coeff * reg.mean()
        else:
            # if curr_epoch < 2:
            #     return torch.zeros_like(reg).mean()
            # else:
            return self.coeff * min(1.0, math.pow(curr_epoch/(0.7*self.total_epochs), 1))* reg.mean()


    
class EvidentialSiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, uncertainty_weight=0.01):
        super().__init__()
        self.lambd = lambd
        self.uncertainty_weight = uncertainty_weight
        self.aleatoric_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        mu, alpha, beta, gamma = torch.chunk(pred, 4, dim=1)
        mu = mu.squeeze(1)
        alpha = alpha.squeeze(1)
        beta = beta.squeeze(1)
        valid_mask = (target > 0).detach()
        sigma2_alea = torch.clamp(beta / (alpha - 1 + 1e-6), min=1e-6)
        log_diff = torch.log(target[valid_mask]) - torch.log(mu[valid_mask])
        silog_loss = torch.sqrt(log_diff.pow(2).mean() - self.lambd * log_diff.mean().pow(2))
        uncertainty_reg = self.aleatoric_loss(sigma2_alea, torch.abs(target - mu))
        sigma2_epi  = beta / (gamma * (alpha - 1.0))
        sigma2_pred = sigma2_alea + sigma2_epi
        loss_var = torch.abs(
              torch.log(sigma2_pred + 1e-8) -
              torch.log((target - mu).pow(2) + 1e-6)
          ).mean()
        return silog_loss 
   
class EvidentialCrossEntropyLoss(nn.Module):
    def __init__(self, kl_coeff=0.1, exp_weight = 5, total_epochs=500, focal=False, gamma=2, dice=False):
        super().__init__()
        self.total_epochs = total_epochs
        self.kl_coeff = kl_coeff
        self.exp_weight = exp_weight
        self.focal = focal
        self.gamma = gamma
        self.dice = dice

    def evidential_dice_loss(self, alpha, gt):
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S  # Expected probabilities

        n_classes = alpha.shape[1]
        one_hot_gt = F.one_hot(gt.long(), num_classes=n_classes).permute([0,3,1,2]).to(alpha.device)

        # Compute Dice loss
        intersection = torch.sum(one_hot_gt * probs, dim=(1, 2))
        total = torch.sum(one_hot_gt, dim=(1, 2)) + torch.sum(probs, dim=(1, 2))
        dice_loss = 1 - (2 * intersection + 1e-6) / (total + 1e-6)
        return torch.mean(dice_loss)

    def forward(self, pred, target, curr_epoch = None):
        alpha = torch.clamp(pred, min=1.0)
        num_classes = alpha.shape[1]
        S = torch.sum(alpha, dim=1, keepdim=True)  # Shape: (batch_size, 1, H, W)
        one_hot_gt = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(alpha.device)
        evid_ce_loss = torch.sum(one_hot_gt * (torch.log(S) - torch.log(alpha)), dim=1)  # Sum over classes

        if self.focal == False and self.dice == False:
            return torch.mean(evid_ce_loss)
        elif self.focal == True:
            expected_prob = alpha / S  # Shape: (batch_size, num_classes, H, W)
            p_true = torch.sum(one_hot_gt * expected_prob, dim=1)  # Shape: (batch_size, H, W)
            focal_term = (1 - p_true) ** self.gamma  # Shape: (batch_size, H, W)
            focal_evid_ce_loss = focal_term * evid_ce_loss
            return torch.mean(focal_evid_ce_loss)
        elif self.dice == True:
            dice_loss = self.evidential_dice_loss(alpha, target)
            return 0.5*(dice_loss + torch.mean(evid_ce_loss))
        else:
            raise ValueError("Invalid loss type")
        
            
class EvidentialKLDivergenceLoss(nn.Module):
    def __init__(self, kl_coeff=0.1, exp_weight = 5, total_epochs=500):
        super().__init__()
        self.total_epochs = total_epochs
        self.kl_coeff = kl_coeff
        self.exp_weight = exp_weight

    def forward(self, pred, target, curr_epoch = None):
        alpha = torch.clamp(pred, min=1.0)
        num_classes = alpha.shape[1]
        S = torch.sum(alpha, dim=1, keepdim=True)  # Shape: (batch_size, 1, H, W)
        one_hot_gt = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(alpha.device)
        alpha_tilde = one_hot_gt + (1 - one_hot_gt) * alpha
        alpha_tilde = alpha_tilde.permute(0, 2, 3, 1) 
        dir_null = torch.distributions.dirichlet.Dirichlet(torch.ones(num_classes, device=alpha.device).unsqueeze(0).unsqueeze(0).unsqueeze(0), validate_args=True)
        dir_pred = torch.distributions.dirichlet.Dirichlet(alpha_tilde, validate_args=True)
        reg_loss = torch.distributions.kl.kl_divergence(dir_pred, dir_null).mean()

        if curr_epoch is None:
            return self.kl_coeff * reg_loss#
        else:
            return self.kl_coeff * min(1, curr_epoch / (0.7*self.total_epochs)) * reg_loss
        
class CombinedSiLogNLLLoss(nn.Module):
    def __init__(self, lambd=0.5, uncertainty_weight=0.01, lambda_nll=0.1):
        super().__init__()
        self.evi_silog_loss = EvidentialSiLogLoss(lambd, uncertainty_weight)
        self.nig_nll_loss = NIG_NLL()
        self.lambda_nll = lambda_nll

    def forward(self, pred, target):
        return self.evi_silog_loss(pred, target) + self.lambda_nll * self.nig_nll_loss(pred, target)    
