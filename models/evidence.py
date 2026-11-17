import torch
import torch.nn as nn
import torch.nn.functional as F

def relu_evidence(logits):
  """Applies ReLU to ensure non-negative evidence values."""
  return F.relu(logits)

def exp_evidence(logits):
  """Applies exponentiation with clipping to prevent overflow/underflow."""
  return torch.exp(torch.clamp(logits, min=-10, max=10))

def softplus_evidence(logits):
  """Applies softplus to ensure positive evidence values while avoiding zero outputs."""
  return F.softplus(logits)

def get_evidence_function(evidence_type):
  """Returns the selected evidence function or None for standard models."""
  if evidence_type == "relu":
    print("relu")
    return relu_evidence
  elif evidence_type == "exp":
    print("exp")
    return exp_evidence
  elif evidence_type == "softplus":
    print("softplus")
    return softplus_evidence
  elif evidence_type == "none":
    print("none")
    return lambda x: x  # No evidence function, use standard logits
  else:
    raise ValueError(f"Invalid evidence function: {evidence_type}. Choose from 'none', 'relu', 'exp', or 'softplus'.")

class NormalInvGammaConv(nn.Module):
  """Evidential Depth Head using Normal-Inverse Gamma (NIG) Distribution (CNN Version)."""

  def __init__(self, in_channels, out_channels=1, kernel_size=3, evidence_type="softplus"):
    """
    Args:
      in_channels: Number of input channels (from decoder).
      out_channels: Number of depth predictions per pixel.
      kernel_size: Convolutional kernel size (default: 3 for local spatial context).
    """
    super(NormalInvGammaConv, self).__init__()
    self.out_channels = out_channels
    self.dense = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    self.evidence_fn = get_evidence_function(evidence_type)
    #nn.init.constant_(self.dense.weight[:, 1 * out_channels:2 * out_channels], 2.0)  # Prior for logalpha
    #nn.init.constant_(self.dense.weight[:, 2 * out_channels:3 * out_channels], 0.1)  # Prior for logbeta
    #nn.init.constant_(self.dense.weight[:, 3 * out_channels:4 * out_channels], 1.0)  # Prior for loggamma
    
  def evidence(self, x):
    """Ensures strictly positive values using Softplus."""
    return self.evidence_fn(x)
  
  def forward(self, x):
    """
    Forward pass to compute Normal-Inverse Gamma (NIG) parameters.

    Args:
      x (Tensor): Feature map from decoder [B, in_channels, H, W].

    Returns:
      Tensor: Concatenated tensor containing [μ, α, β, γ] with shape [B, 4*out_channels, H, W].
    """
    out = self.dense(x)  # Shape: [B, 4*out_channels, H, W]
    assert out is not None, "Error: out_depth is None! Check if self.dense(x) is returning a tensor."
    mu, logalpha, logbeta, loggamma = torch.chunk(out, 4, dim=1)  # Split into 4 channels
    
    mu = torch.clamp(mu, min= 0, max=10)  # Ensure μ ∈ [0, 11]
    alpha = self.evidence(logalpha) + 1  # Ensure α > 1
    beta = torch.clamp(self.evidence(logbeta), min=0.000001)  # Ensure β > 0
    gamma = self.evidence(loggamma) + 0.00001 # γ (renamed precision `v`)
    gamma = torch.clamp(gamma, max=100.0)
    return torch.cat([mu, alpha, beta, gamma], dim=1)
