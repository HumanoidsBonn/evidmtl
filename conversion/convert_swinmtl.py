import os
import sys
import torch
from types import SimpleNamespace

# Ensure "swinmtl" root folder is in sys.path for correct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.torchscript_modelMulti import GLPDepth, TorchScriptGLPDepth  # Import model

# Manually set arguments
args = SimpleNamespace()
args.model_path = "../logs/2025-02-02_01-17-36_nyudepthv2_16_swinv2_small_1k_500k_deconv3_256_4_448_608_00005_3e-05_09_005_500_7_7_7_7_2_2_18_2_swinmtl_nyuv2_swin/final_model.ckpt"
args.output_folder = "../ckpt"
base_name = "best_vanilla_swinmtl_nyuv2labels13_small"

os.makedirs(args.output_folder, exist_ok=True)  # Ensure output folder exists
output_path = os.path.join(args.output_folder, f"{base_name}_scripted.pt")

# Define default training args since they are not in the checkpoint
default_args = SimpleNamespace(
    exp_name="swinmtl_nyuv2_swin", gpu_or_cpu="gpu", data_path="/home/user/menon/workspace/data/nyuv2_dataset",
    dataset="nyudepthv2", num_classes=14, batch_size=16, workers=6, max_depth=10.0, max_depth_eval=10.0, min_depth_eval=0.001,
    backbone="swin_small_v2", pretrained="/home/user/menon/workspace/software/swinmtl/model_zoo/checkpoints/swinv2_small_1k_500k.pth",
    window_size=[7, 7, 7, 7], pretrain_window_size=[23, 23, 23, 11], use_shift=[True, True, True, True], depths=[2, 2, 18, 2],
    drop_path_rate=0.3, use_checkpoint=False, num_deconv=3, num_filters=[256, 128, 64], deconv_kernels=[4, 4, 4],
    shift_window_test=True, shift_size=2, flip_test=True, epochs=500, max_lr=5e-4, min_lr=3e-5, weight_decay=0.05,
    layer_decay=0.9, crop_h=448, crop_w=608, log_dir="./logs", val_freq=1, pro_bar=False, save_freq=1,
    print_freq=100, save_model=True, auto_resume=True
)

# Load checkpoint
ckpt_dict = torch.load(args.model_path, map_location="cpu")
state_dict = ckpt_dict["model"]

# Initialize model with default args (same as training)
model = GLPDepth(default_args)

# Adjust checkpoint keys if they contain "module." (for DataParallel models)
state_dict = {key[len("module."):] if key.startswith("module.") else key: value for key, value in state_dict.items()}
model.load_state_dict(state_dict, strict=False)  # Load weights
model.eval()

# Wrap the model for TorchScript
scriptable_model = TorchScriptGLPDepth(model)

# Convert to TorchScript
example_input = torch.randn(1, 3, 448, 608)  # Example input
scripted_model = torch.jit.script(scriptable_model, example_input)
scripted_model.save(output_path)

print(f"TorchScript model saved to {output_path}")
