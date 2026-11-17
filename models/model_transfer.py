import torch
import torch.nn as nn
#from modelMulti import GLPDepth as VanillaModel

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


def transfer_weights(ckpt, evidential_model):
    """
    Loads a vanilla model checkpoint and transfers weights to an evidential model.
    Ignores the last layer and initializes the evidential head randomly.

    Args:
        ckpt (str): Path to vanilla model checkpoint.
        evidential_model (torch.nn.Module): Evidential model to initialize.
        vanilla_model_class (torch.nn.Module): Class definition of the vanilla model.

    Returns:
        evidential_model (torch.nn.Module): Model with transferred weights.
    """
    print(f"🔄 Loading Vanilla Model from: {ckpt}")

    # Initialize vanilla model

    # Load checkpoint
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    vanilla_state_dict = ckpt_dict['model']

    # Remove "module." prefix if trained with DataParallel
    weights = {}
    for key, value in vanilla_state_dict.items():
        new_key = key[len('module.'):] if key.startswith('module.') else key
        weights[new_key] = value

    # Get evidential model's state_dict
    evidential_state_dict = evidential_model.state_dict()

    # Identify last layer keys (assumed to be named 'last_layer' in vanilla model)
    last_layer_keys = [key for key in weights.keys() if "last_layer" in key]
    for key in last_layer_keys:
        print(f"⚠️ Skipping last layer: {key}")
        del weights[key]  # Remove last layer from transfer

    # Load transferred weights into evidential model
    evidential_model.load_state_dict(weights, strict=False)  # `strict=False` allows missing layers

    # Reinitialize evidential head (`μ, α, β, γ`)
    for key in evidential_state_dict.keys():
        if "last_layer" in key:  # Adjust this based on evidential model's last layer name
            print(f"🔄 Reinitializing {key} in the evidential model...")
            if evidential_state_dict[key].dim() >= 2:  # Ensure tensor has at least 2 dimensions
              torch.nn.init.kaiming_normal_(evidential_state_dict[key])  # Use Kaiming initialization for weights
            else:
              torch.nn.init.zeros_(evidential_state_dict[key])  # Use zeros for biases or scalars


    print("✅ Successfully transferred weights from Vanilla to Evidential Model (except last layer).")
    # 🔍 Validation: Compare Transferred Weights
    for key in weights.keys():
        if key in evidential_state_dict:
            vanilla_w = weights[key].to(torch.float32)  # Ensure float32
            evidential_w = evidential_state_dict[key].to(torch.float32)  # Ensure float32

            diff = torch.norm(vanilla_w - evidential_w, p=2).item()  # L2 Norm Difference
            print(f"  🔹 {key}: L2 norm difference = {diff:.6f}")


    print("✅ Weight transfer validation complete.\n")
    return evidential_model  # Return the modified evidential model
