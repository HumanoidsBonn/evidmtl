# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch
import numpy as np

def compute_depth_ece(y_true, y_pred, uncertainty, device, n_bins=10):
    """
    Compute Depth Expected Calibration Error (Depth-ECE) using PyTorch.

    Parameters:
    y_true: torch.Tensor
        Ground truth depth values (N,)
    y_pred: torch.Tensor
        Predicted depth values (N,)
    aleatoric_uncertainty: torch.Tensor
        Predicted aleatoric uncertainty (variance) (N,)
    n_bins: int
        Number of bins for calibration

    Returns:
    float
        Depth-ECE score
    """
    # Compute absolute depth errors
    depth_errors = torch.abs(y_true - y_pred)/10.0
    #print("depth_errors shape: ", depth_errors.shape)
    #print("uncertainty shape: ", uncertainty.shape)
    
    unc_sqrt = torch.sqrt(uncertainty)/10.0

    depth_errors_flat = depth_errors.flatten()
    uncertainty_flat = unc_sqrt.flatten()
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    #print("depth_errors_flat shape: ", depth_errors_flat.shape)
    #print("uncertainty shape: ", uncertainty_flat.shape)   

    # Manually sort based on aleatoric uncertainty
    sorted_data = sorted(zip(uncertainty_flat.tolist(), y_true_flat.tolist(), y_pred_flat.tolist(), depth_errors_flat.tolist()))
    sorted_uncertainty, y_true_sorted, y_pred_sorted, errors_sorted = zip(*sorted_data)

    # Convert back to tensors
    uncertainty_sorted = torch.tensor(sorted_uncertainty)
    y_true_sorted = torch.tensor(y_true_sorted)
    y_pred_sorted = torch.tensor(y_pred_sorted)
    errors_sorted = torch.tensor(errors_sorted)

    # Split into bins
    bin_size = len(y_true_flat) // n_bins
    depth_ece = 0.0

    check_nan_inf(uncertainty_sorted, "uncertainty_sorted")
    check_nan_inf(errors_sorted, "errors_sorted")

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true_flat)
        length = end - start
        #print(f"Bin {i}: {length} samples")
        if length == 0:
            continue
        # Mean aleatoric uncertainty and mean depth error per bin
        mean_uncertainty = torch.mean(uncertainty_sorted[start:end])
        mean_error = torch.mean(errors_sorted[start:end])#[~torch.isnan(uncertainty_sorted[start:end])])


        # Compute weighted absolute difference
        bin_weight = (end - start) / len(y_true_flat)
        #print(f"Bin {i}: Uncertainty={mean_uncertainty:.4f}, Error={mean_error:.4f}, bin_weight={bin_weight:.4f}")
        #delta = torch.abs(mean_uncertainty - mean_error)
    
        depth_ece += bin_weight * torch.abs(mean_uncertainty - mean_error)

    return {"depth_ece": (depth_ece).item()}


def compute_negative_log_likelihood(y_true, y_pred, uncertainty):
    """
    Compute Negative Log-Likelihood (NLL) for depth regression using PyTorch.

    Parameters:
    y_true: torch.Tensor
        Ground truth depth values (N,)
    y_pred: torch.Tensor
        Predicted depth values (N,)
    uncertaint: torch.Tensor
        Predicted uncertainty (variance) (N,)

    Returns:
    float
        NLL score
    """
    epsilon = 1e-6  # To prevent division by zero
    variance = uncertainty + epsilon
    log_likelihood = 0.5 * torch.log(2 * torch.pi * variance) + ((y_true - y_pred) ** 2) / (2 * variance)
    return {"depth_nll": torch.mean(log_likelihood).item()}

def check_nan_inf(tensor, name=""):
    if torch.isnan(tensor).any():
        print(f"❌ NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"❌ Inf detected in {name}")

def eval_depth(pred, target):
    assert pred.shape == target.shape
    #torch.clamp(pred, min=1e-4, max=10)
    #torch.clamp(target, min=1e-4, max=10)
    check_nan_inf(pred, "pred")
    check_nan_inf(target, "target")

    thresh = torch.max((target / pred), (pred / target))
    check_nan_inf(thresh, "thresh")

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    check_nan_inf(d1, "d1")
    check_nan_inf(d2, "d2") 
    check_nan_inf (d3, "d3")

    diff = pred - target
    check_nan_inf(diff, "diff")
    diff_log = torch.log(pred) - torch.log(target)
    check_nan_inf(diff_log, "diff_log")

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    check_nan_inf(abs_rel, "abs_rel")
    check_nan_inf(sq_rel, "sq_rel") 

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))
    check_nan_inf(rmse, "rmse")
    check_nan_inf(rmse_log, "rmse_log")

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))
    check_nan_inf(log10, "log10")
    check_nan_inf(silog, "silog")

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}

def eval_evid_seg(pred, target, num_classes=41, num_bins=10):
  """
  Evaluates evidential semantic segmentation by computing multiple uncertainty-aware metrics.

  Args:
    pred: Dict containing model outputs with key 'pred_seg' as Dirichlet parameters (evidence + 1).
    target: Tensor (N, H, W) - Ground truth labels.
    num_classes: int - Number of segmentation classes.
    num_bins: int - Number of bins for ECE computation.

  Returns:
    results: Dict with evidential segmentation metrics.
  """
  N, C, H, W = pred.shape
  pred = pred.view(N, C, -1).permute(0, 2, 1)  # Reshape to (N, HW, C)
  target = target.view(N, -1).to(torch.int64)  # Flatten target to (N, HW)

  # Compute total evidence and expected probability
  S = torch.sum(pred, dim=2, keepdim=True)  # Total evidence per pixel
  p_hat = pred / S  # Expected class probabilities

  # One-hot encode target labels
  target_one_hot = torch.nn.functional.one_hot(target, num_classes).float()
  #target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).to(pred.device)

  # Compute Brier Score
  brier_score = torch.mean(torch.sum((p_hat - target_one_hot) ** 2, dim=2))

  # Compute Top-2 Accuracy
  top2_preds = torch.topk(p_hat, 2, dim=2).indices  # Get top-2 class predictions
  top2_correct = torch.any(top2_preds == target.unsqueeze(2), dim=2).float()
  top2_acc = top2_correct.mean()

  # Compute Evidential Soft mIoU
  miou = 0.0
  for j in range(num_classes):
    y_mask = (target == j)

    soft_tp = torch.sum(p_hat[:, :, j] * y_mask)
    soft_fn = torch.sum((1 - p_hat[:, :, j]) * y_mask)
    soft_fp = torch.sum(p_hat[:, :, j] * (~y_mask))

    iou_j = soft_tp / (soft_tp + soft_fn + soft_fp + 1e-6)  # Avoid div by zero
    miou += iou_j

  evid_miou = miou / num_classes

  # Compute Expected Calibration Error (ECE)
  conf, preds = torch.max(p_hat, dim=2)  # Confidence and predicted class
  correct = (preds == target).float()  # Binary correctness
  bin_boundaries = torch.linspace(0, 1, num_bins + 1)  # Binning edges
  ece = torch.tensor(0.0, device=pred.device)

  for i in range(num_bins):
    bin_mask = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
    if bin_mask.any():
      bin_acc = correct[bin_mask].mean()
      bin_conf = conf[bin_mask].mean()
      ece += bin_mask.float().mean() * torch.abs(bin_acc - bin_conf)  # Weighted sum

  # Return all metrics in a dictionary
  return {
    'brier': brier_score.item(),
    'ev_mIOU': evid_miou.item(),
    'top2_acc': top2_acc.item(),
    'ece': ece.item()
}
def safe_item(val):
    return val.item() if isinstance(val, torch.Tensor) else val

def eval_seg(pred, target, num_classes=14, iteration=0):
    assert pred.shape == target.shape

    # mIoU pixel_acc class_acc
    pred=torch.flatten(pred).to(torch.int64)
    target=torch.flatten(target).to(torch.int64)

    
    valid_mask = (target != 255) & (target != -1) #remaining : 0-18
    pred = pred[valid_mask]
    target = target[valid_mask]

    #compute miou
    mIoU=0
    iou = 0
    class_iou = {}
    #nyuv2_13_classes = ["Background", "Wall", "Floor", "Cabinet", "Bed", "Chair", "Sofa", "Table", "Door", "Window", "Bookshelf", "Picture", "Counter", "Ceiling"]
    #assert len(nyuv2_13_classes) == num_classes
    #conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32).to(pred.device)
    for i in range(num_classes):
        #if i in target
        if i not in target:
            continue
        
        TP=torch.sum((pred==i)&(target==i))
        FP=torch.sum((pred==i)&(target!=i))
        FN=torch.sum((pred!=i)&(target==i))
        iou = TP/(TP+FP+FN+1e-6)
        mIoU+=iou
        #class_iou[nyuv2_13_classes[i]] = iou.item()

    class_count_target=len(torch.unique(target))
    mIoU=mIoU/class_count_target

    # pixel_acc
    pixel_acc = torch.sum(pred == target) / len(target)
    
    # if iteration%8 == 0:
    #     print("per class iou:")
    #     print(" | ".join([f"{key}: {value:.4f}" for key, value in class_iou.items()]))


    # Compute Confusion Matrix
    # for t, p in zip(target, pred):
    #     if t < class_count_target and p < num_classes:  # Ignore invalid class indices
    #         conf_matrix[t, p] += 1

    # # Convert to fractional form by normalizing each row
    # row_sums = conf_matrix.sum(dim=1, keepdim=True)  # Sum over predicted classes per true class
    # conf_matrix_frac = conf_matrix / (row_sums + 1e-6)  # Avoid division by zero
    
    
    #print_confusion_matrix(conf_matrix_frac, class_labels=nyuv2_13_classes)

    #class_acc = torch.mean(class_acc)
    return {'mIoU': safe_item(mIoU), 'pixel_acc': safe_item(pixel_acc)}#, 'class_iou': class_iou}#, 'class_acc': class_acc.item()}

def print_confusion_matrix(conf_matrix, class_labels=None, truncate_length=5):
    """
    Prints a confusion matrix in a well-formatted way with aligned columns.

    Args:
        conf_matrix (torch.Tensor): Confusion matrix (num_classes x num_classes).
        class_labels (list, optional): List of class names. Defaults to index numbers.
        truncate_length (int, optional): Max length of labels for alignment.
    """
    num_classes = conf_matrix.shape[0]
    conf_matrix = conf_matrix.cpu().numpy()  # Convert to NumPy for better formatting

    # If no class labels are provided, use numbers
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]
    else:
        # Truncate long labels to maintain formatting
        class_labels = [label[:truncate_length] for label in class_labels]

    # Determine the column width (max of longest label or max number in conf matrix)
    max_label_length = max(len(label) for label in class_labels)
    max_number_length = max(len(str(int(conf_matrix.max()))), 5)  # Ensure at least 5 spaces
    col_width = max(max_label_length, max_number_length) + 2  # Add padding

    # Print header row
    header = ["True \ Pred".ljust(col_width)] + [label.ljust(col_width) for label in class_labels]
    print("\n" + " | ".join(header))
    print("-" * len(" | ".join(header)))

    # Print each row
    for i in range(num_classes):
        row_label = class_labels[i].ljust(col_width)
        row_values = [f"{int(conf_matrix[i, j]):>{col_width}}" for j in range(num_classes)]
        print(" | ".join([row_label] + row_values))


def cropping_img(args, pred, gt_depth):
    #print("dataset name: ", args.dataset)
    min_depth_eval = args.min_depth_eval
    #print("pred shape:", pred.shape)
    #print("gt_depth shape:", gt_depth.shape)
    max_depth_eval = args.max_depth_eval
    
    check_nan_inf(pred, "pred")
    check_nan_inf(gt_depth, "gt_depth")
    pred[torch.isinf(pred)] = min_depth_eval
    pred[torch.isnan(pred)] = max_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    
    #print("gt depth min and max ", torch.min(gt_depth), torch.max(gt_depth))
    #check if valid mask has some True values
    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2' or args.dataset == "nyudepthv2labels40" or args.dataset == 'nyudepthv2labels20scannet':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        #print("nyudepthv2 eval_mask shape: ", eval_mask.shape)
        if gt_depth.shape.__len__() == 3:
            h_min, h_max = 0, min(472, gt_depth.shape[1])  # Ensure within valid height
            w_min, w_max = 0, min(632, gt_depth.shape[2])  # Ensure within valid width
        if gt_depth.shape.__len__() == 2:
            h_min, h_max = 0, min(472, gt_depth.shape[0])
            w_min, w_max = 0, min(632, gt_depth.shape[1])
        if h_min >= h_max or w_min >= w_max:
            print("🚨 Error: NYU crop dimensions are invalid! Skipping crop.")
            eval_mask[:] = 1  # Use the full valid mask instead
        else:
            if gt_depth.shape.__len__() == 3:
                eval_mask[:, h_min:h_max, w_min:w_max] = 1
            if gt_depth.shape.__len__() == 2:
                eval_mask[h_min:h_max, w_min:w_max] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)
    #print(f"Final valid_mask sum: {valid_mask.sum()} / {valid_mask.numel()} ({valid_mask.sum().float() / valid_mask.numel() * 100:.2f}% valid pixels)")

    if valid_mask.sum() == 0:
        print("🚨 Warning: No valid pixels remain after cropping!")
        print(f"Final eval_mask sum: {eval_mask.sum()} / {eval_mask.numel()} ({eval_mask.sum().float() / eval_mask.numel() * 100:.2f}% valid pixels)")
        print(f"Final valid_mask sum: {valid_mask.sum()} / {valid_mask.numel()} ({valid_mask.sum().float() / valid_mask.numel() * 100:.2f}% valid pixels)")


    return pred[valid_mask], gt_depth[valid_mask]
