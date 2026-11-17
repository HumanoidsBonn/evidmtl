import os
import numpy as np
import torch
from PIL import Image
from dataset.base_dataset import BaseDataset
import cv2
from tqdm import tqdm  # Progress bar for large datasets
import pandas as pd
import random


# One-time label check
max_label = 0  # Track highest class index




class scannet_v2(BaseDataset):
  def __init__(self, data_path, split='train', dmin=1e-4, dmax=10.0, crop_size=(640, 480), scale_size=None):
    super().__init__(crop_size)  # Initialize BaseDataset
    self.data_path = data_path  # Base dataset path
    self.split = split  # Use direct split (no renaming)
    self.scale_size = scale_size
    self.dmin = dmin
    self.dmax = dmax
    self.nyu40toscannet20_mapping_csv = os.path.join(data_path, "semantic_label_mapping.csv")
    self.nyu40toscannet20_mapping =  pd.read_csv(self.nyu40toscannet20_mapping_csv)
    self.mapping_dict = dict(zip(self.nyu40toscannet20_mapping["nyu40"], self.nyu40toscannet20_mapping["scannet_20"]))

    # Read train/val scene lists
    scene_list_file = os.path.join(self.data_path, f"{self.split}.txt")
    if not os.path.exists(scene_list_file):
      raise FileNotFoundError(f"Scene list file {scene_list_file} not found.")

    with open(scene_list_file, "r") as f:
      self.scenes = f.read().splitlines()

    # Dictionary to store file paths with pseudo filenames
    self.file_dict = {}

    for scene in self.scenes:
      scene_path = os.path.join(self.data_path, scene)

      # Ensure scene directories exist
      color_dir = os.path.join(scene_path, "color")
      label_dir = os.path.join(scene_path, "label")
      depth_dir = os.path.join(scene_path, "depth")

      if not all(os.path.exists(d) for d in [color_dir, label_dir, depth_dir]):
        raise FileNotFoundError(f"Missing subdirectories in scene {scene}")

      # Extract frame numbers and match files correctly
      rgb_files = {f.split('.')[0]: f for f in os.listdir(color_dir)}
      label_files = {f.split('.')[0]: f for f in os.listdir(label_dir)}
      depth_files = {f.split('.')[0]: f for f in os.listdir(depth_dir)}

      # Find common frames in all three folders
      common_frames = sorted(set(rgb_files.keys()) & set(label_files.keys()) & set(depth_files.keys()))

      for frame in common_frames:
        pseudo_filename = f"{scene}_{frame}"  # sceneXXXX_XX_00000
        self.file_dict[pseudo_filename] = {
          "rgb": os.path.join(color_dir, rgb_files[frame]),
          "depth": os.path.join(depth_dir, depth_files[frame]),
          "label": os.path.join(label_dir, label_files[frame]),
          "intrinsics_depth": os.path.join(scene_path, "intrinsics_depth.txt"),
          "intrinsics_color": os.path.join(scene_path, "intrinsics_color.txt"),
        }

    # Store sorted keys for deterministic order
    self.file_keys = sorted(self.file_dict.keys())
    
    self.num_samples = len(self.file_keys)  # Total number of samples
    print(f"✅ Found {self.num_samples} samples in {self.split} split.")
    self.skip_probability = 0.001
    # print("Checking max label index in dataset...")
    # for pseudo_filename in tqdm(self.file_keys, desc="Scanning Labels"):
    #     label_path = self.file_dict[pseudo_filename]["label"]  # Get label file path
    #     label = np.array(Image.open(label_path), dtype=np.uint8)  # Load label image
        
    #     max_label = max(max_label, label.max())  # Update max label if higher

    # print(f"✅ Highest class index found: {max_label}")
  def convert_label(self, label):
    mapped_img = np.vectorize(lambda x: self.mapping_dict.get(x, 0))(label)
    return mapped_img.astype(np.uint8)

  def __len__(self):
    return len(self.file_keys)

  def __getitem__(self, idx):
    # Randomly skip based on probability
  
    pseudo_filename = self.file_keys[idx]
    file_data = self.file_dict[pseudo_filename]

    filename = pseudo_filename #os.path.basename(img_path)

    # Load image, label, and depth
    image_raw = Image.open(file_data["rgb"]).convert("RGB")
    image = np.array(image_raw, dtype=np.uint8)
    depth = np.array(Image.open(file_data["depth"]), dtype=np.uint16)
    label_nyu40 = np.array(Image.open(file_data["label"]).convert("L"), dtype=np.uint8)
    label = self.convert_label(label_nyu40)
    # Load intrinsics
    intrinsics_depth = self.read_intrinsics_4x4(file_data["intrinsics_depth"])
    intrinsics_color = self.read_intrinsics_4x4(file_data["intrinsics_color"])

    depth = depth.astype(np.float32)
    # Align depth to RGB
    # Replace missing depth (0 values) with mean depth
    depth[depth == 0] = np.nan
    mean_depth = np.nanmean(depth)
    depth[np.isnan(depth)] = mean_depth
    rgb_size = image_raw.size  # (W, H)
    #print("rgb_size", rgb_size)
    #print("rgb_array", image.shape)
    depth_aligned = self.align_depth_to_rgb(depth, intrinsics_depth, intrinsics_color, rgb_size)
    #print("depth_aligned", depth_aligned.shape)
    

    # Normalize depth (ScanNet depth is up to ~5m)
    dmin = self.dmin
    dmax = self.dmax
    depth_aligned = np.log(depth_aligned.clip(dmin, dmax) / dmin) / np.log(dmax / dmin)

    # Ensure image, depth, and label have the same size
    if image.shape[:2] != depth_aligned.shape[:2] or image.shape[:2] != label.shape[:2]:
        raise ValueError(f"Size mismatch: image size {image.shape[:2]}, depth size {depth_aligned.shape[:2]}, label size {label.shape[:2]}")

    # Apply augmentations
    if self.split == 'train':
      image, depth_aligned, label = self.augment_training_data(image, depth_aligned, label)
    else:
      image, depth_aligned, label = self.augment_test_data(image, depth_aligned, label)

    return {'image': image, 'depth': depth_aligned, 'seg': label, 'filename': filename}
  
  def align_depth_to_rgb(self, depth, K_depth, K_rgb, rgb_size=(1296, 968), depth_scale=1000.0):
    """Reprojects depth pixels into the RGB frame and resizes to match RGB size."""
    h, w = depth.shape  # Original depth size (640×480)
    fx_d, fy_d, cx_d, cy_d = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]

    # Generate 2D grid of depth pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()

    # Convert depth to meters
    Z = depth.flatten() / depth_scale
    X = (x - cx_d) * Z / fx_d
    Y = (y - cy_d) * Z / fy_d

    # Construct 3D points in depth camera frame
    points_3D = np.vstack((X, Y, Z, np.ones_like(Z)))

    # Transform to RGB camera frame using intrinsics
    points_2D = K_rgb @ points_3D[:3, :]
    
    # Convert to pixel coordinates in RGB frame
    x_rgb = (points_2D[0, :] / points_2D[2, :]).astype(np.int32)
    y_rgb = (points_2D[1, :] / points_2D[2, :]).astype(np.int32)

    # Create aligned depth image
    depth_aligned = np.zeros((h, w), dtype=np.float32)
    mask = (x_rgb >= 0) & (x_rgb < w) & (y_rgb >= 0) & (y_rgb < h) & (Z > 0)
    depth_aligned[y_rgb[mask], x_rgb[mask]] = Z[mask]

    # Resize depth to match RGB resolution
    depth_resized = cv2.resize(depth_aligned, rgb_size, interpolation=cv2.INTER_NEAREST)

    return depth_resized


  def read_intrinsics_4x4(self, intrinsics_path):
    """Reads a 4×4 intrinsic matrix and extracts the 3×3 camera matrix."""
    with open(intrinsics_path, "r") as f:
        K_4x4 = np.array([list(map(float, line.split())) for line in f.readlines()])
    
    # Extract 3×3 intrinsic matrix (ignore last column)
    K = K_4x4[:3, :3]
    return K



