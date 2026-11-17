import os
import numpy as np
import torch
from PIL import Image
from dataset.base_dataset import BaseDataset

class nyudepthv2labels40(BaseDataset):
  """
  NYUv2 dataset loader for multi-task learning (depth + segmentation).
  
  Directory structure:
      nyuv2/
        images/
          train/
            img_0001.jpg
            ...
          test/
            img_1001.jpg
        labels13/
          train/
            img_0001.png
            ...
          test/
            img_1001.png
        depth/
          train/
            img_0001.png
            ...
          test/
            img_1001.png
  """
  def __init__(self, data_path, split='train', crop_size=(448, 608), scale_size=None):
    super().__init__(crop_size)  # Initialize BaseDataset
    self.data_path = "/home/user/menon/workspace/data/nyuv2_dataset"#data_path
    self.split = "test" if split == 'val' else split
    self.scale_size = scale_size

    # Define dataset paths
    images_dir = os.path.join(self.data_path, 'images', self.split)
    print(images_dir)
    labels_dir = os.path.join(self.data_path, 'labels40', self.split)
    print(labels_dir)
    depth_dir  = os.path.join(self.data_path, 'depth', self.split)
    print(depth_dir)

    # Ensure dataset directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir) or not os.path.exists(depth_dir):
      raise FileNotFoundError(f"Missing dataset directories in {data_path}")

    # Retrieve file lists
    self.image_files = sorted(os.listdir(images_dir))
    self.label_files = sorted(os.listdir(labels_dir))
    self.depth_files = sorted(os.listdir(depth_dir))

    # Ensure equal number of files
    if len(self.image_files) != len(self.label_files) or len(self.image_files) != len(self.depth_files):
      raise ValueError("Mismatch in number of images, labels, and depth files.")

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    # Construct full file paths
    img_path = os.path.join(self.data_path, 'images', self.split, self.image_files[idx])
    lbl_path = os.path.join(self.data_path, 'labels40', self.split, self.label_files[idx])
    dpt_path = os.path.join(self.data_path, 'depth', self.split, self.depth_files[idx])

    filename = os.path.basename(img_path)

    # Load images
    image = Image.open(img_path).convert("RGB")
    label = Image.open(lbl_path).convert("L")  # Grayscale segmentation labels
    depth = Image.open(dpt_path).convert("I")  # 16-bit depth

    # Remove 16px white border (crop: left, upper, right, lower)
    width, height = image.size
    crop_box = (8, 8, width - 8, height - 8)
    image = image.crop(crop_box)
    label = label.crop(crop_box)
    depth = depth.crop(crop_box)

    # Convert to NumPy arrays
    image = np.array(image, dtype=np.uint8)
    label = np.array(label, dtype=np.uint8)
    depth = np.array(depth, dtype=np.float32)/1000.0

    # Normalize depth (NYUv2 indoor dataset: dmax = 10m)
    dmin = 1e-3
    dmax = 10.0
    depth = np.log(depth.clip(dmin, dmax) / dmin) / np.log(dmax / dmin)
    if np.isnan(depth).any() or np.isinf(depth).any():
      print("NaN/Inf detected in depth_gt before transformation:", depth.min(), depth.max())

    # Apply augmentations
    if self.split == 'train':
      image, depth, label = self.augment_training_data(image, depth, label)
    else:
      image, depth, label = self.augment_test_data(image, depth, label)

    return {'image': image, 'depth': depth, 'seg': label, 'filename': filename}
