# EvidMTL

EvidMTL extends SwinMTL with evidential depth estimation and semantic
segmentation. This public release corresponds to the pre-depth-completion
`wonka` source snapshot used by the combined Evidential Mapping project.

## What is new in EvidMTL

Relative to the upstream
[SwinMTL](https://github.com/PardisTaghavi/SwinMTL) architecture, this work
adds:

- evidential heads for joint semantic segmentation and monocular depth;
- semantic and depth uncertainty estimates alongside the task predictions;
- an evidential SiLog depth objective with uncertainty regularization and a
  Normal-Inverse-Gamma prior; and
- the model outputs used by EvidKimera for uncertainty-aware semantic TSDF
  fusion.

SwinMTL remains the underlying shared encoder-decoder architecture. The
original SwinMTL authors and paper are credited below; the modifications above
are described in the EvidMTL paper.

Model checkpoints are hosted separately on
[Hugging Face](https://huggingface.co/rohitmenon86/evidmtl_dc):

- `silog_klmuprior_alpha1.01_gamma0.001_betacalc_final_model.ckpt` uses a
  noisy ground-truth mean prior.
- `silog_klmuprior_alpha2_beta0.1_gamma1.0_no_noise_final_model.ckpt` uses the
  ground-truth mean directly as the prior.

The source code remains licensed under GPL-3.0; see [LICENSE](LICENSE).

## Citation and attribution

If you use this repository, please cite both EvidMTL and the upstream SwinMTL
paper:

```bibtex
@inproceedings{menon2025evidmtl,
  author={Menon, Rohit and Dengler, Nils and Pan, Sicong and Chenchani, Gokul Krishna and Bennewitz, Maren},
  title={EvidMTL: Evidential Multi-Task Learning for Uncertainty-Aware Semantic Surface Mapping from Monocular RGB Images},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  doi={10.1109/IROS60139.2025.11247528}
}

@inproceedings{taghavi2024swinmtl,
  author={Taghavi, Pardis and Langari, Reza and Pandey, Gaurav},
  title={SwinMTL: A Shared Architecture for Simultaneous Depth Estimation and Semantic Segmentation from Monocular Camera Images},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2024},
  url={https://arxiv.org/abs/2403.10662}
}
```

## SwinMTL

### [IROS24] SwinMTL: Multi-Task Learning with Swin Transformer

A multi-task learning framework designed for simultaneous depth estimation and semantic segmentation using the Swin Transformer architecture.

[![Project Page](https://img.shields.io/badge/Project_Page-SwinMTL-green?style=flat)](https://pardistaghavi.github.io/SwinMTL.html) [![Paper](https://img.shields.io/badge/Paper-SwinMTL-blue?style=flat)](https://arxiv.org/abs/2403.10662) [![Papers with Code](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swinmtl-a-shared-architecture-for/multi-task-learning-on-nyuv2)](https://paperswithcode.com/sota/multi-task-learning-on-nyuv2?p=swinmtl-a-shared-architecture-for)

### News
- **[30th June] Paper Accepted at the IROS 2024 Conference 🔥🔥🔥** 


<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/results/qualititativeResults2.png" alt="Qualitative Results" width="800"/>
</div>

### Installation

To get started, follow these steps:

0. Only for ROS installation (otherwise skip this part)
   ```bash
   cd catkin_ws/src
   catkin_create_pkg SwinMTL_ROS std_msgs rospy
   cd ..
   catkin_make
   source devel/setup.bash
   cd src/SwinMTL_ROS/src
   git clone https://github.com/PardisTaghavi/SwinMTL.git
   chmod +x inference_ros.py
   mv ./launch/ ./..  
   ```
1. Clone the repository:
    ```bash
    git clone https://github.com/PardisTaghavi/SwinMTL.git
    cd SwinMTL
    ```

2. Create a conda environment and activate it:
    ```bash
    conda env create --file environment.yml
    conda activate prc
    ```

### Testing

To run the testing for the project, follow the below steps:

1. Download the required EvidMTL checkpoint from the
   [Hugging Face model repository](https://huggingface.co/rohitmenon86/evidmtl_dc).

2. Move Pretrained Models:
    - Create a `weights` directory and place the EvidMTL checkpoint there.
    - Third-party Swin backbone checkpoints expected under `model_zoo` are not
      redistributed by this repository.
    - Refer to `testLive.ipynb` for testing.
  
### ROS Launch
```bash
roslaunch SwinMTL_ROS swinmtl_launch.launch
```
<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/ros3nodes.gif" alt="Zero-shot Results" width="300">
</div>

### 3D Mapping

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/voxelmapDemo-ezgif.com-video-to-gif-converter.gif" alt="3D Mapping Results" width="400">
</div>





### Zero-shot Results on the Kitti Dataset

<div align="center">
  <img src="https://github.com/PardisTaghavi/SwinMTL/blob/main/KittiZeroShotDemo.gif" alt="Zero-shot Results" width="300">
</div>

### Contributions

- Introduction of a multi-task learning approach for joint depth estimation and semantic segmentation.
- Achievement of state-of-the-art performance on Cityscapes and NYUv2 datasets.
- Utilization of an efficient shared encoder-decoder architecture coupled with novel techniques to enhance accuracy.

We welcome feedback and contributions to the SwinMTL project. Feel free to contact taghavi.pardis@gmail.com.

---

### Acknowledgments

Special thanks to the authors of the following projects for laying the foundation of this work. Our code relies on:

- [GLPDepth](https://github.com/vinvino02/GLPDepth)
- [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation?tab=readme-ov-file)
- [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
