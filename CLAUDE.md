# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SLAM3R (CVPR 2025 Highlight) is a real-time dense scene reconstruction system that regresses 3D points from video frames using feed-forward neural networks. It performs dense 3D reconstruction via points regression without explicitly estimating camera parameters.

## Build Commands

### Environment Setup
```bash
# Create conda environment
conda create -n slam3r python=3.11 cmake=3.14.0
conda activate slam3r

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
# Optional: Install additional packages for visualization and preprocessing
pip install -r requirements_optional.txt
```

### Optional: Performance Acceleration
```bash
# Install XFormers (adjust for your PyTorch version)
pip install xformers==0.0.28.post2

# Compile CUDA kernels for RoPE
cd slam3r/pos_embed/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Model Downloads
```bash
# Download model checkpoints (happens automatically during first run)
# Or fetch manually for the Docker environment
mkdir -p checkpoints
# For DUSt3R pretrained weights (if needed for training)
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/
```

## Demo Commands

### Replica Dataset
```bash
# Run demo on Replica dataset
bash scripts/demo_replica.sh
```

### Wild Outdoor Data
```bash
# Run demo on wild data
bash scripts/demo_wild.sh

# Visualize incremental reconstruction (after running with --save_preds)
bash scripts/demo_vis_wild.sh
```

### Gradio Interface
```bash
# Launch the Gradio web interface
python app.py
```

## Evaluation Commands
```bash
# Process ground truth
python evaluation/process_gt.py

# Evaluate on Replica dataset
bash scripts/eval_replica.sh
```

## Training Commands
```bash
# Train Image-to-Points model
bash scripts/train_i2p.sh

# Train Local-to-World model
bash scripts/train_l2w.sh
```

## Docker Commands
```bash
# Build SLAM3R service
docker-compose build slam3r

# Run SLAM3R service
docker-compose up slam3r
```

## System Architecture

SLAM3R consists of two main components:

1. **Image-to-Points (I2P) Model**: Regresses 3D points from RGB frames in camera coordinates.
2. **Local-to-World (L2W) Model**: Registers local point clouds into a global coordinate system.

The reconstruction pipeline:
- Initial scene setup with first few frames
- Keyframe selection with adaptive stride
- Per-frame local point cloud generation (I2P)
- Registration into global coordinates (L2W)
- Multi-keyframe co-registration for maintaining global consistency

The system flows as follows:
1. Load models and parameters from configuration
2. Initialize SLAM system with camera intrinsics
3. Process images through I2P model to get local point clouds
4. Register points via L2W model to world coordinates
5. Maintain keyframe selection based on adaptive stride
6. Output poses, point clouds, and reconstruction data

## Important Files and Components

- `slam3r_processor.py`: Main service entry point for RabbitMQ integration
- `recon.py`: Core reconstruction pipeline implementation
- `slam3r/models.py`: Model definitions for I2P and L2W
- `slam3r/inference.py`: Inference functions for models
- `configs/camera_intrinsics.yaml`: Camera parameters
- `configs/wild.yaml`: Configuration for reconstruction parameters

## Debug and Troubleshooting

- Ensure camera intrinsics YAML has properly formatted numerical values
- When encountering issues with image preprocessing, check the logging output for intrinsic values
- Different versions of CUDA, PyTorch, and xformers can lead to slight variations in results