# slam3r/slam3r_processor.py
# This script serves as the entry point for the SLAM3R Docker service, orchestrating
# the entire 3D reconstruction pipeline. It continuously listens for incoming RGB image
# frames from a RabbitMQ message queue, processes them using the SLAM3R (Simultaneous
# Localization and Mapping with 3D Reconstruction) engine, and then publishes the
# resulting outputs—such as camera poses, 3D point clouds, and incremental map updates—
# to designated RabbitMQ exchanges for consumption by other services (e.g., visualization tools).
#
# Data Flow:
# 1. Input: RGB image frames and associated metadata (e.g., timestamps) are consumed
#    from the 'VIDEO_FRAMES_EXCHANGE_IN' RabbitMQ exchange. A 'RESTART_EXCHANGE_IN'
#    allows for resetting the SLAM system state.
# 2. Preprocessing: Incoming images are decoded, resized to a target resolution
#    (e.g., 640x480), and converted to RGB. Camera intrinsics, if provided, are
#    loaded and adjusted for the resized images. Images are then normalized and
#    converted to PyTorch tensors.
# 3. SLAM3R Processing:
#    a. Initialization: For a new session, an initial set of frames is buffered.
#       Image tokens (features) are extracted using the Image2Points (I2P) model's encoder.
#       These frames are used to initialize the SLAM scene via `slam3r_initialize_scene`,
#       which generates initial 3D point clouds in a common world frame. An initial
#       keyframe stride can be adapted using `slam3r_adapt_keyframe_stride`.
#       The quality of this initial map is checked against configurable thresholds.
#    b. Incremental Processing: For subsequent frames:
#       i. Image Token Extraction: Features are extracted using `slam3r_get_img_tokens`.
#       ii. Local Reconstruction (I2P): The `i2p_inference_batch` function, using the
#          Image2PointsModel, processes the current frame along with a recent keyframe
#          to generate a local 3D point cloud (pts3d_cam) and confidence scores.
#       iii. Global Registration (L2W): Relevant keyframes from the history are selected
#          using `slam3r_scene_frame_retrieve`. The `l2w_inference` function, using
#          the Local2WorldModel, registers the current frame's local point cloud
#          into the global world coordinate system, yielding `pts3d_world` and a refined pose.
#       iv. Pose Estimation: A rigid transformation (rotation matrix and translation vector)
#           representing the camera's pose in the world frame is estimated using SVD
#           (`estimate_rigid_transform_svd`) from corresponding local and world points.
#       v. Keyframe Selection: Frames meeting certain criteria (e.g., stride-based)
#          are designated as new keyframes and added to the history.
# 4. Output:
#    - Camera Poses: Published to 'SLAM3R_POSE_EXCHANGE_OUT', including position
#      (x,y,z) and orientation (quaternion x,y,z,w), along with the raw 4x4 pose matrix.
#    - Point Clouds: Incremental point clouds (current frame's world points) are
#      published to 'SLAM3R_POINTCLOUD_EXCHANGE_OUT'.
#    - Reconstruction Visualization: Data for visualizing the reconstruction (e.g.,
#      newly added world points, keyframe ID) is published to
#      'SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT'.
#
# Models Used:
# - Image2PointsModel (I2P): Pre-trained model (e.g., "siyan824/slam3r_i2p" from
#   Hugging Face) responsible for lifting 2D image features to local 3D point clouds.
#   It typically consists of an image encoder and a depth/point prediction head.
# - Local2WorldModel (L2W): Pre-trained model (e.g., "siyan824/slam3r_l2w" from
#   Hugging Face) responsible for aligning local point clouds/features from new frames
#   with existing keyframes in the global map, enabling robust tracking and global
#   consistency.
#
# Core Imports Purpose:
# - asyncio, aio_pika: For asynchronous operations, particularly RabbitMQ communication.
# - os: Accessing environment variables for configuration.
# - json: Serializing/deserializing data for RabbitMQ messages.
# - logging: Recording operational information, warnings, and errors.
# - cv2 (OpenCV): Image decoding, resizing, and color space conversions.
# - numpy: Numerical operations, especially for matrix manipulations in pose estimation
#   and point cloud handling.
# - datetime, time: Timestamping and performance measurement.
# - torch (PyTorch): Core deep learning framework for loading models, tensor operations,
#   and running inference on CPU/GPU.
# - yaml: Loading configuration files (e.g., SLAM3R parameters, camera intrinsics).
# - shutil, requests, pathlib: File system and HTTP utilities (less central to core loop).
# - torchvision.transforms: Standard PyTorch image transformations (though custom
#   preprocessing is mainly used here).

import asyncio
import os
import json
import logging
import cv2
import numpy as np
import aio_pika
from datetime import datetime
import torch
import yaml
import shutil
import requests
from pathlib import Path
import time
from torchvision import transforms # Added for potential future use with SLAM3R utils

# Rerun import
import rerun as rr

# Configure logging (Moved Up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import SLAM3R engine components
SLAM3R_ENGINE_AVAILABLE = False
try:
    # Imports from SLAM3R_engine.recon (similar to SLAM3R_engine/app.py)
    # These functions form the core building blocks of the SLAM3R reconstruction pipeline.
    from SLAM3R_engine.recon import (
        get_img_tokens as slam3r_get_img_tokens,         # Extracts feature tokens from images.
        initialize_scene as slam3r_initialize_scene,     # Initializes the SLAM map from initial frames.
        adapt_keyframe_stride as slam3r_adapt_keyframe_stride, # Dynamically adjusts keyframe selection rate.
        i2p_inference_batch,                             # Performs Image-to-Points model inference.
        l2w_inference,                                   # Performs Local-to-World model inference.
        scene_frame_retrieve as slam3r_scene_frame_retrieve # Selects relevant keyframes for L2W.
    )
    # Imports from SLAM3R_engine.slam3r.models
    # These are the PyTorch model classes for the I2P and L2W networks.
    from SLAM3R_engine.slam3r.models import Image2PointsModel, Local2WorldModel
    
    # Optional: If SLAM3R's specific image transformation is needed for better alignment
    # from SLAM3R_engine.slam3r.utils.recon_utils import transform_img

    SLAM3R_ENGINE_AVAILABLE = True
    logger.info("Successfully imported SLAM3R engine components from SLAM3R_engine.")
except ImportError as e:
    logger.error(f"Failed to import SLAM3R engine components: {e}. SLAM3R processing will be disabled. Ensure SLAM3R_engine is in the PYTHONPATH and all dependencies are installed.")
    raise e

# --- Environment Variables ---
# These variables configure the connection to RabbitMQ, exchange names,
# paths to model checkpoints and configuration files, and image processing parameters.

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
# Input RabbitMQ exchange for receiving video frames.
VIDEO_FRAMES_EXCHANGE_IN = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
# Input RabbitMQ exchange for receiving restart commands.
RESTART_EXCHANGE_IN = os.getenv("RESTART_EXCHANGE", "restart_exchange")

# Output RabbitMQ exchange for publishing estimated camera poses.
SLAM3R_POSE_EXCHANGE_OUT = os.getenv("SLAM3R_POSE_EXCHANGE", "slam3r_pose_exchange")
# Output RabbitMQ exchange for publishing generated point clouds.
SLAM3R_POINTCLOUD_EXCHANGE_OUT = os.getenv("SLAM3R_POINTCLOUD_EXCHANGE", "slam3r_pointcloud_exchange")
# Output RabbitMQ exchange for publishing reconstruction visualization updates.
SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT = os.getenv("SLAM3R_RECONSTRUCTION_VIS_EXCHANGE", "slam3r_reconstruction_vis_exchange")

# Directory where SLAM3R model checkpoints are stored (though Hugging Face cache is primary).
CHECKPOINTS_DIR = os.getenv("SLAM3R_CHECKPOINTS_DIR", "/checkpoints_mount")
# Path to the SLAM3R configuration YAML file within the container.
SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER = os.getenv("SLAM3R_CONFIG_FILE", "/app/SLAM3R_engine/configs/wild.yaml")
# Path to the camera intrinsics YAML file.
CAMERA_INTRINSICS_FILE_PATH = os.getenv("CAMERA_INTRINSICS_FILE", "/app/SLAM3R_engine/configs/camera_intrinsics.yaml")

# Image Processing Config
# Target width for image preprocessing, expected by SLAM3R models.
TARGET_IMAGE_WIDTH = int(os.getenv("TARGET_IMAGE_WIDTH", "640"))
# Target height for image preprocessing, expected by SLAM3R models.
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "480"))

# --- Initialization Quality Thresholds ---
# These thresholds determine if the initial SLAM map quality is sufficient.
# If not, the system will attempt to re-initialize.
# Minimum average confidence of points in the initial map.
INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE = float(os.getenv("INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE", "1.0"))
# Minimum total number of valid points in the initial map.
INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS = int(os.getenv("INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS", "100"))

# --- SLAM3R Global State ---
# These variables hold the global state of the SLAM system, including models,
# parameters, and data buffers that persist across frame processing.

slam_system = None # Placeholder for a potential integrated SLAM system object (not currently used).
is_slam_system_initialized = False # Flag indicating if models and parameters are loaded.
camera_intrinsics_dict = None # Stores loaded camera intrinsic parameters.
device = None # PyTorch device (CUDA or CPU) for model inference.

# SLAM3R specific models and state
i2p_model = None # The loaded Image2Points model.
l2w_model = None # The loaded Local2World model.
slam_params = {} # Dictionary to store parameters loaded from the SLAM3R config YAML.

# Per-session SLAM state (these are reset upon receiving a restart command)
# History of processed frames, storing tensors, tokens, points, poses, etc.
processed_frames_history = []
# List of indices into processed_frames_history that correspond to keyframes.
keyframe_indices = []
# Buffer accumulating 3D points in the world coordinate system.
world_point_cloud_buffer = []
# Monotonically increasing index for incoming frames within a session.
current_frame_index = 0
# Flag indicating if the SLAM scene has been successfully initialized for the current session.
is_slam_initialized_for_session = False
# Temporary buffer to hold frames during the initial SLAM setup phase.
slam_initialization_buffer = []
# Reference view ID determined during SLAM session initialization.
reference_view_id_current_session = 0
# Current keyframe stride, which might be dynamically adapted.
active_kf_stride = 1

# Global flag to indicate if Rerun is connected
rerun_connected = False

# Transformation utilities
def matrix_to_quaternion(matrix_3x3):
    """
    Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).

    Args:
        matrix_3x3 (np.ndarray or list of lists): The 3x3 rotation matrix.

    Returns:
        np.ndarray: A 4-element numpy array representing the quaternion [x, y, z, w].
    """
    # Ensure matrix is numpy array
    m = np.asarray(matrix_3x3)
    
    # Check if it's a valid rotation matrix (optional, but good practice)
    # if not (np.allclose(np.dot(m, m.T), np.eye(3)) and np.isclose(np.linalg.det(m), 1.0)):
    #     logger.warning("Input matrix is not a valid rotation matrix. Quaternion may be incorrect.")

    q = np.empty((4, ))
    t = np.trace(m)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        q[3] = 0.5 * t # w
        t = 0.5 / t
        q[0] = (m[2, 1] - m[1, 2]) * t # x
        q[1] = (m[0, 2] - m[2, 0]) * t # y
        q[2] = (m[1, 0] - m[0, 1]) * t # z
    else:
        i = 0
        if m[1, 1] > m[0, 0]:
            i = 1
        if m[2, 2] > m[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t # w
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t
    return q # x, y, z, w

def estimate_rigid_transform_svd(points_src_np, points_tgt_np):
    """
    Estimates the rigid transformation (Rotation R, Translation t) from points_src to points_tgt
    such that: points_tgt = R @ points_src + t. This is often used for aligning point clouds
    or determining camera pose from 3D-3D correspondences.

    Args:
        points_src_np (np.ndarray): Source points, shape (N, 3).
        points_tgt_np (np.ndarray): Target points, shape (N, 3), corresponding to points_src_np.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - R (np.ndarray): The estimated 3x3 rotation matrix.
            - t (np.ndarray): The estimated 3x1 translation vector.
        Returns (np.eye(3), np.zeros((3,1))) if estimation is not possible (e.g., insufficient points).
    """
    if points_src_np.shape[0] < 3 or points_src_np.shape != points_tgt_np.shape:
        logger.warning(f"Not enough points or mismatched shapes for SVD-based pose estimation. Src: {points_src_np.shape}, Tgt: {points_tgt_np.shape}. Returning identity.")
        return np.eye(3), np.zeros((3, 1))

    centroid_src = np.mean(points_src_np, axis=0, keepdims=True)    # (1,3)
    centroid_tgt = np.mean(points_tgt_np, axis=0, keepdims=True)    # (1,3)

    P_src_centered = points_src_np - centroid_src  # (N,3)
    P_tgt_centered = points_tgt_np - centroid_tgt  # (N,3)

    H = P_src_centered.T @ P_tgt_centered  # (3,N) @ (N,3) = (3,3)

    try:
        U, _, Vt = np.linalg.svd(H) # S (singular values) is not used directly for R
    except np.linalg.LinAlgError:
        logger.error("SVD computation failed during pose estimation. Returning identity pose.")
        return np.eye(3), np.zeros((3, 1))

    R = Vt.T @ U.T  # (3,3)

    # Ensure a proper rotation matrix (determinant must be +1)
    if np.linalg.det(R) < 0:
        # logger.debug("Reflection detected in SVD. Correcting R.")
        Vt_corrected = Vt.copy()
        Vt_corrected[2, :] *= -1  # Multiply last row of Vt by -1
        R = Vt_corrected.T @ U.T
        if np.linalg.det(R) < 0: # Should not happen now
             logger.warning("Determinant still < 0 after SVD reflection correction. Pose might be incorrect.")


    t = centroid_tgt.T - R @ centroid_src.T  # (3,1) - (3,3)@(3,1) = (3,1)
    return R, t

def pose_to_dict(position_np, orientation_quat_np):
    """
    Converts numpy arrays for position and quaternion orientation into a dictionary format.

    Args:
        position_np (np.ndarray): A 3-element numpy array for position [x, y, z].
        orientation_quat_np (np.ndarray): A 4-element numpy array for quaternion [x, y, z, w].

    Returns:
        dict: A dictionary containing 'position' and 'orientation' keys,
              each with nested 'x', 'y', 'z' (and 'w' for orientation) float values.
    """
    return {
        "position": {"x": float(position_np[0]), "y": float(position_np[1]), "z": float(position_np[2])},
        "orientation": {"x": float(orientation_quat_np[0]), "y": float(orientation_quat_np[1]), "z": float(orientation_quat_np[2]), "w": float(orientation_quat_np[3])}
    }

def load_camera_intrinsics(file_path):
    """Loads camera intrinsics from a YAML file.

    The YAML file is expected to contain 'fx', 'fy', 'cx', 'cy' keys
    representing the focal lengths and principal point coordinates.

    Args:
        file_path (str): The path to the YAML file containing camera intrinsics.

    Returns:
        dict or None: A dictionary with 'fx', 'fy', 'cx', 'cy' as float values
                      if successful, otherwise None.
    """
    try:
        with open(file_path, 'r') as f:
            intrinsics = yaml.safe_load(f)
        if not all(k in intrinsics for k in ['fx', 'fy', 'cx', 'cy']):
            logger.error(f"Camera intrinsics file {file_path} is missing one or more keys (fx, fy, cx, cy).")
            return None
        
        # Convert intrinsic values to float
        try:
            intrinsics['fx'] = float(intrinsics['fx'])
            intrinsics['fy'] = float(intrinsics['fy'])
            intrinsics['cx'] = float(intrinsics['cx'])
            intrinsics['cy'] = float(intrinsics['cy'])
        except ValueError as e:
            logger.error(f"Error converting camera intrinsic values to float in {file_path}: {intrinsics}. Error: {e}. Check YAML format (values should be numbers or numerical strings, without unexpected characters like trailing spaces or symbols).")
            return None
        except TypeError as e: # Handles cases where fx, fy, etc. might not be string/numeric (e.g. list)
            logger.error(f"Type error during conversion of camera intrinsic values in {file_path}: {intrinsics}. Error: {e}. Ensure values are simple numbers or numerical strings.")
            return None

        logger.info(f"Loaded and converted camera intrinsics from {file_path}: {intrinsics}")
        return intrinsics
    except FileNotFoundError:
        logger.warning(f"Camera intrinsics file not found at {file_path}. Proceeding without specific intrinsics.")
        return None
    except Exception as e:
        logger.error(f"Error loading camera intrinsics from {file_path}: {e}", exc_info=True)
        return None

async def initialize_slam_system():
    """
    Initializes the SLAM3R system by loading models, configurations, and resetting session state.
    This function is called at startup and upon receiving a restart command. It sets up
    the Image2Points (I2P) and Local2World (L2W) models, loads SLAM parameters from a
    configuration file, and prepares global state variables for a new SLAM session.

    Global variables modified:
        - slam_system, is_slam_system_initialized, camera_intrinsics_dict, device
        - i2p_model, l2w_model, slam_params
        - processed_frames_history, keyframe_indices, world_point_cloud_buffer
        - current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer,
        - reference_view_id_current_session, active_kf_stride
        - rerun_connected

    Returns:
        bool: True if initialization was successful or system was already initialized,
              False otherwise.
    """
    global slam_system, is_slam_system_initialized, camera_intrinsics_dict, device
    global i2p_model, l2w_model, slam_params
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer
    global reference_view_id_current_session, active_kf_stride
    global rerun_connected
    
    # ------------------------------------------------------------------
    # Robust Rerun viewer connection (only attempted once per container)
    # ------------------------------------------------------------------
    if os.getenv("RERUN_ENABLED", "true").lower() == "true" and not rerun_connected:
        rr.init("SLAM3R_Processor", spawn=False)

        # Build a list of candidate URLs in order of preference.
        candidate_urls: list[str] = []

        # 1) Explicit env var (best‑effort)
        env_connect_url = os.getenv("RERUN_CONNECT_URL")
        if env_connect_url:
            candidate_urls.append(env_connect_url)

        # 2) If the user supplied a plain viewer address, derive the proxy URL.
        env_viewer_addr = os.getenv("RERUN_VIEWER_ADDRESS")
        if env_viewer_addr:
            # If already in "rerun+http" form just keep it, otherwise convert.
            if env_viewer_addr.startswith("rerun+"):
                candidate_urls.append(env_viewer_addr)
            else:
                candidate_urls.append(f"rerun+http://{env_viewer_addr}/proxy")

        # 3) Common docker-host fallbacks.
        candidate_urls.extend([
            "rerun+http://host.docker.internal:9876/proxy",
            "rerun+http://127.0.0.1:9876/proxy",
            "rerun+http://localhost:9876/proxy",
        ])

        logger.info(f"Attempting to connect to Rerun viewer. Candidate URLs: {candidate_urls}")

        connected = False
        for url in candidate_urls:
            try:
                logger.info(f"Trying Rerun gRPC connect: {url}")
                print(f"Trying Rerun gRPC connect: {url}")
                rr.connect_grpc(url, flush_timeout_sec=10.0)
                logger.info(f"✅ Connected to Rerun viewer at {url}")
                # Establish a canonical coordinate system once connected.
                rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
                demo_point_grid()  # quick sanity‑check visual
                connected = True
                break
            except Exception as conn_err:
                logger.warning(f"Connection failed for {url}: {conn_err}")

        rerun_connected = connected
        if not rerun_connected:
            logger.error(
                "❌ Unable to connect to any Rerun viewer instance. "
                "Rerun visualization will be disabled for this session."
            )
    
    if not SLAM3R_ENGINE_AVAILABLE:
        logger.error("SLAM3R Engine components are not available. Cannot initialize SLAM system.")
        is_slam_system_initialized = False
        return False

    if is_slam_system_initialized:
        logger.info("SLAM3R system already initialized.")
        return True

    logger.info(f"Attempting to initialize SLAM3R system.")
    logger.info(f"Using SLAM3R config: {SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER}")
    logger.info(f"Attempting to load checkpoints from: {CHECKPOINTS_DIR}")
    logger.info(f"Attempting to load camera_intrinsics from: {CAMERA_INTRINSICS_FILE_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"SLAM3R will use device: {device}")

    camera_intrinsics_dict = load_camera_intrinsics(CAMERA_INTRINSICS_FILE_PATH)

    try:
        # Define the HuggingFace model IDs
        i2p_model_id = "siyan824/slam3r_i2p"
        l2w_model_id = "siyan824/slam3r_l2w"
        # Models are expected to be pre-downloaded in the Docker image build (via Hugging Face cache).
        # The CHECKPOINTS_DIR is still logged but models are not loaded from there directly by this function.

        logger.info(f"Loading Image2Points model '{i2p_model_id}' from Hugging Face cache (pre-downloaded).")
        i2p_model = Image2PointsModel.from_pretrained(i2p_model_id).to(device).eval()
        logger.info("Image2Points model loaded.")

        logger.info(f"Loading Local2World model '{l2w_model_id}' from Hugging Face cache (pre-downloaded).")
        l2w_model = Local2WorldModel.from_pretrained(l2w_model_id).to(device).eval()
        logger.info("Local2World model loaded.")
        
        # Load SLAM3R specific configuration (e.g., from a YAML file provided by SLAM3R_engine)
        # This config will provide parameters used in recon.py, app.py
        if os.path.exists(SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER):
            with open(SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER, 'r') as f:
                slam_config_yaml = yaml.safe_load(f) 
            logger.info(f"Loaded SLAM3R parameters from {SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER}")
            # Extract parameters similar to argparse in recon.py or Gradio inputs in app.py
            slam_params['keyframe_stride'] = slam_config_yaml.get('recon_pipeline', {}).get('keyframe_stride', -1) # -1 for auto
            slam_params['initial_winsize'] = slam_config_yaml.get('recon_pipeline', {}).get('initial_winsize', 5)
            slam_params['win_r'] = slam_config_yaml.get('recon_pipeline', {}).get('win_r', 3) # radius for I2P local window
            slam_params['conf_thres_i2p'] = slam_config_yaml.get('recon_pipeline', {}).get('conf_thres_i2p', 1.5)
            slam_params['num_scene_frame'] = slam_config_yaml.get('recon_pipeline', {}).get('num_scene_frame', 10) # for L2W
            slam_params['conf_thres_l2w'] = slam_config_yaml.get('recon_pipeline', {}).get('conf_thres_l2w', 12.0) # for final filtering of points
            slam_params['update_buffer_intv_factor'] = slam_config_yaml.get('recon_pipeline', {}).get('update_buffer_intv_factor', 1) # Multiplied by kf_stride
            slam_params['buffer_size'] = slam_config_yaml.get('recon_pipeline', {}).get('buffer_size', 100)
            slam_params['buffer_strategy'] = slam_config_yaml.get('recon_pipeline', {}).get('buffer_strategy', 'reservoir')
            slam_params['norm_input_l2w'] = slam_config_yaml.get('recon_pipeline', {}).get('norm_input_l2w', False)
            # Keyframe adaptation params (if keyframe_stride is -1)
            slam_params['keyframe_adapt_min'] = slam_config_yaml.get('keyframe_adaptation', {}).get('adapt_min', 1)
            slam_params['keyframe_adapt_max'] = slam_config_yaml.get('keyframe_adaptation', {}).get('adapt_max', 5) # Smaller default for streaming
            slam_params['keyframe_adapt_stride_step'] = slam_config_yaml.get('keyframe_adaptation', {}).get('adapt_stride_step', 1)

            active_kf_stride = slam_params['keyframe_stride'] if slam_params['keyframe_stride'] > 0 else 1 # Default if auto not run yet
            logger.info(f"SLAM parameters: {slam_params}")
        else:
            logger.warning(f"SLAM3R config file not found at {SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER}. Using default SLAM parameters.")
            # Set default slam_params if file not found (mirroring some defaults from recon.py/app.py)
            slam_params = {
                'keyframe_stride': 1, 'initial_winsize': 5, 'win_r': 3, 'conf_thres_i2p': 1.5,
                'num_scene_frame': 10, 'conf_thres_l2w': 12.0, 'update_buffer_intv_factor': 1,
                'buffer_size': 100, 'buffer_strategy': 'reservoir', 'norm_input_l2w': False,
                'keyframe_adapt_min': 1, 'keyframe_adapt_max': 5, 'keyframe_adapt_stride_step': 1
            }
            active_kf_stride = slam_params['keyframe_stride']

        # Reset per-session SLAM state
        processed_frames_history = []
        keyframe_indices = []
        world_point_cloud_buffer = [] # Or load from a map if supported
        current_frame_index = 0
        is_slam_initialized_for_session = False
        slam_initialization_buffer = []
        reference_view_id_current_session = 0
        
        logger.info("SLAM3R models and parameters loaded. System ready for session initialization.")
        is_slam_system_initialized = True

    except ImportError as e:
        logger.error(f"Failed to import SLAM3R components. Ensure SLAM3R_engine is in PYTHONPATH and submodule is initialized: {e}", exc_info=True)
        is_slam_system_initialized = False
    except Exception as e:
        logger.error(f"Failed to initialize SLAM3R system: {e}", exc_info=True)
        is_slam_system_initialized = False
    return is_slam_system_initialized

# ---------------------------------------------------------------------------
# Demo: coloured lattice so we can see data arriving in the viewer
# ---------------------------------------------------------------------------
def demo_point_grid(size: int = 10):
    xs = np.linspace(-10, 10, size, dtype=np.float32)
    positions = (
        np.stack(np.meshgrid(xs, xs, xs), axis=-1)     # (size,size,size,3)
        .reshape(-1, 3)                                # (N,3)
    )

    cs = np.linspace(0, 255, size, dtype=np.uint8)
    colors = (
        np.stack(np.meshgrid(cs, cs, cs), axis=-1)
        .reshape(-1, 3)                                # (N,3)
    )

    # radii must be one‑per‑point (or None)
    radii = np.full((positions.shape[0],), 0.5, dtype=np.float32)

    rr.log("world/demo_point_grid", rr.Points3D(positions=positions, colors=colors, radii=radii))

def preprocess_image(image_np, target_width, target_height, intrinsics=None):
    """
    Preprocesses an input image (numpy array) for SLAM3R processing.
    This involves resizing, color conversion (BGR to RGB), normalization,
    and conversion to a PyTorch tensor. It also adjusts camera intrinsics
    if provided, to match the new image dimensions.

    Args:
        image_np (np.ndarray): The input image in BGR format (H, W, C).
        target_width (int): The desired width for the processed image.
        target_height (int): The desired height for the processed image.
        intrinsics (dict, optional): Original camera intrinsics {'fx', 'fy', 'cx', 'cy'}.
                                     If provided, they will be scaled to the target dimensions.

    Returns:
        Tuple[torch.Tensor, dict or None]:
            - torch.Tensor: The preprocessed image tensor (C, H, W), scaled to [0,1], on the configured device.
            - dict or None: Adjusted camera intrinsics if input intrinsics were provided, else None.
    """
    # Resize
    height, width, _ = image_np.shape
    resized_img = cv2.resize(image_np, (target_width, target_height))

    # BGR to RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Normalize and convert to tensor (typical PyTorch preprocessing)
    # These values (mean, std) should match what SLAM3R used for training.
    # SLAM3R might have its own preprocessing utilities.
    img_tensor = torch.tensor(rgb_img, dtype=torch.float32).permute(2, 0, 1) / 255.0 # HWC to CHW, scale to [0,1]
    # SLAM3R's transform_img (from recon_utils) might be more aligned if it includes specific normalization
    # For now, this basic normalization is kept. If issues, align with SLAM3R_engine.slam3r.utils.recon_utils.transform_img
    
    # Example using SLAM3R's transform_img if it standardizes inputs:
    # view_dict = {'img': torch.tensor(rgb_img[None])} # Needs batch dim
    # transformed_view = transform_img(view_dict)
    # img_tensor = transformed_view['img'][0] # Remove batch dim if added

    # Adjust intrinsics if image was resized
    adjusted_intrinsics = None
    if intrinsics:
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        scale_x = target_width / width
        scale_y = target_height / height
        logger.info(f"Debugging intrinsics: fx={fx} (type={type(fx)}), fy={fy} (type={type(fy)}), cx={cx} (type={type(cx)}), cy={cy} (type={type(cy)})")
        logger.info(f"Debugging scales: scale_x={scale_x} (type={type(scale_x)}), scale_y={scale_y} (type={type(scale_y)})")
        logger.info(f"Debugging image dimensions: target_width={target_width}, target_height={target_height}, width={width}, height={height}")
        adjusted_intrinsics = {
            "fx": fx * scale_x, "fy": fy * scale_y,
            "cx": cx * scale_x, "cy": cy * scale_y
        }
        # Convert intrinsics to tensor if SLAM3R expects it
        # adjusted_intrinsics_tensor = torch.tensor([adjusted_intrinsics['fx'], adjusted_intrinsics['fy'], adjusted_intrinsics['cx'], adjusted_intrinsics['cy']], dtype=torch.float32)

    return img_tensor.to(device), adjusted_intrinsics # Return tensor and dict for now

async def process_image_with_slam3r(image_np, timestamp_ns, headers):
    """
    Core function to process a single image frame using the SLAM3R pipeline.
    This function handles:
    1. Image preprocessing.
    2. SLAM session initialization (if not already done) using a buffer of initial frames.
       This includes feature extraction, scene initialization with I2P, and quality checks.
    3. Incremental SLAM processing for subsequent frames:
       - Feature (token) extraction via `slam3r_get_img_tokens`.
       - Local point cloud generation using Image2Points model (`i2p_inference_batch`).
       - Global registration and pose refinement using Local2World model (`l2w_inference`),
         leveraging selected keyframes (`slam3r_scene_frame_retrieve`).
       - Pose estimation from 3D-3D correspondences (`estimate_rigid_transform_svd`).
       - Keyframe selection and updating the world point cloud buffer.
    4. Formatting and returning pose data, point cloud data, and reconstruction updates.

    Args:
        image_np (np.ndarray): The input image as a NumPy array (BGR format).
        timestamp_ns (int): Timestamp of the image in nanoseconds.
        headers (dict): Headers from the RabbitMQ message, potentially containing metadata.

    Global variables accessed/modified:
        - i2p_model, l2w_model, slam_params, camera_intrinsics_dict, device
        - processed_frames_history, keyframe_indices, world_point_cloud_buffer
        - current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer,
        - reference_view_id_current_session, active_kf_stride

    Returns:
        Tuple[dict, dict, dict] or Tuple[None, None, None]:
            - pose_data (dict): Estimated camera pose.
            - point_cloud_data (dict): Generated 3D point cloud for the current frame.
            - reconstruction_update_data (dict): Incremental update for visualization.
            Returns (None, None, None) if processing fails or system is not initialized.
    """
    global i2p_model, l2w_model, slam_params, camera_intrinsics_dict, device
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer
    global reference_view_id_current_session, active_kf_stride

    start_time = time.time()

    if not is_slam_system_initialized or not i2p_model or not l2w_model:
        logger.warning("SLAM3R system not (fully) initialized with models. Skipping frame processing.")
        return None, None, None

    try:
        # 1. Preprocess the image
        preprocessed_image_tensor, current_frame_intrinsics = preprocess_image(
            image_np, 
            TARGET_IMAGE_WIDTH, 
            TARGET_IMAGE_HEIGHT,
            camera_intrinsics_dict 
        )
        preprocessed_image_tensor = preprocessed_image_tensor.to(device)

        if rerun_connected:
            # Log RGB image (convert BGR to RGB)
            rgb_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            rr.log("world/camera/image", rr.Image(rgb_image_np))

            # Log Camera Intrinsics if available
            if current_frame_intrinsics:
                fx = current_frame_intrinsics['fx']
                fy = current_frame_intrinsics['fy']
                cx = current_frame_intrinsics['cx']
                cy = current_frame_intrinsics['cy']
                intrinsics_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                rr.log(
                    "world/camera",
                    rr.Pinhole(
                        image_from_camera=intrinsics_matrix,
                        resolution=[TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT],
                    ),
                )

        # Prepare view dict as expected by SLAM3R utils (based on recon.py and app.py)
        # The 'img' tensor for SLAM3R is typically CHW, no batch dim for single processing then batched by utils.
        # `true_shape` refers to original shape before padding/cropping by model if any, here it's our target processing shape.
        current_view_minimal = {
            'img': preprocessed_image_tensor.unsqueeze(0), # Add batch dimension for model processing
            'true_shape': torch.tensor([[TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH]], device=device), # H, W
            'img_pos': None, # Will be filled by get_img_tokens if needed by L2W directly
            'label': f"frame_{current_frame_index}"
        }
        # `to_device` is usually for dicts of tensors.
        # to_device(current_view_minimal, device) # already on device

        # Get image tokens (encoder output) - crucial for L2W model and some I2P setups
        # get_img_tokens expects a list of views
        # This is a heavy operation if done per frame for both models.
        # recon.py pre-extracts all tokens. For streaming, we do it as needed.
        # The i2p_model itself has an encoder. l2w_model can be set with `need_encoder=False` if tokens are fed.

        # SLAM3R's get_img_tokens typically processes a list of views and returns lists.
        # For a single view:
        # _, current_img_tokens, current_img_pos = slam3r_get_img_tokens([current_view_minimal], i2p_model) # This is how app.py uses it.
        # current_img_tokens = current_img_tokens[0]
        # current_img_pos = current_img_pos[0]
        # The above requires i2p_model to have _encode_multiview.
        # Simpler: I2P uses its encoder internally. L2W needs tokens if need_encoder=False.
        # For now, assume i2p_model handles its encoding, and L2W will get tokens from keyframes.
        
        frame_data_for_history = {
            'img_tensor': preprocessed_image_tensor, # CHW
            # 'img_tokens': current_img_tokens, # Store if L2W needs them directly from non-keyframes
            # 'img_pos': current_img_pos,
            'true_shape': torch.tensor([TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH], device=device), # H, W
            'timestamp_ns': timestamp_ns,
            'keyframe_id': None, # To be filled if it becomes a keyframe
            'pts3d_cam': None,   # Local points from I2P (tensor)
            'conf_cam': None,    # Confidence for local points (tensor)
            'pts3d_world': None, # Global points after L2W (tensor)
            'conf_world': None,  # Confidence for world points (tensor)
            'raw_pose_matrix': np.eye(4).tolist() # Default identity, to be updated by L2W
        }

        # --- SLAM Logic ---
        temp_pose_matrix_4x4 = np.eye(4) # Default identity
        temp_points_xyz_list = []
        temp_mesh_vertices_list = []
        temp_mesh_faces_list = []
        keyframe_id_for_output = None

        if not is_slam_initialized_for_session:
            logger.info(f"SLAM session not initialized. Buffering frame {current_frame_index}.")
            # Add necessary parts of current_view_minimal for initialization
            init_view_data = {
                'img_tokens': None, # Will be computed during initialization
                'img_pos': None,    # Will be computed during initialization
                'true_shape': current_view_minimal['true_shape'][0], # Remove batch
                'img': current_view_minimal['img'], # Keep batch for get_img_tokens
                'label': current_view_minimal['label']
            }
            slam_initialization_buffer.append(init_view_data)

            if len(slam_initialization_buffer) >= slam_params['initial_winsize']:
                init_start_time = time.time()
                logger.info(f"Collected {len(slam_initialization_buffer)} frames. Attempting SLAM session initialization.")
                
                # Pre-extract img_tokens for all views in buffer (as in recon.py)
                # Ensure views for slam3r_get_img_tokens are structured correctly (list of dicts)
                # Each dict must have 'img', 'true_shape'. 'img' should be batched tensor.
                # 'true_shape' should be [H, W] tensor.
                
                # We need to pass the list of views that slam3r_get_img_tokens expects
                # Each view in this list is a dict: {'img': tensor (B,C,H,W), 'true_shape': tensor (B,2), ...}
                # Our slam_initialization_buffer has 'img' as (1,C,H,W) and 'true_shape' as (2,)
                # Let's re-structure for slam3r_get_img_tokens:
                batched_init_views_for_tokens = []
                for view_data in slam_initialization_buffer:
                    # view_data['img'] is already (1,C,H,W)
                    # view_data['true_shape'] is (2,) -> needs to be (1,2)
                    batched_init_views_for_tokens.append({
                        'img': view_data['img'].to(device),
                        'true_shape': view_data['true_shape'].unsqueeze(0).to(device), # Add batch dim
                        'label': view_data['label']
                        # img_pos will be added by get_img_tokens
                    })

                # Call get_img_tokens (this populates 'img_tokens' and 'img_pos' in each dict)
                # It expects a list of dicts, and modifies them or returns new structures.
                # Based on recon.py: res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)
                # Then it constructs input_views with these. Let's adapt.
                try:
                    # This function might expect 'img' (B,C,H,W) and 'true_shape' (B,2)
                    # and adds 'img_tokens', 'img_pos' to each view dictionary in the list.
                    # Or it returns separate lists. The function signature used in app.py is:
                    # res_shapes, res_feats, res_poses = get_img_tokens(data_views, i2p_model)
                    # Let's assume it modifies the list of dicts in place for simplicity here, or we adapt.
                    # For now, let's assume we need to prepare a list of views in the expected format for slam3r_initialize_scene
                    
                    initial_input_views_for_slam = []
                    # Get tokens for each view separately if slam3r_get_img_tokens is tricky with batches here
                    for i, view_data in enumerate(slam_initialization_buffer):
                        # view_data['img'] is (1,C,H,W)
                        # view_data['true_shape'] is (H,W)
                        # We need to pass a list containing this single view to get_img_tokens
                        single_view_list_for_tokens = [{
                            'img': view_data['img'].to(device), # Already batched [1,C,H,W]
                            'true_shape': view_data['true_shape'].unsqueeze(0).to(device) # Batch [1,2]
                        }]
                        token_start_time = time.time()
                        _, view_token, view_pos = slam3r_get_img_tokens(single_view_list_for_tokens, i2p_model)
                        token_end_time = time.time()
                        logger.info(f"get_img_tokens for view {i} took {token_end_time - token_start_time:.4f} seconds")
                        
                        initial_input_views_for_slam.append({
                            'img_tokens': view_token[0], # Remove list wrapper
                            'img_pos': view_pos[0],       # Remove list wrapper
                            'true_shape': view_data['true_shape'], # Original H,W tensor
                            'label': view_data['label'],
                            # Store original tensor for history
                            'img_tensor': processed_frames_history[current_frame_index - len(slam_initialization_buffer) + i + 1]['img_tensor'] if (current_frame_index - len(slam_initialization_buffer) + i + 1) < len(processed_frames_history) else frame_data_for_history['img_tensor']
                        })


                    # Determine keyframe stride if set to auto
                    if active_kf_stride == -1 and slam_params.get('keyframe_stride', -1) == -1:
                        logger.info("Determining optimal keyframe stride...")
                        # Ensure initial_input_views_for_slam has the structure adapt_keyframe_stride expects
                        # (list of dicts, each with 'img_tokens', 'true_shape', 'img_pos')
                        active_kf_stride = slam3r_adapt_keyframe_stride(
                            initial_input_views_for_slam, i2p_model,
                            win_r=slam_params.get('win_r_adapt', 3), # Use a specific win_r for adaptation if needed
                            adapt_min=slam_params['keyframe_adapt_min'],
                            adapt_max=slam_params['keyframe_adapt_max'],
                            adapt_stride=slam_params['keyframe_adapt_stride_step']
                        )
                        logger.info(f"Adapted keyframe stride to: {active_kf_stride}")
                        if active_kf_stride <= 0: active_kf_stride = 1 # Ensure positive

                    # Initialize scene using these views with tokens
                    # `slam3r_initialize_scene` expects a list of views (dicts with img_tokens, etc.)
                    # It returns initial_pcds (list of tensors), initial_confs (list of tensors), init_ref_id
                    init_scene_start_time = time.time()
                    initial_pcds_tensors, initial_confs_tensors, init_ref_id = slam3r_initialize_scene(
                        initial_input_views_for_slam[::active_kf_stride], # Pass only keyframes based on stride
                        i2p_model,
                        winsize=min(slam_params['initial_winsize'], len(initial_input_views_for_slam[::active_kf_stride])),
                        conf_thres=slam_params['conf_thres_i2p'], # This is for point generation, not the quality check threshold
                        return_ref_id=True
                    )
                    init_scene_end_time = time.time()
                    logger.info(f"slam3r_initialize_scene took {init_scene_end_time - init_scene_start_time:.4f} seconds")
                    reference_view_id_current_session = init_ref_id # This is local index within the initial_input_views_for_slam[::active_kf_stride]

                    # --- Initialization Quality Check ---
                    total_valid_points_init = 0
                    sum_confidence_init = 0.0
                    total_points_considered_init = 0

                    for conf_tensor_kf in initial_confs_tensors:
                        conf_np_kf = conf_tensor_kf.cpu().numpy().flatten()
                        valid_mask_kf = conf_np_kf > slam_params['conf_thres_i2p']
                        total_valid_points_init += np.sum(valid_mask_kf)
                        sum_confidence_init += np.sum(conf_np_kf[valid_mask_kf]) # Sum confidences of valid points
                        total_points_considered_init += len(conf_np_kf) # All points considered for average
                    
                    avg_confidence_init = (sum_confidence_init / total_valid_points_init) if total_valid_points_init > 0 else 0

                    logger.info(f"Initialization attempt results: Avg Confidence (valid points) = {avg_confidence_init:.2f}, Total Valid Points = {total_valid_points_init}")

                    if avg_confidence_init < INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE or \
                       total_valid_points_init < INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS:
                        logger.warning("Initial SLAM scene quality below thresholds. Resetting and re-attempting initialization.")
                        logger.warning(f"Metrics: AvgConf={avg_confidence_init:.2f} (MinReq={INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE}), ValidPts={total_valid_points_init} (MinReq={INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS})")
                        slam_initialization_buffer = [] # Clear buffer to gather new frames
                        is_slam_initialized_for_session = False
                        # Clear any potential KFs or world points from this failed attempt
                        keyframe_indices = [] 
                        world_point_cloud_buffer = []
                        # current_frame_index and processed_frames_history continue, new buffer will be filled
                        # No return here, the outer loop will continue, and this block will be re-entered after more frames.
                    else:
                        logger.info("Initial SLAM scene quality check PASSED.")
                        # Store initial keyframes and their world points
                        # The initial_pcds are already in a common (world) frame relative to init_ref_id
                        init_kf_counter = 0
                        for i in range(len(slam_initialization_buffer)):
                            history_idx = current_frame_index - len(slam_initialization_buffer) + i + 1

                            # Check if history_idx is within range
                            if history_idx >= len(processed_frames_history):
                                logger.warning(f"History index {history_idx} out of range for processed_frames_history (length {len(processed_frames_history)})")
                                # Append empty frame data to processed_frames_history if needed
                                while len(processed_frames_history) <= history_idx:
                                    processed_frames_history.append({
                                        'img_tensor': None,
                                        'img_tokens': None,
                                        'img_pos': None,
                                        'true_shape': None,
                                        'timestamp_ns': None,
                                        'keyframe_id': None,
                                        'pts3d_cam': None,
                                        'conf_cam': None,
                                        'pts3d_world': None,
                                        'conf_world': None,
                                        'raw_pose_matrix': np.eye(4).tolist()
                                    })

                            # Make sure we have a valid index before accessing initial_input_views_for_slam
                            if i < len(initial_input_views_for_slam):
                                processed_frames_history[history_idx]['img_tokens'] = initial_input_views_for_slam[i]['img_tokens']
                                processed_frames_history[history_idx]['img_pos'] = initial_input_views_for_slam[i]['img_pos']
                            else:
                                logger.warning(f"Index {i} out of range for initial_input_views_for_slam (length {len(initial_input_views_for_slam)})")

                            if i % active_kf_stride == 0 and init_kf_counter < len(initial_pcds_tensors):
                                kf_original_buffer_index = i 
                                
                                keyframe_indices.append(history_idx)
                                processed_frames_history[history_idx]['keyframe_id'] = f"kf_{len(keyframe_indices)-1}"
                                processed_frames_history[history_idx]['pts3d_world'] = initial_pcds_tensors[init_kf_counter] 
                                processed_frames_history[history_idx]['conf_world'] = initial_confs_tensors[init_kf_counter]
                                
                                valid_mask_init = (initial_confs_tensors[init_kf_counter] > slam_params['conf_thres_i2p']).squeeze()
                                points_to_add_init = initial_pcds_tensors[init_kf_counter].squeeze()[valid_mask_init].cpu().numpy()
                                world_point_cloud_buffer.extend(points_to_add_init.tolist())
                                
                                if history_idx == keyframe_indices[-1]:
                                    temp_points_xyz_list = points_to_add_init.tolist()
                                    keyframe_id_for_output = processed_frames_history[history_idx]['keyframe_id']
                                
                                init_kf_counter +=1
                        
                        slam_initialization_buffer = [] # Clear buffer as it's now processed
                        is_slam_initialized_for_session = True
                        logger.info(f"SLAM session initialized. Ref ID (local to init KF batch): {init_ref_id}. Total Keyframes: {len(keyframe_indices)}. World points: {len(world_point_cloud_buffer)}")

                except Exception as e_init:
                    logger.error(f"Error during SLAM session initialization: {e_init}", exc_info=True)
                    slam_initialization_buffer = [] 
                    is_slam_initialized_for_session = False
                    keyframe_indices = [] # Ensure clean state on error too
                    world_point_cloud_buffer = []


        elif is_slam_initialized_for_session: # Standard processing after initialization
            # Get tokens for the current single frame (needed for L2W)
            # We need to provide current_view_minimal to get_img_tokens
            single_view_list_for_tokens = [{
                'img': current_view_minimal['img'].to(device), 
                'true_shape': current_view_minimal['true_shape'].to(device)
            }]
            _, view_token, view_pos = slam3r_get_img_tokens(single_view_list_for_tokens, i2p_model)
            frame_data_for_history['img_tokens'] = view_token[0]
            frame_data_for_history['img_pos'] = view_pos[0]

            # 1. I2P: Get local point cloud for the current view
            # i2p_inference_batch expects a list of windows, each window a list of views
            # For a single frame, we might form a "window" around it if win_r > 0, using past frames
            # Or, if win_r=0 (not typical for I2P in recon), process individually.
            # Let's assume a simplified I2P for streaming: process current frame in a minimal "window" (itself)
            # This needs careful adaptation of i2p_inference_batch or a stream-friendly version.
            # For now, let's try with a window of 1 (current frame only) as ref.
            
            # The model expects multiple views - it can't process single views
            # Create a multi-view input with the current frame and at least one previous frame
            # Let's get the latest keyframe as a reference view
            multiview_input = []

            if keyframe_indices:
                # Get the most recent keyframe
                ref_kf_idx = keyframe_indices[-1]
                ref_kf_view = {
                    'img_tokens': processed_frames_history[ref_kf_idx]['img_tokens'],
                    'img_pos': processed_frames_history[ref_kf_idx]['img_pos'],
                    'true_shape': processed_frames_history[ref_kf_idx]['true_shape'],
                    'label': f"keyframe_{ref_kf_idx}"
                }

                current_i2p_input_view = {
                    'img_tokens': frame_data_for_history['img_tokens'],
                    'img_pos': frame_data_for_history['img_pos'],
                    'true_shape': frame_data_for_history['true_shape'],
                    'label': f"frame_{current_frame_index}"
                }

                # Create the multi-view input with the reference view first, then current view
                multiview_input = [[ref_kf_view, current_i2p_input_view]]

                logger.info(f"Using multi-view input with reference keyframe {ref_kf_idx} and current frame {current_frame_index}")

                # i2p_inference_batch takes list_of_lists_of_views.
                # ref_id is local to the inner list - we use 0 to set the keyframe as reference
                i2p_output = i2p_inference_batch(multiview_input, i2p_model, ref_id=0, tocpu=False, unsqueeze=False)
            else:
                logger.error("No keyframes available for I2P multi-view processing. Cannot process current frame.")
                return None, None, None
            
            # i2p_output['preds'] is a list (outer batch) of lists (inner window) of dicts
            # Each dict has 'pts3d', 'conf'
            # Since we're using a reference view (0) and current view (1), we need the result for the current view
            try:
                # The structure is different now - we need to get the current view results, which is index 1
                # [0] is the first batch, [1] is the current frame (second view)
                current_pts3d_cam = i2p_output['preds'][0][1]['pts3d']  # Shape: (1, H, W, 3) or (1, N, 3)
                current_conf_cam = i2p_output['preds'][0][1]['conf']    # Shape: (1, H, W) or (1, N)
                frame_data_for_history['pts3d_cam'] = current_pts3d_cam
                frame_data_for_history['conf_cam'] = current_conf_cam
                logger.info(f"Successfully processed I2P for frame {current_frame_index} with shape {current_pts3d_cam.shape}")
            except (IndexError, KeyError) as e:
                logger.error(f"Error accessing I2P output: {e}. Output structure: {list(i2p_output.keys())} with preds length: {len(i2p_output.get('preds', []))}")
                frame_data_for_history['pts3d_cam'] = None
                frame_data_for_history['conf_cam'] = None
                return None, None, None


            # 2. L2W: Register local points (current_pts3d_cam) to world frame
            # Initialize pose matrix to identity for this frame, in case L2W fails or no KFs
            current_pose_R = np.eye(3)
            current_pose_t = np.zeros((3,1))

            # Skip L2W if we couldn't get valid points from I2P
            if frame_data_for_history['pts3d_cam'] is None:
                logger.warning(f"No valid points from I2P for frame {current_frame_index}. Skipping L2W registration.")
            elif not keyframe_indices:
                logger.warning("No keyframes available for L2W, skipping L2W for current frame.")
            else: # We have valid points and keyframes, proceed with L2W
                # Prepare L2W input: current frame + selected keyframes
                # Keyframes need 'pts3d_world', 'img_tokens', 'img_pos', 'true_shape'
                
                # Select reference keyframes using scene_frame_retrieve (complex for streaming, needs adaptation)
                # Simplified: use last N keyframes or a fixed set for now
                num_ref_keyframes = min(slam_params['num_scene_frame'], len(keyframe_indices))
                # ref_kf_indices = keyframe_indices[-num_ref_keyframes:] # Last N keyframes
                
                # Use scene_frame_retrieve to pick best KFs
                # cand_ref_views are the KFs from processed_frames_history
                candidate_kf_views_for_l2w = []
                for kf_hist_idx in keyframe_indices:
                    hist_entry = processed_frames_history[kf_hist_idx]
                    candidate_kf_views_for_l2w.append({
                        'img_tokens': hist_entry['img_tokens'],
                        'img_pos': hist_entry['img_pos'],
                        'true_shape': hist_entry['true_shape'],
                        'pts3d_world': hist_entry['pts3d_world'], # Crucial for L2W reference
                        'label': f"kf_hist_{kf_hist_idx}"
                        # 'conf_world': hist_entry['conf_world'] # for weighting if scene_frame_retrieve uses it
                    })
                
                # src_view for L2W is current frame's cam points
                # It needs 'img_tokens', 'img_pos', 'true_shape', 'pts3d_cam'
                current_view_for_l2w_src = {
                    'img_tokens': frame_data_for_history['img_tokens'],
                    'img_pos': frame_data_for_history['img_pos'],
                    'true_shape': frame_data_for_history['true_shape'],
                    'pts3d_cam': current_pts3d_cam, # From I2P
                    'label': f"current_{current_frame_index}"
                }

                # Select reference keyframes (views that have 'pts3d_world')
                # slam3r_scene_frame_retrieve expects candi_views (KFs) and src_views (current)
                # It needs i2p_model for correlation scores.
                selected_ref_kf_views, _ = slam3r_scene_frame_retrieve(
                    candidate_kf_views_for_l2w, 
                    [current_view_for_l2w_src], # src_views is a list
                    i2p_model, 
                    sel_num=num_ref_keyframes,
                    # cand_registered_confs=[v.get('conf_world') for v in candidate_kf_views_for_l2w] # if available & used
                )

                l2w_input_views = selected_ref_kf_views + [current_view_for_l2w_src]
                l2w_ref_ids = list(range(len(selected_ref_kf_views))) # IDs of reference KFs in l2w_input_views

                l2w_output = l2w_inference(
                    l2w_input_views, l2w_model, 
                    ref_ids=l2w_ref_ids, 
                    device=device,
                    normalize=slam_params['norm_input_l2w']
                )
                # l2w_output is a list of dicts. Last one is for current_view_for_l2w_src
                # It should contain 'pts3d_in_other_view' (world points) and 'conf'
                current_pts3d_world = l2w_output[-1]['pts3d_in_other_view'] # (1, H, W, 3) or (1, N, 3)
                current_conf_world = l2w_output[-1]['conf'] # (1, H, W) or (1, N)
                frame_data_for_history['pts3d_world'] = current_pts3d_world
                frame_data_for_history['conf_world'] = current_conf_world

                # --- Pose Estimation from L2W registration ---
                # We have:
                # 1. Local points (from I2P, used as source for L2W):
                #    `current_view_for_l2w_src['pts3d_cam']` -> this is `frame_data_for_history['pts3d_cam']`
                #    Confidences: `frame_data_for_history['conf_cam']`
                # 2. World points (output of L2W for current frame):
                #    `current_pts3d_world`
                #    Confidences: `current_conf_world`

                try:
                    # Ensure tensors are on CPU and converted to numpy
                    # Squeeze batch dimension and handle potential H,W,3 to N,3 reshape
                    local_pts_tensor = frame_data_for_history['pts3d_cam'].squeeze(0).cpu() # H,W,3 or N,3
                    local_conf_tensor = frame_data_for_history['conf_cam'].squeeze(0).cpu() # H,W or N
                    world_pts_tensor = current_pts3d_world.squeeze(0).cpu() # H,W,3 or N,3
                    world_conf_tensor = current_conf_world.squeeze(0).cpu() # H,W or N
                    
                    # Reshape to (M, 3) for points and (M,) for confidences
                    P_local_np = local_pts_tensor.reshape(-1, 3).numpy()
                    C_local_np = local_conf_tensor.reshape(-1).numpy()
                    P_world_np = world_pts_tensor.reshape(-1, 3).numpy()
                    C_world_np = world_conf_tensor.reshape(-1).numpy()

                    # Create valid masks based on confidences
                    # Ensure conf_thres_i2p and conf_thres_l2w are present in slam_params
                    conf_i2p = slam_params.get('conf_thres_i2p', 1.5)
                    conf_l2w = slam_params.get('conf_thres_l2w', 12.0) # Use L2W threshold for world points

                    valid_mask_local = C_local_np > conf_i2p
                    valid_mask_world = C_world_np > conf_l2w
                    combined_valid_mask = valid_mask_local & valid_mask_world
                    
                    num_valid_points = np.sum(combined_valid_mask)

                    if num_valid_points >= 3: # Need at least 3 points for SVD
                        P_local_filtered = P_local_np[combined_valid_mask]
                        P_world_filtered = P_world_np[combined_valid_mask]

                        R_estimated, t_estimated = estimate_rigid_transform_svd(P_local_filtered, P_world_filtered)
                        
                        # Update current_pose_R and current_pose_t if estimation is valid
                        if not (np.allclose(R_estimated, np.eye(3)) and np.allclose(t_estimated, np.zeros((3,1)))):
                             current_pose_R = R_estimated
                             current_pose_t = t_estimated.reshape(3,1) # Ensure t is (3,1)
                             logger.info(f"Frame {current_frame_index}: Estimated pose from {num_valid_points} points.")
                        else:
                            logger.warning(f"Frame {current_frame_index}: SVD pose estimation resulted in identity/zero with {num_valid_points} points. Using previous or identity pose.")
                    else:
                        logger.warning(f"Frame {current_frame_index}: Not enough valid points ({num_valid_points}) for pose estimation after filtering. Using previous or identity pose.")
                        # Keep current_pose_R, current_pose_t as identity or potentially use last known good pose (more complex)

                except Exception as e_pose:
                    logger.error(f"Error during pose estimation for frame {current_frame_index}: {e_pose}", exc_info=True)
                    # current_pose_R, current_pose_t remain identity
                
                # Construct the 4x4 pose matrix T_world_cam (transforms points from camera to world)
                # raw_pose_matrix = [ [R00, R01, R02, t0],
                #                     [R10, R11, R12, t1],
                #                     [R20, R21, R22, t2],
                #                     [  0,   0,   0,  1] ]
                pose_matrix_4x4 = np.eye(4)
                pose_matrix_4x4[:3, :3] = current_pose_R
                pose_matrix_4x4[:3, 3] = current_pose_t.squeeze() # t is (3,1), squeeze to (3,)
                frame_data_for_history['raw_pose_matrix'] = pose_matrix_4x4.tolist()
                
                # Add to world_point_cloud_buffer (filter by L2W confidence)
                # Note: The points added to buffer are already in world coordinates from L2W
                # The 'valid_mask_l2w' was computed above as 'world_conf_tensor > conf_l2w'
                # This filtering is for the point cloud, separate from pose estimation points.
                valid_mask_for_pc = world_conf_tensor > slam_params.get('conf_thres_l2w', 12.0)
                points_to_add_world = P_world_np[valid_mask_for_pc] # Use P_world_np directly
                
                world_point_cloud_buffer.extend(points_to_add_world.tolist())
                temp_points_xyz_list = points_to_add_world.tolist() # For current frame output

                # Keyframe decision logic (simplified)
                # Based on stride, or confidence, or if L2W was successful
                if current_frame_index % active_kf_stride == 0: # Simplified KF selection
                    is_new_keyframe = True 
                    # More advanced: check motion, L2W confidence, etc.
                    # from recon.py: update buffering set using confidence scores.
                    # This is complex for streaming, needs careful state management.
                    if is_new_keyframe:
                        keyframe_indices.append(current_frame_index)
                        frame_data_for_history['keyframe_id'] = f"kf_{len(keyframe_indices)-1}"
                        keyframe_id_for_output = frame_data_for_history['keyframe_id']
                        logger.info(f"Frame {current_frame_index} selected as Keyframe {frame_data_for_history['keyframe_id']}. Total KFs: {len(keyframe_indices)}")
                        if rerun_connected:
                            rr.log("log/info", rr.TextLog(text=f"Keyframe {frame_data_for_history['keyframe_id']} added at frame {current_frame_index}", color=[0, 255, 0]))
                            # Log keyframe pose distinctly if desired
                            kf_pose_matrix = np.array(frame_data_for_history['raw_pose_matrix'])
                            kf_position = kf_pose_matrix[:3, 3]
                            kf_orientation_m = kf_pose_matrix[:3,:3]
                            kf_orientation_q = matrix_to_quaternion(kf_orientation_m) # x,y,z,w

                            rr.log(
                                f"world/keyframes/{frame_data_for_history['keyframe_id']}",
                                rr.Transform3D(
                                    translation=kf_position,
                                    rotation=rr.Quaternion(xyzw=kf_orientation_q)
                                )
                            )
                            rr.log(f"world/keyframes/{frame_data_for_history['keyframe_id']}/center", rr.Points3D(positions=[[0,0,0]], radii=[0.02], colors=[[255,255,0]])) # Use Points3D for single points
        
        # Update history and index
        processed_frames_history.append(frame_data_for_history)
        current_frame_index += 1

        # --- Prepare output data ---
        # Pose data uses the 'raw_pose_matrix' from frame_data_for_history, which is now updated.
        current_pose_matrix_list = frame_data_for_history.get('raw_pose_matrix', np.eye(4).tolist())
        current_pose_matrix_np = np.array(current_pose_matrix_list)
        
        position_np_arr = current_pose_matrix_np[:3, 3]
        orientation_m = current_pose_matrix_np[:3,:3]
        orientation_q_xyzw = matrix_to_quaternion(orientation_m) # x,y,z,w

        if rerun_connected:
            # Log current camera pose
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=position_np_arr,
                    rotation=rr.Quaternion(xyzw=orientation_q_xyzw)
                )
            )

            # Log local points (from I2P, in camera frame)
            if frame_data_for_history['pts3d_cam'] is not None and frame_data_for_history['conf_cam'] is not None:
                local_pts_tensor_viz = frame_data_for_history['pts3d_cam'].squeeze(0).cpu() # H,W,3 or N,3
                local_conf_tensor_viz = frame_data_for_history['conf_cam'].squeeze(0).cpu() # H,W or N
                
                P_local_np_viz = local_pts_tensor_viz.reshape(-1, 3).numpy()
                C_local_np_viz = local_conf_tensor_viz.reshape(-1).numpy()
                
                conf_i2p_viz = slam_params.get('conf_thres_i2p', 1.5)
                valid_mask_local_viz = C_local_np_viz > conf_i2p_viz
                
                if np.any(valid_mask_local_viz):
                    P_local_filtered_viz = P_local_np_viz[valid_mask_local_viz]
                    # rr.log_points("world/camera/local_scan", P_local_filtered_viz, colors=[255, 0, 255], radii=0.005) # Magenta for local scan
                    rr.log("world/camera/local_scan", rr.Points3D(positions=P_local_filtered_viz, colors=[255, 0, 255], radii=0.005))

            # Log incremental world points (from L2W)
            if temp_points_xyz_list: # This list contains newly added world points
                # rr.log_points("world/points", np.array(temp_points_xyz_list), colors=[0, 0, 255], radii=0.007) # Blue for world points
                rr.log("world/points", rr.Points3D(positions=np.array(temp_points_xyz_list), colors=[0, 0, 255], radii=0.007))
            

        pose_data = {
            "timestamp_ns": timestamp_ns,
            "processing_timestamp": str(datetime.now().timestamp()),
            "position": {"x": float(position_np_arr[0]), "y": float(position_np_arr[1]), "z": float(position_np_arr[2])},
            "orientation": {"x": float(orientation_q_xyzw[0]), "y": float(orientation_q_xyzw[1]), "z": float(orientation_q_xyzw[2]), "w": float(orientation_q_xyzw[3])},
            "raw_pose_matrix": current_pose_matrix_list
        }
        
        point_cloud_data = {
            "timestamp_ns": timestamp_ns,
            "processing_timestamp": str(datetime.now().timestamp()),
            "points": temp_points_xyz_list # Points from current frame's L2W output (world coords) or init
        }
        
        # For reconstruction_update_data, send current frame's world points as an incremental update
        # No faces, as SLAM3R here is point-based for this flow.
        reconstruction_update_data = {
            "timestamp_ns": timestamp_ns,
            "processing_timestamp": str(datetime.now().timestamp()),
            "type": "points_update_incremental", # Changed from mesh to points
            "vertices": temp_points_xyz_list, # Use current frame's world points
            "faces": [], # No faces
            "keyframe_id": keyframe_id_for_output 
        }
        
        # Optional: Prune processed_frames_history if it gets too large and items are not needed
        # (e.g. if not a keyframe and its tokens/points are no longer needed for L2W lookback)

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total processing time for frame {current_frame_index}: {total_time:.4f} seconds")

        return pose_data, point_cloud_data, reconstruction_update_data

    except Exception as e:
        logger.error(f"Error during SLAM3R processing for frame {timestamp_ns}: {e}", exc_info=True)
        return None, None, None

async def on_video_frame_message(message: aio_pika.IncomingMessage, exchanges):
    """
    Callback function to handle incoming video frame messages from RabbitMQ.
    It decodes the image, processes it using `process_image_with_slam3r`,
    and publishes the results (pose, point cloud, reconstruction update)
    to their respective RabbitMQ exchanges.

    Args:
        message (aio_pika.IncomingMessage): The received RabbitMQ message.
        exchanges (dict): A dictionary of declared RabbitMQ exchanges for publishing.
    """
    async with message.process(): # Acknowledges the message upon successful processing
        try:
            image_data = message.body
            headers = message.headers
            timestamp_ns_str = headers.get("timestamp_ns")

            if not timestamp_ns_str:
                logger.warning("Received frame without 'timestamp_ns' in headers. Skipping.")
                return
            
            timestamp_ns = int(timestamp_ns_str)
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_np is None:
                logger.warning(f"Failed to decode image for timestamp {timestamp_ns}. Skipping.")
                return
            
            pose, point_cloud, recon_update = await process_image_with_slam3r(img_np, timestamp_ns, headers)

            if pose and SLAM3R_POSE_EXCHANGE_OUT in exchanges:
                pose_message = aio_pika.Message(
                    body=json.dumps(pose).encode(),
                    content_type="application/json",
                    headers={"source_timestamp_ns": str(timestamp_ns)}
                )
                await exchanges[SLAM3R_POSE_EXCHANGE_OUT].publish(pose_message, routing_key="")

            if point_cloud and SLAM3R_POINTCLOUD_EXCHANGE_OUT in exchanges:
                pc_message = aio_pika.Message(
                    body=json.dumps(point_cloud).encode(), # Consider more efficient serialization for large point clouds
                    content_type="application/json",
                    headers={"source_timestamp_ns": str(timestamp_ns)}
                )
                await exchanges[SLAM3R_POINTCLOUD_EXCHANGE_OUT].publish(pc_message, routing_key="")
            
            if recon_update and SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT in exchanges:
                recon_message = aio_pika.Message(
                    body=json.dumps(recon_update).encode(),
                    content_type="application/json",
                    headers={"source_timestamp_ns": str(timestamp_ns)}
                )
                await exchanges[SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT].publish(recon_message, routing_key="")

        except Exception as e:
            logger.error(f"Error processing video frame message: {e}", exc_info=True)

async def on_restart_message(message: aio_pika.IncomingMessage):
    """
    Callback function to handle restart messages from RabbitMQ.
    It triggers the re-initialization of the SLAM system by resetting global state
    variables (including models, parameters, frame history, and session state)
    and then calling `initialize_slam_system`. This allows for a clean restart
    of the SLAM process without restarting the Docker container.

    Args:
        message (aio_pika.IncomingMessage): The received RabbitMQ restart message.

    Global variables modified:
        - is_slam_system_initialized, slam_system
        - i2p_model, l2w_model, slam_params
        - processed_frames_history, keyframe_indices, world_point_cloud_buffer
        - current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer,
        - active_kf_stride, reference_view_id_current_session
        - rerun_connected
    """
    global is_slam_system_initialized, slam_system
    global i2p_model, l2w_model, slam_params
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer
    global active_kf_stride, reference_view_id_current_session
    global rerun_connected

    async with message.process():
        try:
            msg_body = json.loads(message.body.decode())
            logger.info(f"Received restart message: {msg_body}. Re-initializing SLAM system and resetting session state.")
            
            is_slam_system_initialized = False # Force re-initialization of models and params
            
            # Clear SLAM3R models (they will be reloaded by initialize_slam_system)
            i2p_model = None
            l2w_model = None
            slam_params = {}
            
            # Reset per-session SLAM state
            processed_frames_history = []
            keyframe_indices = []
            world_point_cloud_buffer = []
            current_frame_index = 0
            is_slam_initialized_for_session = False # Critical to reset this
            slam_initialization_buffer = []
            active_kf_stride = 1 # Reset to default or value from config
            reference_view_id_current_session = 0
            
            # Rerun connection status is not reset here, as init will try to reconnect.
            # If a specific Rerun session reset is needed, rr.save() or re-init could be used.
            # For now, keep existing connection or let initialize_slam_system handle re-connection.

            if device and device.type == 'cuda':
                torch.cuda.empty_cache() # Clear CUDA cache
            
            # Re-initialize the SLAM system, which will also attempt to reconnect Rerun
            await initialize_slam_system() 
        except Exception as e:
            logger.error(f"Error processing restart message: {e}", exc_info=True)

async def main():
    """
    Main asynchronous function for the SLAM3R processor service.
    It performs the following steps:
    1. Initializes the SLAM system (`initialize_slam_system`).
    2. Establishes a robust connection to RabbitMQ, with retries.
    3. Declares necessary RabbitMQ exchanges (input and output).
    4. Declares and binds queues for video frames and restart commands.
    5. Starts consuming messages from these queues, assigning them to their
       respective handler functions (`on_video_frame_message`, `on_restart_message`).
    6. Keeps the service running indefinitely until interrupted.
    7. Handles graceful shutdown by closing the RabbitMQ connection.
    """
    # Initial attempt to load models and initialize system
    # This ensures that if RabbitMQ connection fails initially, we still try to load SLAM
    # so that subsequent restart messages or reconnections can use an initialized system.
    await initialize_slam_system() 

    connection = None
    while True: 
        try:
            connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=30, heartbeat=60)
            logger.info(f"Connected to RabbitMQ at {RABBITMQ_URL}")
            break
        except (aio_pika.exceptions.AMQPConnectionError, ConnectionRefusedError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}. Retrying in 10 seconds...")
            await asyncio.sleep(10)
    
    async with connection:
        channel = await connection.channel()
        # Set prefetch_count to 1 to ensure we process one frame at a time
        # This prevents message buildup when processing is slow (which it is during initialization)
        await channel.set_qos(prefetch_count=1) 

        exchanges = {}
        # Declare input exchanges
        exchanges[VIDEO_FRAMES_EXCHANGE_IN] = await channel.declare_exchange(
            VIDEO_FRAMES_EXCHANGE_IN, aio_pika.ExchangeType.FANOUT, durable=True)
        exchanges[RESTART_EXCHANGE_IN] = await channel.declare_exchange(
            RESTART_EXCHANGE_IN, aio_pika.ExchangeType.FANOUT, durable=True)
        
        # Declare output exchanges
        exchanges[SLAM3R_POSE_EXCHANGE_OUT] = await channel.declare_exchange(
            SLAM3R_POSE_EXCHANGE_OUT, aio_pika.ExchangeType.FANOUT, durable=True)
        exchanges[SLAM3R_POINTCLOUD_EXCHANGE_OUT] = await channel.declare_exchange(
            SLAM3R_POINTCLOUD_EXCHANGE_OUT, aio_pika.ExchangeType.FANOUT, durable=True)
        exchanges[SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT] = await channel.declare_exchange(
            SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT, aio_pika.ExchangeType.FANOUT, durable=True)

        logger.info("Declared RabbitMQ exchanges.")

        video_queue_name = "slam3r_video_frames_queue"
        video_queue = await channel.declare_queue(name=video_queue_name, durable=True, auto_delete=False)
        await video_queue.bind(exchanges[VIDEO_FRAMES_EXCHANGE_IN])
        await video_queue.consume(lambda msg: on_video_frame_message(msg, exchanges))
        logger.info(f"Consuming from '{VIDEO_FRAMES_EXCHANGE_IN}' via '{video_queue_name}'")

        restart_queue_name = "slam3r_restart_queue"
        restart_queue = await channel.declare_queue(name=restart_queue_name, durable=True, auto_delete=False)
        await restart_queue.bind(exchanges[RESTART_EXCHANGE_IN])
        await restart_queue.consume(on_restart_message)
        logger.info(f"Consuming from '{RESTART_EXCHANGE_IN}' via '{restart_queue_name}'")

        logger.info("SLAM3R processor service started. Waiting for messages...")
        try:
            await asyncio.Future() # Keep the main coroutine alive
        except (asyncio.CancelledError, KeyboardInterrupt) : # Added KeyboardInterrupt
            logger.info("SLAM3R processor shutting down or interrupted.")
        finally:
            # No explicit shutdown for models like i2p_model. Python's GC will handle.
            if connection and not connection.is_closed:
                await connection.close()
                logger.info("RabbitMQ connection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("SLAM3R processor stopped by user.")
    except Exception as e:
        # Critical log for unhandled exceptions at the top level, ensuring visibility.
        logger.critical(f"Unhandled exception in SLAM3R processor: {e}", exc_info=True) 