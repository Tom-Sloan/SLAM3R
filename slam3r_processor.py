# slam3r/slam3r_processor.py
# This script is the entry point for the SLAM3R Docker service.
# It connects to RabbitMQ, consumes RGB images, processes them using SLAM3R,
# and publishes the output (poses, point clouds, maps) to respective exchanges.
# Influenced by user prompts regarding RabbitMQ integration, SLAM3R model handling,
# desired outputs for visualization, full implementation of processing logic,
# and addition of SLAM3R_engine imports.
# Current Prompt: Implement all todos and placeholders.

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
from torchvision import transforms # Added for potential future use with SLAM3R utils

# Configure logging (Moved Up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import SLAM3R engine components
SLAM3R_ENGINE_AVAILABLE = False
try:
    # Imports from SLAM3R_engine.recon (similar to SLAM3R_engine/app.py)
    from SLAM3R_engine.recon import (
        get_img_tokens as slam3r_get_img_tokens,
        initialize_scene as slam3r_initialize_scene,
        adapt_keyframe_stride as slam3r_adapt_keyframe_stride,
        i2p_inference_batch,
        l2w_inference,
        scene_frame_retrieve as slam3r_scene_frame_retrieve
    )
    # Imports from SLAM3R_engine.slam3r.models
    from SLAM3R_engine.slam3r.models import Image2PointsModel, Local2WorldModel
    
    # Optional: If SLAM3R's specific image transformation is needed for better alignment
    # from SLAM3R_engine.slam3r.utils.recon_utils import transform_img

    SLAM3R_ENGINE_AVAILABLE = True
    logger.info("Successfully imported SLAM3R engine components from SLAM3R_engine.")
except ImportError as e:
    logger.error(f"Failed to import SLAM3R engine components: {e}. SLAM3R processing will be disabled. Ensure SLAM3R_engine is in the PYTHONPATH and all dependencies are installed.")
    # Define dummy functions or classes if engine is not available to prevent NameErrors
    # The SLAM3R_ENGINE_AVAILABLE flag should primarily guard against their use.
    def slam3r_get_img_tokens(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    def slam3r_initialize_scene(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    def slam3r_adapt_keyframe_stride(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    def i2p_inference_batch(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    def l2w_inference(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    def slam3r_scene_frame_retrieve(*args, **kwargs): raise NotImplementedError("SLAM3R engine not available due to import error")
    class Image2PointsModel:
        @staticmethod
        def from_pretrained(path): raise NotImplementedError("SLAM3R engine not available due to import error")
    class Local2WorldModel:
        @staticmethod
        def from_pretrained(path): raise NotImplementedError("SLAM3R engine not available due to import error")

# --- Environment Variables ---
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE_IN = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
RESTART_EXCHANGE_IN = os.getenv("RESTART_EXCHANGE", "restart_exchange")

SLAM3R_POSE_EXCHANGE_OUT = os.getenv("SLAM3R_POSE_EXCHANGE", "slam3r_pose_exchange")
SLAM3R_POINTCLOUD_EXCHANGE_OUT = os.getenv("SLAM3R_POINTCLOUD_EXCHANGE", "slam3r_pointcloud_exchange")
SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT = os.getenv("SLAM3R_RECONSTRUCTION_VIS_EXCHANGE", "slam3r_reconstruction_vis_exchange")

CHECKPOINTS_DIR = os.getenv("SLAM3R_CHECKPOINTS_DIR", "/checkpoints_mount")
SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER = os.getenv("SLAM3R_CONFIG_FILE", "/app/SLAM3R_engine/configs/wild.yaml") # Default path to SLAM3R config
CAMERA_INTRINSICS_FILE_PATH = os.getenv("CAMERA_INTRINSICS_FILE", "/app/SLAM3R_engine/camera_intrinsics.yaml") # Path to camera intrinsics YAML

# Image Processing Config
TARGET_IMAGE_WIDTH = int(os.getenv("TARGET_IMAGE_WIDTH", "640")) # Width SLAM3R expects
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", "480")) # Height SLAM3R expects

# --- Initialization Quality Thresholds ---
# These thresholds are used to check the quality of the initial SLAM map.
# If the initial map quality is below these, initialization will be re-attempted.
# Setting defaults to 0 to effectively bypass the check for initial experiments.
INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE = float(os.getenv("INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE", "0.0"))
INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS = int(os.getenv("INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS", "0"))

# --- SLAM3R Global State ---
slam_system = None
is_slam_system_initialized = False
camera_intrinsics_dict = None
device = None

# SLAM3R specific models and state
i2p_model = None
l2w_model = None
slam_params = {} # To store parameters from SLAM3R config YAML

# Per-session SLAM state (needs reset on restart)
processed_frames_history = [] # Stores dicts of { 'img_tensor', 'img_tokens', 'true_shape', 'img_pos', 'pts3d_cam', 'conf_cam', 'pts3d_world', 'conf_world', 'timestamp_ns', 'keyframe_id', 'raw_pose_matrix' }
keyframe_indices = [] # Indices into processed_frames_history that are keyframes
world_point_cloud_buffer = [] # Aggregated world points (e.g., list of [x,y,z] points)
current_frame_index = 0 # Incremental index for incoming frames
is_slam_initialized_for_session = False # Tracks if initial scene setup is done for current session
slam_initialization_buffer = [] # Buffer for frames during initial SLAM setup
reference_view_id_current_session = 0 # Reference view ID for current SLAM session
active_kf_stride = 1 # Current keyframe stride, might be adapted

# Transformation utilities
def matrix_to_quaternion(matrix_3x3):
    """
    Convert a 3x3 rotation matrix to a quaternion (x, y, z, w).
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
    such that: points_tgt = R @ points_src + t
    Args:
        points_src_np (np.ndarray): Source points (N, 3).
        points_tgt_np (np.ndarray): Target points (N, 3), corresponding to points_src_np.
    Returns:
        Tuple[np.ndarray, np.ndarray]: (R, t) where R is (3,3) rotation matrix, t is (3,1) translation vector.
                                       Returns (np.eye(3), np.zeros((3,1))) on failure.
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
    return {
        "position": {"x": float(position_np[0]), "y": float(position_np[1]), "z": float(position_np[2])},
        "orientation": {"x": float(orientation_quat_np[0]), "y": float(orientation_quat_np[1]), "z": float(orientation_quat_np[2]), "w": float(orientation_quat_np[3])}
    }

def load_camera_intrinsics(file_path):
    """Loads camera intrinsics from a YAML file."""
    try:
        with open(file_path, 'r') as f:
            intrinsics = yaml.safe_load(f)
        if not all(k in intrinsics for k in ['fx', 'fy', 'cx', 'cy']):
            logger.error(f"Camera intrinsics file {file_path} is missing one or more keys (fx, fy, cx, cy).")
            return None
        logger.info(f"Loaded camera intrinsics from {file_path}: {intrinsics}")
        return intrinsics
    except FileNotFoundError:
        logger.warning(f"Camera intrinsics file not found at {file_path}. Proceeding without specific intrinsics.")
        return None
    except Exception as e:
        logger.error(f"Error loading camera intrinsics from {file_path}: {e}", exc_info=True)
        return None

async def initialize_slam_system():
    """Loads SLAM3R models and initializes the SLAM system."""
    global slam_system, is_slam_system_initialized, camera_intrinsics_dict, device
    global i2p_model, l2w_model, slam_params
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer, reference_view_id_current_session, active_kf_stride
    
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
        # Define the HuggingFace model IDs and paths
        i2p_model_id = "siyan824/slam3r_i2p"
        l2w_model_id = "siyan824/slam3r_l2w"
        path_i2p_model_dir = os.path.join(CHECKPOINTS_DIR, i2p_model_id)
        path_l2w_model_dir = os.path.join(CHECKPOINTS_DIR, l2w_model_id)

        # Ensure checkpoint directory exists
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # Download I2P model if missing
        if not os.path.isdir(path_i2p_model_dir):
            logger.info(f"Image2Points model directory not found at: {path_i2p_model_dir}. Downloading from HuggingFace...")
            try:
                # Create model directory structure
                os.makedirs(path_i2p_model_dir, exist_ok=True)
                
                # Use the HuggingFace Hub's from_pretrained functionality
                # This will download and initialize the model
                i2p_model = Image2PointsModel.from_pretrained(i2p_model_id)
                logger.info(f"Successfully downloaded Image2Points model from HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to download Image2Points model: {e}")
                is_slam_system_initialized = False
                return False
        
        # Download L2W model if missing
        if not os.path.isdir(path_l2w_model_dir):
            logger.info(f"Local2World model directory not found at: {path_l2w_model_dir}. Downloading from HuggingFace...")
            try:
                # Create model directory structure
                os.makedirs(path_l2w_model_dir, exist_ok=True)
                
                # Use the HuggingFace Hub's from_pretrained functionality
                # This will download and initialize the model
                l2w_model = Local2WorldModel.from_pretrained(l2w_model_id)
                logger.info(f"Successfully downloaded Local2World model from HuggingFace Hub")
            except Exception as e:
                logger.error(f"Failed to download Local2World model: {e}")
                is_slam_system_initialized = False
                return False

        logger.info(f"Loading Image2Points model from {path_i2p_model_dir}")
        i2p_model = Image2PointsModel.from_pretrained(path_i2p_model_dir).to(device).eval()
        logger.info("Image2Points model loaded.")

        logger.info(f"Loading Local2World model from {path_l2w_model_dir}")
        l2w_model = Local2WorldModel.from_pretrained(path_l2w_model_dir).to(device).eval()
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

def preprocess_image(image_np, target_width, target_height, intrinsics=None):
    """Preprocesses the image for SLAM3R."""
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
        adjusted_intrinsics = {
            "fx": fx * scale_x, "fy": fy * scale_y,
            "cx": cx * scale_x, "cy": cy * scale_y
        }
        # Convert intrinsics to tensor if SLAM3R expects it
        # adjusted_intrinsics_tensor = torch.tensor([adjusted_intrinsics['fx'], adjusted_intrinsics['fy'], adjusted_intrinsics['cx'], adjusted_intrinsics['cy']], dtype=torch.float32)

    return img_tensor.to(device), adjusted_intrinsics # Return tensor and dict for now

async def process_image_with_slam3r(image_np, timestamp_ns, headers):
    """Processes an image with the initialized SLAM3R system and returns outputs."""
    global i2p_model, l2w_model, slam_params, camera_intrinsics_dict, device
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer, reference_view_id_current_session, active_kf_stride
    
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
                        _, view_token, view_pos = slam3r_get_img_tokens(single_view_list_for_tokens, i2p_model)
                        
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
                    initial_pcds_tensors, initial_confs_tensors, init_ref_id = slam3r_initialize_scene(
                        initial_input_views_for_slam[::active_kf_stride], # Pass only keyframes based on stride
                        i2p_model,
                        winsize=min(slam_params['initial_winsize'], len(initial_input_views_for_slam[::active_kf_stride])),
                        conf_thres=slam_params['conf_thres_i2p'], # This is for point generation, not the quality check threshold
                        return_ref_id=True
                    )
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
                            
                            processed_frames_history[history_idx]['img_tokens'] = initial_input_views_for_slam[i]['img_tokens']
                            processed_frames_history[history_idx]['img_pos'] = initial_input_views_for_slam[i]['img_pos']

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
            
            current_i2p_input_view = { # This is the 'view' structure for i2p_inference_batch
                'img_tokens': frame_data_for_history['img_tokens'],
                'img_pos': frame_data_for_history['img_pos'],
                'true_shape': frame_data_for_history['true_shape'],
                'label': f"frame_{current_frame_index}"
            }
            # i2p_inference_batch takes list_of_lists_of_views.
            # ref_id is local to the inner list.
            i2p_output = i2p_inference_batch([[current_i2p_input_view]], i2p_model, ref_id=0, tocpu=False, unsqueeze=False)
            
            # i2p_output['preds'] is a list (outer batch) of lists (inner window) of dicts
            # Each dict has 'pts3d', 'conf'
            current_pts3d_cam = i2p_output['preds'][0][0]['pts3d']  # Shape: (1, H, W, 3) or (1, N, 3)
            current_conf_cam = i2p_output['preds'][0][0]['conf']    # Shape: (1, H, W) or (1, N)
            frame_data_for_history['pts3d_cam'] = current_pts3d_cam
            frame_data_for_history['conf_cam'] = current_conf_cam


            # 2. L2W: Register local points (current_pts3d_cam) to world frame
            # Initialize pose matrix to identity for this frame, in case L2W fails or no KFs
            current_pose_R = np.eye(3)
            current_pose_t = np.zeros((3,1))

            if keyframe_indices: # Ensure we have keyframes to register against
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
            else: # Should not happen if initialization was successful
                logger.warning("No keyframes available for L2W, skipping L2W for current frame.")
        
        # Update history and index
        processed_frames_history.append(frame_data_for_history)
        current_frame_index += 1

        # --- Prepare output data ---
        # Pose data uses the 'raw_pose_matrix' from frame_data_for_history, which is now updated.
        current_pose_matrix_list = frame_data_for_history.get('raw_pose_matrix', np.eye(4).tolist())
        current_pose_matrix_np = np.array(current_pose_matrix_list)
        
        position = current_pose_matrix_np[:3, 3]
        orientation_q = matrix_to_quaternion(current_pose_matrix_np[:3,:3]) # x,y,z,w

        pose_data = {
            "timestamp_ns": timestamp_ns,
            "processing_timestamp": str(datetime.now().timestamp()),
            "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
            "orientation": {"x": float(orientation_q[0]), "y": float(orientation_q[1]), "z": float(orientation_q[2]), "w": float(orientation_q[3])},
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

        return pose_data, point_cloud_data, reconstruction_update_data

    except Exception as e:
        logger.error(f"Error during SLAM3R processing for frame {timestamp_ns}: {e}", exc_info=True)
        return None, None, None

async def on_video_frame_message(message: aio_pika.IncomingMessage, exchanges):
    async with message.process():
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
    global is_slam_system_initialized, slam_system
    global i2p_model, l2w_model, slam_params # Added SLAM3R specific models and params
    global processed_frames_history, keyframe_indices, world_point_cloud_buffer
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer, active_kf_stride, reference_view_id_current_session

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

            if device and device.type == 'cuda':
                torch.cuda.empty_cache() # Clear CUDA cache
            
            await initialize_slam_system() # This will reload models and parameters
        except Exception as e:
            logger.error(f"Error processing restart message: {e}", exc_info=True)

async def main():
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
        # Lower prefetch_count if SLAM processing is slow to avoid message buildup
        await channel.set_qos(prefetch_count=5) 

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
        logger.critical(f"Unhandled exception in SLAM3R processor: {e}", exc_info=True) 