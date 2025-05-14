# slam3r/slam3r_processor.py
# – Consumes RGB frames from RabbitMQ
# – Runs the SLAM3R incremental pipeline (I2P → L2W → pose SVD → KF logic)
# – Publishes pose / point‑cloud / recon‑viz messages (and optional Rerun stream)
#
# Default env‑vars are inlined so the script works out‑of‑the‑box.

import asyncio, gzip, json, logging, os, random, time
from datetime import datetime
from pathlib import Path

import aio_pika, cv2, numpy as np, torch, yaml
import rerun as rr

# ───────────────────────────────────────────────────────────────────────────────
# Imports from SLAM3R engine
# ───────────────────────────────────────────────────────────────────────────────
SLAM3R_ENGINE_AVAILABLE = False
try:
    from SLAM3R_engine.recon import (
        get_img_tokens            as slam3r_get_img_tokens,
        initialize_scene          as slam3r_initialize_scene,
        adapt_keyframe_stride     as slam3r_adapt_keyframe_stride,
        i2p_inference_batch,
        l2w_inference,
        scene_frame_retrieve      as slam3r_scene_frame_retrieve,
    )
    from SLAM3R_engine.slam3r.models import Image2PointsModel, Local2WorldModel
    SLAM3R_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.error("SLAM3R_engine not importable: %s", e)
    raise e

# ───────────────────────────────────────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("slam3r_processor")

# ───────────────────────────────────────────────────────────────────────────────
# Environment / config (with sane defaults)
# ───────────────────────────────────────────────────────────────────────────────
RABBITMQ_URL                       = os.getenv("RABBITMQ_URL", "amqp://rabbitmq")
VIDEO_FRAMES_EXCHANGE_IN           = os.getenv("VIDEO_FRAMES_EXCHANGE", "video_frames_exchange")
RESTART_EXCHANGE_IN                = os.getenv("RESTART_EXCHANGE",    "restart_exchange")

SLAM3R_POSE_EXCHANGE_OUT           = os.getenv("SLAM3R_POSE_EXCHANGE",              "slam3r_pose_exchange")
SLAM3R_POINTCLOUD_EXCHANGE_OUT     = os.getenv("SLAM3R_POINTCLOUD_EXCHANGE",        "slam3r_pointcloud_exchange")
SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT = os.getenv("SLAM3R_RECONSTRUCTION_VIS_EXCHANGE",
                                                   "slam3r_reconstruction_vis_exchange")
OUTPUT_TO_RABBITMQ                 = os.getenv("SLAM3R_OUTPUT_TO_RABBITMQ", "false").lower() == "true"

CHECKPOINTS_DIR                    = os.getenv("SLAM3R_CHECKPOINTS_DIR", "/checkpoints_mount")
SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER= os.getenv("SLAM3R_CONFIG_FILE", "/app/SLAM3R_engine/configs/wild.yaml")
CAMERA_INTRINSICS_FILE_PATH        = os.getenv("CAMERA_INTRINSICS_FILE", "/app/SLAM3R_engine/configs/camera_intrinsics.yaml")

DEFAULT_MODEL_INPUT_RESOLUTION = 224
INFERENCE_WINDOW_BATCH = 1

TARGET_IMAGE_WIDTH  = int(os.getenv("TARGET_IMAGE_WIDTH",  DEFAULT_MODEL_INPUT_RESOLUTION))
TARGET_IMAGE_HEIGHT = int(os.getenv("TARGET_IMAGE_HEIGHT", DEFAULT_MODEL_INPUT_RESOLUTION))

INIT_QUALITY_MIN_CONF   = float(os.getenv("INITIALIZATION_QUALITY_MIN_AVG_CONFIDENCE", "1.0"))
INIT_QUALITY_MIN_POINTS = int  (os.getenv("INITIALIZATION_QUALITY_MIN_TOTAL_VALID_POINTS", "100"))

# ───────────────────────────────────────────────────────────────────────────────
# Global run‑time state
# ───────────────────────────────────────────────────────────────────────────────
device                   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
i2p_model                = l2w_model = None
slam_params              = {}
is_slam_system_initialized = False

processed_frames_history : list = []
keyframe_indices         : list = []
world_point_cloud_buffer : list = []   # (xyz, rgb)
camera_positions         : list = []

current_frame_index              = 0
slam_initialization_buffer : list = []
is_slam_initialized_for_session  = False
active_kf_stride                 = 1

rerun_connected = False  # live viewer flag

# ───────────────────────────────────────────────────────────────────────────────
# Camera intrinsics helper
# ───────────────────────────────────────────────────────────────────────────────
def load_yaml_intrinsics(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text())
        return {k: float(data[k]) for k in ("fx", "fy", "cx", "cy")}
    except Exception as e:
        logger.warning("Failed to parse camera intrinsics YAML: %s", e)
        return None

camera_intrinsics = load_yaml_intrinsics(CAMERA_INTRINSICS_FILE_PATH) or {}
if not camera_intrinsics:
    logger.warning("No camera intrinsics found – skipping frustum logging.")

# ───────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────────────────────
def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    q = np.empty(4); t = np.trace(m)
    if t > 0:
        t = np.sqrt(t + 1); q[3] = 0.5 * t; t = 0.5 / t
        q[0] = (m[2,1]-m[1,2])*t; q[1] = (m[0,2]-m[2,0])*t; q[2] = (m[1,0]-m[0,1])*t
    else:
        i = np.argmax(np.diag(m)); j, k = (i+1)%3, (i+2)%3
        t = np.sqrt(m[i,i]-m[j,j]-m[k,k]+1)
        q[i] = 0.5*t; t = 0.5/t
        q[3] = (m[k,j]-m[j,k])*t; q[j] = (m[j,i]+m[i,j])*t; q[k] = (m[k,i]+m[i,k])*t
    return q

def cv_to_rerun_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.copy()
    xyz[:,1] *= -1.0   # flip vertical
    xyz[:,0] *= -1.0   # flip left/right
    return xyz

def estimate_rigid_transform_svd(P: np.ndarray, Q: np.ndarray):
    if len(P) < 3:
        return np.eye(3), np.zeros((3,1))
    Pc, Qc = P-P.mean(0), Q-Q.mean(0)
    U,_,Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2] *= -1; R = Vt.T @ U.T
    t = Q.mean(0,keepdims=True).T - R @ P.mean(0,keepdims=True).T
    return R, t

def preprocess_image(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr,(TARGET_IMAGE_WIDTH,TARGET_IMAGE_HEIGHT)),cv2.COLOR_BGR2RGB)
    tensor  = torch.tensor(img_rgb,dtype=torch.float32).permute(2,0,1)/255.0
    return tensor.to(device)

def colors_from_image(tensor_chw: torch.Tensor) -> np.ndarray:
    return (tensor_chw.permute(1,2,0).cpu().numpy()*255).astype(np.uint8).reshape(-1,3)

def log_points_to_rerun(label:str, pts_col:list, radius:float=0.007):
    if not rerun_connected or not pts_col:
        return
    xyz, rgb = map(np.asarray, zip(*pts_col))
    rr.log(label, rr.Points3D(positions=cv_to_rerun_xyz(xyz.astype(np.float32)),
                              colors=rgb.astype(np.uint8),
                              radii=np.full(len(xyz), radius, np.float32)))

def _to_dev(x):
    return x.to(device) if torch.is_tensor(x) and x.device != device else x

# ───────────────────────────────────────────────────────────────────────────────
# Initialisation
# ───────────────────────────────────────────────────────────────────────────────
async def initialise_models_and_params():
    global i2p_model, l2w_model, slam_params, is_slam_system_initialized, rerun_connected
    if is_slam_system_initialized or not SLAM3R_ENGINE_AVAILABLE:
        return

    # Rerun handshake
    if os.getenv("RERUN_ENABLED", "true") == "true":
        rr.init("SLAM3R_Processor", spawn=False)
        for host in filter(None,[os.getenv("RERUN_CONNECT_URL"),
                                 "rerun+http://host.docker.internal:9876/proxy",
                                 "rerun+http://127.0.0.1:9876/proxy"]):
            try:
                rr.connect_grpc(host, flush_timeout_sec=15)
                rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
                rerun_connected = True
                logger.info("Connected to Rerun at %s", host)
                break
            except Exception as e:
                logger.warning("Rerun connect failed at %s: %s", host, e)

    logger.info("Loading SLAM3R models on %s…", device)
    i2p_model = Image2PointsModel.from_pretrained("siyan824/slam3r_i2p").to(device).eval()
    l2w_model = Local2WorldModel.from_pretrained("siyan824/slam3r_l2w").to(device).eval()

    cfg = yaml.safe_load(Path(SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER).read_text()) if Path(SLAM3R_CONFIG_FILE_PATH_IN_CONTAINER).exists() else {}
    rp, ka = cfg.get("recon_pipeline", {}), cfg.get("keyframe_adaptation", {})
    slam_params.update({
        "keyframe_stride":      rp.get("keyframe_stride",       -1),
        "initial_winsize":      rp.get("initial_winsize",        5),
        "win_r":                rp.get("win_r",                 3),
        "conf_thres_i2p": float(os.getenv("SLAM3R_CONF_THRES_I2P", rp.get("conf_thres_i2p", 1.5))),
        "conf_thres_l2w": float(os.getenv("SLAM3R_CONF_THRES_L2W", rp.get("conf_thres_l2w", 12.0))),
        "num_scene_frame":      rp.get("num_scene_frame",       10),
        "norm_input_l2w":       rp.get("norm_input_l2w",    False),
        "buffer_size":          rp.get("buffer_size",          100),
        "buffer_strategy":      rp.get("buffer_strategy", "reservoir"),
        "keyframe_adapt_min":   ka.get("adapt_min",             1),
        "keyframe_adapt_max":   ka.get("adapt_max",             20),
        "keyframe_adapt_stride_step": ka.get("adapt_stride_step",1),
    })
    is_slam_system_initialized = True

# ───────────────────────────────────────────────────────────────────────────────
# Per‑frame processing
# ───────────────────────────────────────────────────────────────────────────────
async def process_image_with_slam3r(img_bgr: np.ndarray, ts_ns: int):
    global current_frame_index, is_slam_initialized_for_session, slam_initialization_buffer, active_kf_stride

    start = time.time()
    tensor = preprocess_image(img_bgr)
    img_rgb_u8 = (tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    if rerun_connected:
        rr.log("camera_lowres/rgb", rr.Image(img_rgb_u8))

    view = {"img": tensor.unsqueeze(0),
            "true_shape": torch.tensor([[TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH]], device=device)}
    record = {"img_tensor": tensor,
              "true_shape": view["true_shape"].squeeze(0),
              "timestamp_ns": ts_ns,
              "raw_pose_matrix": np.eye(4).tolist()}

    temp_world_pts: list = []; keyframe_id_out = None

    # ───────── bootstrap ─────────
    if not is_slam_initialized_for_session:
        slam_initialization_buffer.append(view)
        if len(slam_initialization_buffer) < slam_params["initial_winsize"]:
            processed_frames_history.append(record); current_frame_index += 1
            return None, None, None

        init_views = []
        for v in slam_initialization_buffer:
            _, tok, pos = slam3r_get_img_tokens([{
                "img": v["img"].to(device),
                "true_shape": v["true_shape"].unsqueeze(0).to(device)}], i2p_model)
            init_views.append({"img_tokens": tok[0], "img_pos": pos[0], "true_shape": v["true_shape"]})

        pcs, confs, _ = slam3r_initialize_scene(init_views, i2p_model,
                                                winsize=slam_params["initial_winsize"],
                                                conf_thres=slam_params["conf_thres_i2p"],
                                                return_ref_id=True)
        valid_counts = [(c > slam_params["conf_thres_i2p"]).sum().item() for c in confs]
        if sum(valid_counts) < INIT_QUALITY_MIN_POINTS:
            slam_initialization_buffer.clear()
            processed_frames_history.append(record); current_frame_index += 1
            return None, None, None

        for idx, (pc, conf) in enumerate(zip(pcs, confs)):
            hist_idx = current_frame_index - len(slam_initialization_buffer) + idx
            processed_frames_history[hist_idx] |= {
                "img_tokens":   init_views[idx]["img_tokens"],
                "img_pos":      init_views[idx]["img_pos"],
                "pts3d_world": pc,
                "conf_world": conf,
                "keyframe_id": f"kf_{len(keyframe_indices)}",
            }
            keyframe_indices.append(hist_idx)
            pts_np = pc.cpu().numpy().reshape(-1, 3)
            mask   = conf.cpu().numpy().reshape(-1) > slam_params["conf_thres_i2p"]
            cols   = np.tile(np.array([[0,0,255]], np.uint8), (pts_np.shape[0],1))[mask]  # bootstrap → blue
            world_point_cloud_buffer.extend(list(zip(pts_np[mask], cols)))
        slam_initialization_buffer.clear()
        is_slam_initialized_for_session = True
        logger.info("Bootstrap complete with %d keyframes.", len(keyframe_indices))
        processed_frames_history.append(record); current_frame_index += 1
        return None, None, None

    # ────────────────────────── incremental ────────────────────────────
    _, tok, pos = slam3r_get_img_tokens([{"img": view["img"].to(device),
                                          "true_shape": view["true_shape"].to(device)}], i2p_model)
    record["img_tokens"], record["img_pos"] = tok[0], pos[0]

    for kf_idx in list(keyframe_indices):
        if kf_idx >= len(processed_frames_history): continue
        kf_hist = processed_frames_history[kf_idx]
        if kf_hist.get("img_tokens") is None:
            bi = kf_hist["img_tensor"].unsqueeze(0).to(device)
            bt = kf_hist["true_shape"].unsqueeze(0).to(device)
            _, tok_kf, pos_kf = slam3r_get_img_tokens([{"img": bi, "true_shape": bt}], i2p_model)
            kf_hist["img_tokens"], kf_hist["img_pos"] = tok_kf[0], pos_kf[0]

    ref_kf = processed_frames_history[keyframe_indices[-1]]
    if "img" not in ref_kf: ref_kf["img"] = ref_kf["img_tensor"].unsqueeze(0)
    record["img"] = record["img_tensor"].unsqueeze(0)

    window_pair = [
        {k: ref_kf[k] for k in ("img", "img_tokens", "img_pos", "true_shape")},
        {k: record[k] for k in ("img", "img_tokens", "img_pos", "true_shape")},
    ]
    pred = i2p_inference_batch([window_pair]*INFERENCE_WINDOW_BATCH, i2p_model, ref_id=0)["preds"][0]
    record["pts3d_cam"], record["conf_cam"] = pred["pts3d"], pred["conf"]

    # ---------------- colours -----------------
    rgb_flat = colors_from_image(tensor)

    # Build device-safe reference & source views
    cand_views = []
    for idx in keyframe_indices:
        hv = processed_frames_history[idx]
        if "pts3d_world" not in hv: continue
        cand_views.append({k: _to_dev(hv[k]) for k in
                           ("img_tokens", "img_pos", "true_shape", "pts3d_world")})

    if not cand_views:
        processed_frames_history.append(record); current_frame_index += 1
        return None, None, None

    src_view = {
        "img_tokens": _to_dev(record["img_tokens"]),
        "img_pos":    _to_dev(record["img_pos"]),
        "true_shape": _to_dev(record["true_shape"]),
        "pts3d_cam":  _to_dev(record["pts3d_cam"]),
    }
    ref_views, _ = slam3r_scene_frame_retrieve(
        cand_views, [src_view], i2p_model,
        sel_num=min(slam_params["num_scene_frame"], len(cand_views)))
    l2w_out = l2w_inference(ref_views + [src_view], l2w_model,
                            ref_ids=list(range(len(ref_views))),
                            device=device, normalize=slam_params["norm_input_l2w"])[-1]
    record["pts3d_world"], record["conf_world"] = l2w_out["pts3d_in_other_view"], l2w_out["conf"]

    # -------- pose SVD --------
    P_cam   = record["pts3d_cam"].squeeze(0).cpu().reshape(-1, 3).numpy()
    P_world = record["pts3d_world"].squeeze(0).cpu().reshape(-1, 3).numpy()
    conf_cam_flat   = record["conf_cam"].squeeze().cpu().numpy().reshape(-1)
    conf_world_flat = record["conf_world"].squeeze().cpu().numpy().reshape(-1)
    mask = ((conf_cam_flat > slam_params["conf_thres_i2p"]) &
            (conf_world_flat > slam_params["conf_thres_l2w"]))
    if mask.sum() < 3:
        mask = ((conf_cam_flat > slam_params["conf_thres_i2p"]) &
                (conf_world_flat > 0.5 * slam_params["conf_thres_l2w"]))
    R, t = estimate_rigid_transform_svd(P_cam[mask], P_world[mask]) if mask.sum() >=3 else (np.eye(3), np.zeros((3,1)))
    T = np.eye(4); T[:3,:3], T[:3,3] = R, t.squeeze()
    record["raw_pose_matrix"] = T.tolist()

    # -------------- accumulate points --------------
    mask_world = conf_world_flat > slam_params["conf_thres_l2w"]
    if mask_world.sum() < 3:
        mask_world = conf_world_flat > 0.5 * slam_params["conf_thres_l2w"]
    new_pts  = P_world[mask_world]
    cols_flat = rgb_flat[mask_world]
    pts_col_pairs = list(zip(new_pts.tolist(), cols_flat.tolist()))
    world_point_cloud_buffer.extend(pts_col_pairs)
    temp_world_pts.extend(pts_col_pairs)

    # ---------- push history ----------
    record_index = len(processed_frames_history)
    processed_frames_history.append(record)

    # ---------- keyframe selection ----------
    if current_frame_index % (active_kf_stride or 1) == 0:
        record["keyframe_id"] = f"kf_{len(keyframe_indices)}"
        keyframe_indices.append(record_index)
        keyframe_id_out = record["keyframe_id"]

    # ---------- Rerun logging ----------
    if rerun_connected:
        rr.log("world/camera",
               rr.Transform3D(translation=T[:3,3],
                              rotation=rr.Quaternion(xyzw=matrix_to_quaternion(T[:3,:3]))))
        camera_positions.append(T[:3,3].copy())
        if len(camera_positions) > 1:
            rr.log("world/camera_path",
                   rr.LineStrips3D(np.stack(camera_positions, dtype=np.float32)[None]))
        if temp_world_pts:
            log_points_to_rerun(f"world/incremental/frame_{current_frame_index}",
                                temp_world_pts, radius=0.004)
            log_points_to_rerun("world/points", temp_world_pts)
        if keyframe_id_out:
            rr.log("log/info", rr.TextLog(text=f"KF {keyframe_id_out} added", color=[0,255,0]))

        if camera_intrinsics:
            frustum_path = f"world/camera_frustums/frame_{current_frame_index}"
            rr.log(
                frustum_path,
                rr.Transform3D(
                    translation=T[:3, 3],
                    rotation=rr.Quaternion(xyzw=matrix_to_quaternion(T[:3, :3])),
                ),
            )
            rr.log(
                frustum_path + "/pinhole",
                rr.Pinhole(
                    focal_length=[camera_intrinsics['fx'], camera_intrinsics['fy']],
                    principal_point=[camera_intrinsics['cx'], camera_intrinsics['cy']],
                    resolution=[TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT],
                ),
            )
            rr.log(frustum_path + "/image", rr.Image(img_rgb_u8))    

    # ---------- RabbitMQ payloads ----------
    sampled_pairs = random.sample(temp_world_pts, 50_000) if len(temp_world_pts) > 50_000 else temp_world_pts
    xyz_only = [p for p,_ in sampled_pairs]

    q = matrix_to_quaternion(T[:3, :3])
    pose_msg = {
        "timestamp_ns": ts_ns,
        "processing_timestamp": str(datetime.now().timestamp()),
        "position": dict(zip("xyz", T[:3, 3].astype(float))),
        "orientation": {"x":float(q[0]), "y":float(q[1]), "z":float(q[2]), "w":float(q[3])},
        "raw_pose_matrix": record["raw_pose_matrix"],
    }
    pc_msg  = {"timestamp_ns": ts_ns, "points": xyz_only}
    vis_msg = {"timestamp_ns": ts_ns, "type": "points_update_incremental",
               "vertices": xyz_only, "faces": [], "keyframe_id": keyframe_id_out}

    current_frame_index += 1
    logger.info("Frame %d processed in %.2fs", current_frame_index-1, time.time()-start)
    return pose_msg, pc_msg, vis_msg

# ────────────────────────────────────────────────────────────────────────────────
#  RabbitMQ callbacks
# ────────────────────────────────────────────────────────────────────────────────
async def on_video_frame_message(msg: aio_pika.IncomingMessage, exchanges):
    async with msg.process():
        try:
            ts_ns = int(msg.headers.get("timestamp_ns", "0"))
            img   = cv2.imdecode(np.frombuffer(msg.body, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Image decode failed – skipping frame")
                return
            pose, pc, vis = await process_image_with_slam3r(img, ts_ns)
            if not OUTPUT_TO_RABBITMQ:
                return
            if pose:
                await exchanges[SLAM3R_POSE_EXCHANGE_OUT].publish(
                    aio_pika.Message(json.dumps(pose).encode(), content_type="application/json"),
                    routing_key="")
            if pc:
                await exchanges[SLAM3R_POINTCLOUD_EXCHANGE_OUT].publish(
                    aio_pika.Message(gzip.compress(json.dumps(pc).encode()),
                                     content_type="application/json+gzip"),
                    routing_key="")
            if vis:
                await exchanges[SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT].publish(
                    aio_pika.Message(gzip.compress(json.dumps(vis).encode()),
                                     content_type="application/json+gzip"),
                    routing_key="")
        except Exception as e:
            logger.exception("Frame processing error: %s", e)

async def on_restart_message(msg: aio_pika.IncomingMessage):
    async with msg.process():
        global is_slam_system_initialized, processed_frames_history, keyframe_indices
        global world_point_cloud_buffer, current_frame_index, slam_initialization_buffer
        global is_slam_initialized_for_session, active_kf_stride
        logger.info("Restart requested – resetting session state.")
        processed_frames_history.clear()
        keyframe_indices.clear()
        world_point_cloud_buffer.clear()
        slam_initialization_buffer.clear()
        current_frame_index = 0
        is_slam_initialized_for_session = False
        active_kf_stride = 1

# ────────────────────────────────────────────────────────────────────────────────
#  Main service loop
# ────────────────────────────────────────────────────────────────────────────────
async def main():
    await initialise_models_and_params()
    connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=30, heartbeat=60)
    async with connection:
        ch = await connection.channel()
        await ch.set_qos(prefetch_count=1)

        ex_in_frames   = await ch.declare_exchange(VIDEO_FRAMES_EXCHANGE_IN, aio_pika.ExchangeType.FANOUT, durable=True)
        ex_in_restart  = await ch.declare_exchange(RESTART_EXCHANGE_IN,      aio_pika.ExchangeType.FANOUT, durable=True)
        ex_out_pose    = await ch.declare_exchange(SLAM3R_POSE_EXCHANGE_OUT,        aio_pika.ExchangeType.FANOUT, durable=True)
        ex_out_pc      = await ch.declare_exchange(SLAM3R_POINTCLOUD_EXCHANGE_OUT,  aio_pika.ExchangeType.FANOUT, durable=True)
        ex_out_vis     = await ch.declare_exchange(SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT,
                                                   aio_pika.ExchangeType.FANOUT, durable=True)

        exchanges = {SLAM3R_POSE_EXCHANGE_OUT: ex_out_pose,
                     SLAM3R_POINTCLOUD_EXCHANGE_OUT: ex_out_pc,
                     SLAM3R_RECONSTRUCTION_VIS_EXCHANGE_OUT: ex_out_vis}

        q_frames  = await ch.declare_queue("slam3r_video_frames_queue", durable=True)
        q_restart = await ch.declare_queue("slam3r_restart_queue",      durable=True)

        await q_frames.bind(ex_in_frames)
        await q_restart.bind(ex_in_restart)

        await q_frames.consume(lambda m: on_video_frame_message(m, exchanges))
        await q_restart.consume(on_restart_message)

        logger.info("SLAM3R processor ready; awaiting frames…")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")