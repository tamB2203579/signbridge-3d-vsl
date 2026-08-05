"""
ComfyUI-WanAnimatePoseEval — node implementations.

PoseEvaluationMetrics
    Re-runs YOLOv10 + ViTPose (via the WanAnimatePreprocess wrappers) on the
    generated frames, aligns them with the ground-truth pose_data, and computes
    MPJPE + PCK@alpha for Body & Arms, Face Landmarks, Hand Articulations and
    Overall Mean.

SavePoseMetrics
    Writes the metrics to <prefix>.json and <prefix>.csv in the ComfyUI output
    folder.

Structures (verified against ComfyUI-WanAnimatePreprocess):
  - POSEMODEL is a dict {"vitpose": ViTPose, "yolo": Yolo}
    * Yolo is callable as (img_640_transposed[None], shape) -> [ [{ 'bbox': [x1,y1,x2,y2,conf] }] ]
    * ViTPose is callable as (img_norm[None], center[None], scale[None]) -> (N, 133, 3)
  - POSEDATA is a dict with:
    * "pose_metas"            -> list of AAPoseMeta objects (kps in pixels)
    * "pose_metas_original"   -> list of dicts with normalized keypoints
                                  (keypoints_body / _left_hand / _right_hand / _face)
"""

import csv
import json
import os
import time
import traceback

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None


# Debug logging
def _output_dir():
    out = os.environ.get("COMFYUI_OUTPUT_DIR")
    if not out:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "..", "..", "output")
    os.makedirs(out, exist_ok=True)
    return out


def _debug_log(text):
    try:
        path = os.path.join(_output_dir(), "pose_eval_debug.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _bbox_from_detector(bbox, input_resolution=(224, 224), rescale=1.25):
    """
    Exact copy of bbox_from_detector() from
    ComfyUI-WanAnimatePreprocess.pose_utils.pose2d_utils.
    Expected bbox format is [min_x, min_y, max_x, max_y].
    """
    CROP_IMG_HEIGHT, CROP_IMG_WIDTH = input_resolution
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)

    scale = np.array([bbox_size / CROP_ASPECT_RATIO, bbox_size]) / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale


def _get_transform(center, scale, res, rot=0):
    """
    Exact copy of get_transform() from
    ComfyUI-WanAnimatePreprocess.pose_utils.pose2d_utils.
    res: (height, width), (rows, cols)
    """
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def _transform(pt, center, scale, res, invert=0, rot=0):
    """
    Exact copy of transform() from
    ComfyUI-WanAnimatePreprocess.pose_utils.pose2d_utils.
    """
    t = _get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1


def _crop(img, center, scale, res):
    """
    Exact copy of crop() from
    ComfyUI-WanAnimatePreprocess.pose_utils.pose2d_utils.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(_transform([1, 1], center, max(scale), res, invert=1)) - 1
    # Bottom right point
    br = np.array(_transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)
    return new_img, new_shape, (old_x, old_y), (new_x, new_y)


# Ground-truth extraction from POSEDATA
def _gt_group(m, attr, dict_key, conf_attr=None):
    """
    Return (J, 3) NORMALIZED (0..1) keypoints + confidence for a group from a
    single meta, plus the meta (width, height). Handles dict metas and
    AAPoseMeta objects. Column 2 = confidence (0..1).
    """
    w = h = 0
    kps = None
    conf = None
    if isinstance(m, dict):
        w = m.get("width", 0) or 0
        h = m.get("height", 0) or 0
        kps = m.get(dict_key)
        # dict metas already carry confidence in column 2
    else:
        w = getattr(m, "width", 0) or 0
        h = getattr(m, "height", 0) or 0
        kps = getattr(m, attr, None)
        # AAPoseMeta stores confidence separately (e.g. kps_body_p)
        if conf_attr is not None:
            conf = getattr(m, conf_attr, None)
    if kps is None:
        return None, (w, h)
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim < 2 or kps.shape[-1] < 2:
        return None, (w, h)
    arr = kps[:, :2].copy()
    # Normalize to 0..1 using the meta's image size.
    if w and h:
        # If the stored points already look normalized (<= 1.01) keep them.
        if arr.max() > 1.01:
            arr = arr / (w, h)
    # Confidence column
    if kps.shape[1] >= 3:
        conf = kps[:, 2]
    if conf is None:
        conf = np.ones(arr.shape[0], dtype=np.float32)
    else:
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if conf.shape[0] != arr.shape[0]:
            conf = np.ones(arr.shape[0], dtype=np.float32)
    out = np.concatenate([arr, conf[:, None]], axis=1)
    return out, (w, h)


def _extract_gt(pose_data):
    """
    Extract ground-truth keypoint arrays in NORMALIZED (0..1) coordinates.
    Returns (groups_dict, gt_size) where gt_size = (width, height) of the
    ground-truth source space. Prefers pose_metas_original.
    """
    result = {"body": None, "left_hand": None, "right_hand": None, "face": None}
    gt_size = (0, 0)

    if pose_data is None:
        return result, gt_size

    metas = None
    if isinstance(pose_data, dict):
        metas = pose_data.get("pose_metas_original") or pose_data.get("pose_metas")
    elif hasattr(pose_data, "pose_metas_original"):
        metas = pose_data.pose_metas_original or pose_data.pose_metas
    elif hasattr(pose_data, "pose_metas"):
        metas = pose_data.pose_metas

    if not metas:
        return result, gt_size

    # Record the source image size from the first meta
    m0 = metas[0]
    if isinstance(m0, dict):
        gt_size = (m0.get("width", 0) or 0, m0.get("height", 0) or 0)
    else:
        gt_size = (getattr(m0, "width", 0) or 0, getattr(m0, "height", 0) or 0)

    def _stack(getter):
        arrs = []
        for m in metas:
            v, _ = getter(m)
            if v is not None:
                arrs.append(v)
        if not arrs:
            return None
        return np.stack(arrs, axis=0)

    result["body"] = _stack(lambda m: _gt_group(m, "kps_body", "keypoints_body", "kps_body_p"))
    result["left_hand"] = _stack(lambda m: _gt_group(m, "kps_lhand", "keypoints_left_hand", "kps_lhand_p"))
    result["right_hand"] = _stack(lambda m: _gt_group(m, "kps_rhand", "keypoints_right_hand", "kps_rhand_p"))
    result["face"] = _stack(lambda m: _gt_group(m, "kps_face", "keypoints_face", "kps_face_p"))

    return result, gt_size


# Re-run pose detection on generated frames (exact parity with the pack)
def _run_pose_detection(model, images, bbox_overrides=None):
    """
    Re-run YOLO + ViTPose on generated frames using the pack's own wrappers.
    Returns (keypoints_list, bbox_list):
      keypoints_list[i] -> (133, 3) x,y,conf pixels or None
      bbox_list[i]      -> [x1,y1,x2,y2,conf] or None

    bbox_overrides: optional list (len == B) of [x1,y1,x2,y2] boxes in the
    generated-frame pixel space. When a valid override exists for a frame it is
    used for the ViTPose crop instead of a fresh YOLO detection, which makes the
    evaluation robust against detector misses on stylized / blurry frames.
    """
    if not isinstance(model, dict):
        _debug_log(f"[error] model is not a dict: {type(model)}")
        return None, None

    yolo = model.get("yolo")
    vitpose = model.get("vitpose")
    _debug_log(f"[debug] yolo={'found' if yolo else 'MISSING'}, vitpose={'found' if vitpose else 'MISSING'}")

    if cv2 is None:
        _debug_log("[error] cv2 not available")
        return None, None

    imgs = images
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.cpu().numpy()
    imgs = np.asarray(imgs)
    if imgs.ndim == 3:
        imgs = imgs[None, ...]
    if imgs.ndim != 4:
        _debug_log(f"[error] unexpected image shape {imgs.shape}")
        return None, None
    B, H, W, C = imgs.shape
    shape = np.array([[H, W]])
    _debug_log(f"[debug] generated images shape: {imgs.shape}")

    IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
    IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
    input_resolution = (256, 192)
    rescale = 1.25

    try:
        yolo.reinit()
        vitpose.reinit()
    except Exception as e:
        _debug_log(f"[error] model reinit failed: {e}")

    # 1. Determine bboxes: prefer supplied overrides, else YOLO
    bboxes = []
    for i, img in enumerate(imgs):
        override = None
        if bbox_overrides is not None and i < len(bbox_overrides):
            ob = bbox_overrides[i]
            if ob is not None:
                ob = np.asarray(ob, dtype=np.float32).reshape(-1)
                if ob.shape[0] >= 4 and (ob[2] - ob[0]) >= 10 and (ob[3] - ob[1]) >= 10:
                    override = ob
        if override is not None:
            bboxes.append(override)
            continue
        try:
            det = yolo(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape,
            )
            bbox = det[0][0]["bbox"] if det and det[0] else None
        except Exception as e:
            _debug_log(f"[error] yolo detect frame {i}: {e}\n{traceback.format_exc()}")
            bbox = None
        bboxes.append(bbox)

    try:
        yolo.cleanup()
    except Exception:
        pass

    # 2. Extract keypoints
    kp2ds = []
    for i, bbox in enumerate(bboxes):
        img = imgs[i]
        if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
            bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        try:
            center, scale = _bbox_from_detector(bbox, input_resolution, rescale=rescale)
            cropped = _crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
            img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
            keypoints = vitpose(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kps = np.asarray(keypoints[0], dtype=np.float32)  # (133, 3) x,y,conf
            kp2ds.append(kps)
        except Exception as e:
            _debug_log(f"[error] vitpose frame {i}: {e}\n{traceback.format_exc()}")
            kp2ds.append(None)

    try:
        vitpose.cleanup()
    except Exception:
        pass

    return kp2ds, bboxes


# Metrics
def _split_kp2ds_for_aa(kp2ds):
    """
    Exact copy of split_kp2ds_for_aa() from
    ComfyUI-WanAnimatePreprocess.pose_utils.pose2d_utils.
    Input (133, 2) -> (body20, lhand21, rhand21, face70).
    """
    kp2ds_body = (kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] +
                  kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
    kp2ds_lhand = kp2ds[91:112]
    kp2ds_rhand = kp2ds[112:133]
    kp2ds_face = kp2ds[22:91]
    return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy(), kp2ds_face.copy()


def _filter_conf(g, p, threshold):
    """
    Filter joints where BOTH GT and pred confidence >= threshold.
    g / p: (J, 3) normalized x,y,conf.
    Returns (g_xy, p_xy, idx) where idx are the kept joint indices, or
    (None, None, None) when nothing survives.
    """
    if g is None or p is None:
        return None, None, None
    g = np.asarray(g, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    if g.ndim != 2 or p.ndim != 2 or g.shape[0] != p.shape[0] or g.shape[1] < 3 or p.shape[1] < 3:
        return None, None, None
    g_conf = g[:, 2]
    p_conf = p[:, 2]
    mask = (g_conf >= threshold) & (p_conf >= threshold)
    if not np.any(mask):
        return None, None, None
    idx = np.nonzero(mask)[0]
    return g[mask, :2], p[mask, :2], idx


def _pa_mpjpe(g, p):
    """
    Procrustes-aligned MPJPE (PA-MPJPE) in the same normalized units as g / p.
    Fits a similarity transform (scale + rotation + translation) from the
    prediction to the ground truth, then returns the mean joint error. This
    removes global character drift / scale, isolating articulation error.
    """
    g = np.asarray(g, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[0] != p.shape[0]:
        return float(np.mean(np.linalg.norm(p[:, :2] - g[:, :2], axis=1)))
    mu_g = g.mean(axis=0)
    mu_p = p.mean(axis=0)
    g0 = g - mu_g
    p0 = p - mu_p
    H = p0.T @ g0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    var_p = float(np.sum(p0 * p0))
    if var_p <= 1e-12:
        return float(np.mean(np.linalg.norm(p[:, :2] - g[:, :2], axis=1)))
    s = float(np.sum(g0 * (p0 @ R.T)) / var_p)
    aligned = s * (p0 @ R.T) + mu_g
    return float(np.mean(np.linalg.norm(aligned - g, axis=1)))


BODY_JOINT_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear", "left_foot", "right_foot",
]

HAND_JOINT_NAMES = [
    "wrist", "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _compute_metrics(gt_groups, pred_groups, bbox, alpha_1, alpha_2, ref_h, conf_thr):
    """
    Compute MPJPE + PCK (+ PA-MPJPE for body/hands) for a single frame.
    gt_groups / pred_groups: dict with keys body / left_hand / right_hand / face,
    each (J, 3) NORMALIZED (0..1) x,y,conf (same space).
    bbox: person box in NORMALIZED (0..1) coordinates, or None.
    ref_h: reference height (px) used to convert normalized MPJPE to pixels.
    conf_thr: confidence threshold; joints below it are excluded.

    Returns (result, joint_info, pa_vals):
      joint_info: {"body": (idx, dist, norm), "left_hand": (...), ...} per-joint
                  distances (normalized) + kept joint indices + PCK norm.
      pa_vals:    list of per-frame PA-MPJPE (normalized units) for body & hands.
    """
    result = {}
    joint_info = {}
    pa_vals = []
    norm = _pck_norm(bbox, gt_groups, pred_groups)

    def _block(g, p, idx, pa=False):
        dist = np.linalg.norm(g - p, axis=1)
        stats = {
            "MPJPE_px": round(float(np.mean(dist) * ref_h), 4),
            "PCK@0.05": float(np.mean(dist <= alpha_1 * norm) * 100.0),
            "PCK@0.10": float(np.mean(dist <= alpha_2 * norm) * 100.0),
            "n_joints": int(g.shape[0]),
        }
        pa_val = None
        if pa:
            pa_val = _pa_mpjpe(g, p)
            stats["PA-MPJPE_px"] = round(float(pa_val * ref_h), 4)
        return stats, dist, pa_val

    # Body & Arms: 20 joints
    g, p, idx = _filter_conf(gt_groups.get("body"), pred_groups.get("body"), conf_thr)
    if g is not None:
        stats, dist, pa_val = _block(g, p, idx, pa=True)
        result["Body & Arms"] = stats
        joint_info["body"] = (idx, dist, norm)
        if pa_val is not None:
            pa_vals.append(pa_val)

    # Face Landmarks: 70 joints (aggregate only)
    g_face, p_face, idx_face = _filter_conf(gt_groups.get("face"), pred_groups.get("face"), conf_thr)
    if g_face is not None:
        stats, dist, _ = _block(g_face, p_face, idx_face)
        result["Face Landmarks"] = stats
        joint_info["face"] = (idx_face, dist, norm)

    # Hand Articulations: 42 joints (21 left + 21 right)
    g_lh, p_lh, idx_lh = _filter_conf(gt_groups.get("left_hand"), pred_groups.get("left_hand"), conf_thr)
    g_rh, p_rh, idx_rh = _filter_conf(gt_groups.get("right_hand"), pred_groups.get("right_hand"), conf_thr)
    if g_lh is not None and g_rh is not None:
        g_hand = np.concatenate([g_lh, g_rh], axis=0)
        p_hand = np.concatenate([p_lh, p_rh], axis=0)
        dist = np.linalg.norm(g_hand - p_hand, axis=1)
        n_lh = idx_lh.shape[0]
        result["Hand Articulations"] = {
            "MPJPE_px": round(float(np.mean(dist) * ref_h), 4),
            "PCK@0.05": float(np.mean(dist <= alpha_1 * norm) * 100.0),
            "PCK@0.10": float(np.mean(dist <= alpha_2 * norm) * 100.0),
            "PA-MPJPE_px": round(float(_pa_mpjpe(g_hand, p_hand) * ref_h), 4),
            "n_joints": int(g_hand.shape[0]),
        }
        joint_info["left_hand"] = (idx_lh, dist[:n_lh], norm)
        joint_info["right_hand"] = (idx_rh, dist[n_lh:], norm)
        pa_vals.append(_pa_mpjpe(g_hand, p_hand))

    return result, joint_info, pa_vals


def _pck_norm(bbox, gt_groups, pred_groups):
    """
    PCK normalisation: max(bbox H, W) in normalized space.
    Falls back to the extent of the body keypoints (union of GT + pred) when
    no valid bbox is available, so PCK@0.10 is always >= PCK@0.05.
    """
    if bbox is not None:
        bbox = np.asarray(bbox, dtype=np.float32).flatten()
        if bbox.shape[0] >= 4:
            bw = float(bbox[2]) - float(bbox[0])
            bh = float(bbox[3]) - float(bbox[1])
            if bw > 0 and bh > 0:
                return max(bw, bh)
    # Fallback: body keypoint extent (union of GT + pred)
    g = gt_groups.get("body")
    p = pred_groups.get("body")
    if g is not None and p is not None:
        g = np.asarray(g, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
        if g.ndim == 2 and p.ndim == 2 and g.shape[1] >= 2:
            all_pts = np.concatenate([g[:, :2], p[:, :2]], axis=0)
            if all_pts.shape[0] > 0:
                x0, y0 = all_pts.min(axis=0)
                x1, y1 = all_pts.max(axis=0)
                ext = max(x1 - x0, y1 - y0)
                if ext > 0:
                    return float(ext)
    return 1.0


# Nodes
class PoseEvaluationMetrics:
    """Re-extract keypoints from generated frames and compare to ground truth."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("POSEMODEL",),
                "pose_data": ("POSEDATA",),
                "generated_images": ("IMAGE",),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_1": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_2": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frame_offset": ("INT", {"default": 1, "min": 0, "max": 1000}),
                "gt_bboxes": ("BBOX",),
                "use_gt_bbox": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metrics_json",)
    FUNCTION = "evaluate"
    CATEGORY = "WanAnimatePoseEval"

    def evaluate(self, model, pose_data, generated_images, confidence_threshold=0.3,
                 alpha_1=0.05, alpha_2=0.10, frame_offset=1, gt_bboxes=None, use_gt_bbox=True):
        _debug_log("\n===== NEW EVAL RUN =====")
        _debug_log(f"model type: {type(model)}")
        _debug_log(f"pose_data type: {type(pose_data)}")
        _debug_log(f"generated_images type: {type(generated_images)}")
        _debug_log(f"[debug] use_gt_bbox: {use_gt_bbox}, gt_bboxes provided: {gt_bboxes is not None}")

        # Ground truth: (groups_dict, gt_size) in NORMALIZED coords
        gt, gt_size = _extract_gt(pose_data)
        for key in ("body", "left_hand", "right_hand", "face"):
            arr = gt[key]
            _debug_log(f"[debug] gt.{key}: {'shape=' + str(arr.shape) if arr is not None else 'None'}")
        _debug_log(f"[debug] gt_size (w,h): {gt_size}")

        # Generation image size (pixel space of predictions)
        gen_size = (0, 0)
        if isinstance(generated_images, torch.Tensor):
            gen_size = (generated_images.shape[2], generated_images.shape[1])
        else:
            arr = np.asarray(generated_images)
            if arr.ndim == 4:
                gen_size = (arr.shape[2], arr.shape[1])
            elif arr.ndim == 3:
                gen_size = (arr.shape[1], arr.shape[0])
        _debug_log(f"[debug] gen_size (w,h): {gen_size}")

        # Optional: feed the GT boxes straight into the pose detector so crops
        # are always centred on the person (fixes detector misses on limbs).
        bbox_overrides = None
        if use_gt_bbox and gt_bboxes is not None:
            arr = np.asarray(gt_bboxes, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim == 2 and arr.shape[1] >= 4:
                overrides = []
                for row in arr:
                    b = row[:4]
                    if b.max() <= 1.5 and gen_size[0] and gen_size[1]:
                        b = b * np.array([gen_size[0], gen_size[1], gen_size[0], gen_size[1]],
                                         dtype=np.float32)
                    overrides.append(b.astype(np.float32))
                bbox_overrides = overrides
                _debug_log(f"[debug] using {len(bbox_overrides)} GT bbox overrides")

        # Re-run pose detection on generated frames
        pred_kps, pred_boxes = _run_pose_detection(model, generated_images, bbox_overrides)

        gt_body = gt.get("body")
        n_gen = len(pred_kps) if pred_kps else 0
        n_gt = gt_body.shape[0] if gt_body is not None else 0
        n = min(n_gen, max(0, n_gt - frame_offset))
        _debug_log(f"[debug] n_gen={n_gen}, n_gt={n_gt}, n_frames_compared={n}")

        # Reference height (px) to convert normalized MPJPE to pixels
        ref_h = float(gen_size[1]) if gen_size[1] else 1.0

        per_frame = []
        joint_acc = {"body": {}, "left_hand": {}, "right_hand": {}}
        pooled_dist = []
        pooled_norm = []
        pa_vals = []
        for i in range(n):
            # GT group dict: already NORMALIZED (0..1)
            g_groups = {}
            for key in ("body", "left_hand", "right_hand", "face"):
                arr = gt.get(key)
                if arr is not None and i + frame_offset < arr.shape[0]:
                    g_groups[key] = arr[i + frame_offset]

            # Prediction group dict: 133 px -> AA layout -> NORMALIZED (0..1)
            p_groups = {}
            if pred_kps and pred_kps[i] is not None:
                p133 = np.asarray(pred_kps[i], dtype=np.float32)
                if p133.ndim == 2 and p133.shape[0] >= 133:
                    b, lh, rh, fc = _split_kp2ds_for_aa(p133)
                    if gen_size[0] and gen_size[1]:
                        scale = np.array([gen_size[0], gen_size[1], 1.0], dtype=np.float32)
                        b = b / scale
                        lh = lh / scale
                        rh = rh / scale
                        fc = fc / scale
                    p_groups["body"] = b
                    p_groups["left_hand"] = lh
                    p_groups["right_hand"] = rh
                    p_groups["face"] = fc

            # Bbox in NORMALIZED (0..1) coords
            bbox = None
            if gt_bboxes is not None:
                b = np.asarray(gt_bboxes, dtype=np.float32)
                if b.ndim == 2 and b.shape[0] > i:
                    bbox = b[i]
                elif b.ndim == 1 and b.shape[0] >= 4:
                    bbox = b
                if bbox is not None and gen_size[0] and gen_size[1]:
                    bbox = bbox.copy()
                    bbox[0] /= gen_size[0]
                    bbox[1] /= gen_size[1]
                    bbox[2] /= gen_size[0]
                    bbox[3] /= gen_size[1]
            elif pred_boxes and pred_boxes[i] is not None:
                bbox = np.asarray(pred_boxes[i], dtype=np.float32)
                if gen_size[0] and gen_size[1]:
                    bbox = bbox.copy()
                    bbox[0] /= gen_size[0]
                    bbox[1] /= gen_size[1]
                    bbox[2] /= gen_size[0]
                    bbox[3] /= gen_size[1]

            m, jinfo, pa_frame = _compute_metrics(
                g_groups, p_groups, bbox, alpha_1, alpha_2, ref_h, confidence_threshold)
            if m:
                per_frame.append(m)
                pa_vals.extend(pa_frame)
                for key, (idx, dist, norm) in jinfo.items():
                    pooled_dist.extend(dist.tolist())
                    pooled_norm.extend([float(norm)] * int(idx.shape[0]))
                    acc = joint_acc.get(key)
                    if acc is None:
                        continue
                    for j, d in zip(idx.tolist(), dist.tolist()):
                        acc.setdefault(int(j), {"d": [], "n": []})
                        acc[int(j)]["d"].append(float(d))
                        acc[int(j)]["n"].append(float(norm))

        groups = {}
        for gname in ("Body & Arms", "Face Landmarks", "Hand Articulations"):
            vals = [f[gname] for f in per_frame if gname in f]
            if not vals:
                continue
            groups[gname] = {
                "MPJPE_px": round(float(np.mean([v["MPJPE_px"] for v in vals])), 4),
                "PCK@0.05": float(np.mean([v["PCK@0.05"] for v in vals])),
                "PCK@0.10": float(np.mean([v["PCK@0.10"] for v in vals])),
                "n_joints": int(sum(v["n_joints"] for v in vals)),
            }
            if "PA-MPJPE_px" in vals[0]:
                groups[gname]["PA-MPJPE_px"] = round(
                    float(np.mean([v["PA-MPJPE_px"] for v in vals])), 4)

        # Overall = weighted mean across all groups (legacy, kept for compat)
        all_mpjpe = []
        all_pck1 = []
        all_pck2 = []
        for f in per_frame:
            for gname, gdata in f.items():
                all_mpjpe.append(gdata["MPJPE_px"])
                all_pck1.append(gdata["PCK@0.05"])
                all_pck2.append(gdata["PCK@0.10"])
        n_joints = sum(v.get("n_joints", 0) for f in per_frame for v in f.values())

        overall = {
            "MPJPE_px": round(float(np.mean(all_mpjpe)), 4) if all_mpjpe else 0.0,
            "PCK@0.05": float(np.mean(all_pck1)) if all_pck1 else 0.0,
            "PCK@0.10": float(np.mean(all_pck2)) if all_pck2 else 0.0,
            "n_joints": int(n_joints),
        }
        if pa_vals:
            overall["PA-MPJPE_px"] = round(float(np.mean(pa_vals) * ref_h), 4)

        # Joint-count weighted overall (the honest single number)
        overall_w = None
        if pooled_dist:
            pd = np.asarray(pooled_dist, dtype=np.float32)
            pn = np.asarray(pooled_norm, dtype=np.float32)
            overall_w = {
                "MPJPE_px": round(float(np.mean(pd) * ref_h), 4),
                "PCK@0.05": float(np.mean(pd <= alpha_1 * pn) * 100.0),
                "PCK@0.10": float(np.mean(pd <= alpha_2 * pn) * 100.0),
                "n_joints": int(pd.shape[0]),
            }

        # Per-joint breakdown for the groups that matter
        joint_breakdown = {}
        for key, names in (("body", BODY_JOINT_NAMES),
                           ("left_hand", HAND_JOINT_NAMES),
                           ("right_hand", HAND_JOINT_NAMES)):
            rows = []
            for j in sorted(joint_acc.get(key, {})):
                rec = joint_acc[key][j]
                ds = np.asarray(rec["d"], dtype=np.float32)
                ns = np.asarray(rec["n"], dtype=np.float32)
                rows.append({
                    "joint": names[j] if j < len(names) else f"joint_{j}",
                    "MPJPE_px": round(float(np.mean(ds) * ref_h), 4),
                    "PCK@0.05": float(np.mean(ds <= alpha_1 * ns) * 100.0),
                    "PCK@0.10": float(np.mean(ds <= alpha_2 * ns) * 100.0),
                    "n_frames": int(ds.shape[0]),
                })
            joint_breakdown[key] = rows

        result = {
            "config": {
                "confidence_threshold": confidence_threshold,
                "alpha_1": alpha_1,
                "alpha_2": alpha_2,
                "frame_offset": frame_offset,
                "use_gt_bbox": bool(use_gt_bbox),
                "n_frames_compared": n,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "overall": overall,
            "overall_joint_weighted": overall_w,
            "groups": groups,
            "joint_breakdown": joint_breakdown,
        }

        _debug_log(f"[debug] result: {json.dumps(result, ensure_ascii=False)}")
        return (json.dumps(result, ensure_ascii=False, indent=2),)


class SavePoseMetrics:
    """Write the metrics to JSON and CSV in the ComfyUI output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metric_data": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "WanAnimate_pose_eval"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "WanAnimatePoseEval"
    OUTPUT_NODE = True

    def save(self, metric_data, filename_prefix="WanAnimate_pose_eval"):
        if isinstance(metric_data, str):
            try:
                data = json.loads(metric_data)
            except Exception:
                data = {"raw": metric_data}
        else:
            data = metric_data

        output_dir = _output_dir()
        json_path = os.path.join(output_dir, filename_prefix + ".json")
        csv_path = os.path.join(output_dir, filename_prefix + ".csv")

        # JSON: accumulate every run as one record in a list (append, never overwrite)
        records = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    records = existing
                else:
                    records = [existing]
            except Exception:
                records = []
        records.append(data)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        rows = []
        overall = data.get("overall", {})
        overall_w = data.get("overall_joint_weighted", {})
        groups = data.get("groups", {})
        for metric in ("MPJPE_px", "PCK@0.05", "PCK@0.10", "PA-MPJPE_px"):
            if metric in overall:
                rows.append(["Overall Mean", metric, overall.get(metric, "")])
            if metric in overall_w:
                rows.append(["Overall Joint-Weighted", metric, overall_w.get(metric, "")])
        for gname, gdata in groups.items():
            for metric in ("MPJPE_px", "PCK@0.05", "PCK@0.10", "PA-MPJPE_px"):
                if metric in gdata:
                    rows.append([gname, metric, gdata.get(metric, "")])
        for key, joints in (data.get("joint_breakdown", {}) or {}).items():
            for j in joints:
                for metric in ("MPJPE_px", "PCK@0.05", "PCK@0.10"):
                    rows.append([f"{key}:{j.get('joint', '')}", metric, j.get(metric, "")])

        # CSV: append rows; write the header only when the file is new
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Group", "Metric", "Value"])
            writer.writerows(rows)

        print(f"[SavePoseMetrics] Appended run #{len(records)} to {json_path}")
        print(f"[SavePoseMetrics] Appended rows to {csv_path}")
        return {}