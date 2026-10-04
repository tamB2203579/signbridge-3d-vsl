import os
import sys
import argparse
import json
import glob
import numpy as np
import scipy.linalg
import cv2
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

BODY_INDICES = list(range(0, 17))
FACE_INDICES = list(range(23, 91))
HAND_LEFT_INDICES = list(range(91, 112))
HAND_RIGHT_INDICES = list(range(112, 133))
HAND_ALL_INDICES = HAND_LEFT_INDICES + HAND_RIGHT_INDICES


def load_video_frames(video_path, max_frames=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_cap = int(max_frames) if (max_frames is not None and max_frames > 0) else None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if frame_cap is not None and len(frames) >= frame_cap:
            break
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")
    np_frames = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(np_frames)



class ResNetVideoFeatureExtractor:
    """
    Spatio-temporal feature extractor powered by 3D ResNet-18 (R3D-18).
    Extracts pooled spatio-temporal representations across sliding 16-frame video clips.
    """
    def __init__(self, model_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = None
        self._init_resnet(model_path)

    def _init_resnet(self, model_path=None):
        import torchvision.models.video as video_models
        
        local_candidates = [
            model_path,
            os.path.join("models", "r3d_18.pth"),
            os.path.join("models", "r3d_18-b3b33576.pth"),
            os.path.join("models", "resnet3d18.pth")
        ]
        
        loaded = False
        for p in local_candidates:
            if p and os.path.exists(p):
                try:
                    net = video_models.r3d_18(pretrained=False)
                    state_dict = torch.load(p, map_location=self.device)
                    net.load_state_dict(state_dict)
                    net.fc = nn.Identity()
                    net.eval().to(self.device)
                    self.model = net
                    loaded = True
                    break
                except Exception:
                    continue

        if not loaded:
            try:
                weights = video_models.R3D_18_Weights.DEFAULT
                net = video_models.r3d_18(weights=weights)
                net.fc = nn.Identity()
                net.eval().to(self.device)
                self.model = net
            except Exception:
                net = video_models.r3d_18(pretrained=False)
                net.fc = nn.Identity()
                net.eval().to(self.device)
                self.model = net

    def extract_clip_features(self, video_tensor, clip_len=16, stride=8, chunk_size=8):
        T, H, W, C = video_tensor.shape
        if T < clip_len:
            pad = clip_len - T
            video_tensor = torch.cat([video_tensor, video_tensor[-1:].repeat(pad, 1, 1, 1)], dim=0)
            T = clip_len

        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

        eff_stride = stride
        if T > clip_len and (T - clip_len) < stride:
            eff_stride = max(1, (T - clip_len) // 2)

        clip_starts = list(range(0, T - clip_len + 1, eff_stride))
        if not clip_starts:
            clip_starts = [0]
        if len(clip_starts) == 1 and T >= clip_len + 1:
            clip_starts.append(T - clip_len)

        all_feats = []
        for i in range(0, len(clip_starts), chunk_size):
            chunk_starts = clip_starts[i : i + chunk_size]
            clips = []
            for start in chunk_starts:
                clip = video_tensor[start : start + clip_len]
                clip_2d = clip.permute(0, 3, 1, 2).float()
                clip_resized = nn.functional.interpolate(clip_2d, size=(112, 112), mode='bilinear', align_corners=False)
                clips.append(clip_resized.permute(1, 0, 2, 3))

            batch_clips = torch.stack(clips, dim=0).to(self.device)
            batch_clips = (batch_clips - mean) / std

            with torch.no_grad():
                if self.model is not None:
                    chunk_feats = self.model(batch_clips)
                    all_feats.append(chunk_feats.cpu().numpy())
                else:
                    diff = batch_clips[:, :, 1:] - batch_clips[:, :, :-1]
                    spatial_mean = batch_clips.mean(dim=(-2, -1)).view(batch_clips.shape[0], -1)
                    temporal_motion = diff.abs().mean(dim=(-2, -1)).view(batch_clips.shape[0], -1)
                    chunk_feats = torch.cat([spatial_mean, temporal_motion], dim=1)
                    all_feats.append(chunk_feats.cpu().numpy())

        return np.concatenate(all_feats, axis=0)


def align_aspect_ratio(tensor, target_ar):
    T, H, W, C = tensor.shape
    cur_ar = W / H
    if abs(cur_ar - target_ar) < 0.02:
        return tensor
    if cur_ar > target_ar:
        new_w = int(round(H * target_ar))
        start_x = max(0, (W - new_w) // 2)
        return tensor[:, :, start_x : start_x + new_w, :]
    else:
        new_h = int(round(W / target_ar))
        start_y = max(0, (H - new_h) // 2)
        return tensor[:, start_y : start_y + new_h, :, :]


def compute_fvd(real_videos, fake_videos, clip_len=16, stride=8, resnet_extractor=None, device="cuda"):
    """
    Computes Fréchet Video Distance (FVD) using 3D ResNet-18 (R3D-18) features.
    """
    if real_videos is not None and fake_videos is not None:
        real_ar = real_videos.shape[2] / real_videos.shape[1]
        fake_ar = fake_videos.shape[2] / fake_videos.shape[1]
        if abs(real_ar - fake_ar) >= 0.02:
            real_videos = align_aspect_ratio(real_videos, fake_ar)

    extractor = resnet_extractor if resnet_extractor is not None else ResNetVideoFeatureExtractor(device=device)
    feats_real = extractor.extract_clip_features(real_videos, clip_len=clip_len, stride=stride)
    feats_fake = extractor.extract_clip_features(fake_videos, clip_len=clip_len, stride=stride)

    if len(feats_real) < 2 or len(feats_fake) < 2:
        mu_real = np.mean(feats_real, axis=0)
        mu_fake = np.mean(feats_fake, axis=0)
        return float(np.linalg.norm(mu_real - mu_fake)) * 10.0

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    diff = mu_real - mu_fake
    eps = 1e-4
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    # Numerically robust and symmetric calculation for tr((sigma_real * sigma_fake)^0.5)
    try:
        u_real, s_real, _ = np.linalg.svd(sigma_real)
        sqrt_sigma_real = u_real @ np.diag(np.sqrt(np.maximum(s_real, 0.0))) @ u_real.T
        m_mat = sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
        eigvals = np.linalg.eigvalsh(m_mat)
        tr_covmean = float(np.sum(np.sqrt(np.maximum(eigvals, 0.0))))
    except Exception:
        res = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake))
        covmean = res[0] if isinstance(res, tuple) else res
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_real.shape[0]) * 1e-3
            res = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
            covmean = res[0] if isinstance(res, tuple) else res
        tr_covmean = float(np.trace(covmean.real))

    fvd = float(diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * tr_covmean)
    return max(0.0, fvd)


class ViTPoseWholeBodyDetector:
    """
    ViTPose WholeBody 133 Keypoints ONNX detector (Body 17, Face 68, Hands 42).
    Prepares input crops, runs ONNX Runtime on CUDA/GPU, and extracts coordinates.
    """
    def __init__(self, vitpose_onnx_path=None, yolo_onnx_path=None, device="cuda"):
        self.device = device
        self.vitpose_session = None
        self.yolo_session = None

        if not vitpose_onnx_path:
            candidates = [
                os.path.join("models", "vitpose-l-wholebody.onnx"),
                os.path.join("models", "vitpose-b-wholebody.onnx"),
                os.path.join("models", "vitpose_wholebody.onnx"),
                os.path.join("models", "vitpose.onnx")
            ]
            for c in candidates:
                if os.path.exists(c):
                    vitpose_onnx_path = c
                    break

        if not yolo_onnx_path:
            candidates = [
                os.path.join("models", "yolov10m.onnx"),
                os.path.join("models", "yolov8m.onnx"),
                os.path.join("models", "yolo.onnx")
            ]
            for c in candidates:
                if os.path.exists(c):
                    yolo_onnx_path = c
                    break

        self.vitpose_path = vitpose_onnx_path
        self.yolo_path = yolo_onnx_path
        self._init_sessions()

    def _init_sessions(self):
        try:
            import onnxruntime as ort
        except ImportError:
            return

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if (self.device == 'cuda' and torch.cuda.is_available()) else ['CPUExecutionProvider']

        if self.vitpose_path and os.path.exists(self.vitpose_path):
            try:
                self.vitpose_session = ort.InferenceSession(self.vitpose_path, providers=providers)
            except Exception:
                self.vitpose_session = None

        if self.yolo_path and os.path.exists(self.yolo_path):
            try:
                self.yolo_session = ort.InferenceSession(self.yolo_path, providers=providers)
            except Exception:
                self.yolo_session = None

    def is_available(self):
        return self.vitpose_session is not None

    def detect_video_keypoints(self, video_path, max_frames=None):
        if not self.is_available():
            return None, None

        frame_cap = int(max_frames) if (max_frames is not None and max_frames > 0) else None
        cap = cv2.VideoCapture(video_path)
        joints_list = []
        confs_list = []
        frame_idx = 0

        input_name = self.vitpose_session.get_inputs()[0].name
        input_shape = self.vitpose_session.get_inputs()[0].shape
        in_h = input_shape[2] if len(input_shape) >= 4 and isinstance(input_shape[2], int) else 256
        in_w = input_shape[3] if len(input_shape) >= 4 and isinstance(input_shape[3], int) else 192

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_cap is not None and frame_idx >= frame_cap:
                break

            h, w = frame.shape[:2]
            # Letterbox: maintain aspect ratio without stretching
            scale = min(float(in_w) / float(w), float(in_h) / float(h))
            nw, nh = int(round(w * scale)), int(round(h * scale))
            img_resized = cv2.resize(frame, (nw, nh))

            pad_w = (in_w - nw) // 2
            pad_h = (in_h - nh) // 2
            canvas = np.zeros((in_h, in_w, 3), dtype=np.float32)
            canvas[pad_h : pad_h + nh, pad_w : pad_w + nw, :] = (
                img_resized[:, :, ::-1].astype(np.float32) / 255.0
            )

            img_input = canvas.transpose(2, 0, 1)
            img_input = np.expand_dims(img_input, axis=0)

            outputs = self.vitpose_session.run(None, {input_name: img_input})
            heatmaps = outputs[0][0]

            frame_kps = []
            frame_conf = []
            hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]

            for j in range(heatmaps.shape[0]):
                hm = heatmaps[j]
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                py, px = int(max_idx[0]), int(max_idx[1])
                conf = float(hm[py, px])

                # Sub-pixel parabolic refinement
                dx = 0.0
                if 1 <= px < hm_w - 1:
                    denom_x = 2.0 * hm[py, px] - hm[py, px - 1] - hm[py, px + 1]
                    if abs(denom_x) > 1e-5:
                        dx = float(np.clip(0.5 * (hm[py, px + 1] - hm[py, px - 1]) / denom_x, -0.5, 0.5))

                dy = 0.0
                if 1 <= py < hm_h - 1:
                    denom_y = 2.0 * hm[py, px] - hm[py - 1, px] - hm[py + 1, px]
                    if abs(denom_y) > 1e-5:
                        dy = float(np.clip(0.5 * (hm[py + 1, px] - hm[py - 1, px]) / denom_y, -0.5, 0.5))

                # Map cell center to canvas coordinates
                canvas_x = (float(px) + dx + 0.5) * (float(in_w) / float(hm_w))
                canvas_y = (float(py) + dy + 0.5) * (float(in_h) / float(hm_h))

                # Un-pad canvas coordinates back to original frame
                orig_x = (canvas_x - pad_w) / scale
                orig_y = (canvas_y - pad_h) / scale

                orig_x = float(np.clip(orig_x, 0.0, float(w - 1)))
                orig_y = float(np.clip(orig_y, 0.0, float(h - 1)))

                frame_kps.append([orig_x, orig_y])
                frame_conf.append(conf)

            joints_list.append(frame_kps)
            confs_list.append(frame_conf)
            frame_idx += 1

        cap.release()
        return np.array(joints_list, dtype=np.float32), np.array(confs_list, dtype=np.float32)


def parse_vitpose_file(file_path):
    """
    Parses pre-extracted ViTPose WholeBody 133 keypoint files (.json, .npy, .npz).
    """
    if not file_path or not os.path.exists(file_path):
        return None, None

    if file_path.endswith('.npy'):
        arr = np.load(file_path)
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            joints = arr[:, :, :2]
            confs = arr[:, :, 2] if arr.shape[-1] >= 3 else np.ones(arr.shape[:2], dtype=np.float32)
            return joints, confs

    if file_path.endswith('.npz'):
        data = np.load(file_path)
        first_key = list(data.keys())[0]
        arr = data[first_key]
        if arr.ndim == 3 and arr.shape[-1] >= 2:
            joints = arr[:, :, :2]
            confs = arr[:, :, 2] if arr.shape[-1] >= 3 else np.ones(arr.shape[:2], dtype=np.float32)
            return joints, confs

    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            joints_list, confs_list = [], []
            for item in data:
                if isinstance(item, dict):
                    kps = item.get("keypoints") or (item.get("people", [{}])[0].get("pose_keypoints_2d"))
                    if kps:
                        arr = np.array(kps)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 3 if len(arr) % 3 == 0 else 2)
                        joints_list.append(arr[:, :2])
                        confs_list.append(arr[:, 2] if arr.shape[1] >= 3 else np.ones(len(arr)))
                elif isinstance(item, list):
                    arr = np.array(item)
                    joints_list.append(arr[:, :2])
                    confs_list.append(arr[:, 2] if arr.shape[1] >= 3 else np.ones(len(arr)))
            if joints_list:
                min_j = min(len(j) for j in joints_list)
                return np.array([j[:min_j] for j in joints_list], dtype=np.float32), np.array([c[:min_j] for c in confs_list], dtype=np.float32)

    return None, None


def get_procrustes_alignment(Y, X):
    """
    Computes optimal similarity transformation (s, R, t) that aligns X onto Y:
    Y_hat = s * (X @ R) + t
    Minimizing ||Y - Y_hat||_F^2
    """
    if len(Y) < 3 or len(X) < 3:
        return 1.0, np.eye(2, dtype=np.float32), np.zeros((1, 2), dtype=np.float32)

    mu_Y = np.mean(Y, axis=0, keepdims=True)
    mu_X = np.mean(X, axis=0, keepdims=True)
    Y_c = Y - mu_Y
    X_c = X - mu_X

    norm_Y = np.linalg.norm(Y_c)
    norm_X = np.linalg.norm(X_c)
    if norm_X < 1e-6 or norm_Y < 1e-6:
        return 1.0, np.eye(2, dtype=np.float32), (mu_Y - mu_X).astype(np.float32)

    H = (X_c / norm_X).T @ (Y_c / norm_Y)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    # Enforce det(R) = +1 (proper rotation, no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
        s_val = (S[0] - S[1]) if len(S) >= 2 else S[0]
    else:
        s_val = np.sum(S)

    s = float(s_val * (norm_Y / norm_X))
    t_vec = mu_Y - s * (mu_X @ R)
    return s, R, t_vec


def apply_procrustes(X, s, R, t_vec):
    """Applies similarity transformation to coordinates X."""
    return s * (X @ R) + t_vec


def align_procrustes(Y, X):
    """Backward-compatible wrapper returning aligned coordinates X."""
    s, R, t_vec = get_procrustes_alignment(Y, X)
    return apply_procrustes(X, s, R, t_vec)


def is_hand_in_frame(hand_joints, hand_confs=None, conf_thresh=0.35, max_y=None, min_valid=4):
    """
    Determines whether a hand (21 keypoints) is legitimately visible in the video frame
    or represents resting/out-of-frame noise.

    1. Confidence check: Requires at least `min_valid` keypoints with confidence >= conf_thresh.
    2. Boundary check: If max_y is provided, mean y of valid joints must be <= max_y (e.g. 0.95 * H).
       When hands drop to the lap or below the camera frame, ViTPose coordinates cluster at the bottom border.
    """
    if hand_joints is None or len(hand_joints) == 0:
        return False

    if hand_confs is not None:
        valid_mask = hand_confs >= conf_thresh
        if np.sum(valid_mask) < min_valid:
            return False
        valid_joints = hand_joints[valid_mask]
    else:
        valid_joints = hand_joints

    if max_y is not None and len(valid_joints) > 0:
        mean_y = float(np.mean(valid_joints[:, 1]))
        if mean_y > max_y:
            return False

    return True


def match_hands_optimal(Y_L, Y_R, X_L, X_R, val_L, val_R, in_frame_L, in_frame_R):
    """
    Hungarian / Swap-Invariant bipartite matching between Ground Truth and Predicted hands.
    Resolves left-right hand swapping caused by crossed-arm sign gestures or ViTPose identity flips.
    """
    cnt_L = int(np.sum(val_L)) if in_frame_L else 0
    cnt_R = int(np.sum(val_R)) if in_frame_R else 0

    if cnt_L >= 3 and cnt_R >= 3:
        # Both hands active in GT: compare natural vs swapped assignment cost
        d_nat = np.sum(np.linalg.norm(X_L[val_L] - Y_L[val_L], axis=-1)) + np.sum(np.linalg.norm(X_R[val_R] - Y_R[val_R], axis=-1))
        d_swp = np.sum(np.linalg.norm(X_R[val_L] - Y_L[val_L], axis=-1)) + np.sum(np.linalg.norm(X_L[val_R] - Y_R[val_R], axis=-1))
        if d_swp < d_nat:
            return X_R, X_L, True
        return X_L, X_R, False

    elif cnt_L >= 3 and cnt_R < 3:
        # Only left hand active in GT: assign whichever predicted hand is closer
        d_nat = np.mean(np.linalg.norm(X_L[val_L] - Y_L[val_L], axis=-1))
        d_swp = np.mean(np.linalg.norm(X_R[val_L] - Y_L[val_L], axis=-1))
        if d_swp < d_nat:
            return X_R, X_L, True
        return X_L, X_R, False

    elif cnt_R >= 3 and cnt_L < 3:
        # Only right hand active in GT: assign whichever predicted hand is closer
        d_nat = np.mean(np.linalg.norm(X_R[val_R] - Y_R[val_R], axis=-1))
        d_swp = np.mean(np.linalg.norm(X_L[val_R] - Y_R[val_R], axis=-1))
        if d_swp < d_nat:
            return X_R, X_L, True
        return X_L, X_R, False

    return X_L, X_R, False


def compute_wholebody_kinematics(gt_joints, pred_joints, confidences=None, bboxes=None,
                                 alpha_1=0.05, alpha_2=0.10, conf_thresh=0.3, img_size=None):
    """
    Computes PA-MPJPE, PA-PCK@alpha_1, PA-PCK@alpha_2 across subsets (Overall, Pose, Face, Hands, Hands_Shape):
    - Overall: Combined upper-body kinematics (Pose + Face + Active in-frame Hands)
    - Pose: 17 Body keypoints
    - Face: 68 Face landmark keypoints
    - Hands: 42 Hand keypoints (Left & Right) with Hungarian swap-invariant matching and out-of-frame filtering
    - Hands_Shape: Local Procrustes alignment per hand isolating finger articulation/shape from wrist translation
    """
    T, J, _ = gt_joints.shape
    subsets = ["Overall", "Pose", "Face", "Hands", "Hands_Shape"]

    part_errors = {p: [] for p in subsets}
    part_c_a1 = {p: 0 for p in subsets}
    part_c_a2 = {p: 0 for p in subsets}
    part_total = {p: 0 for p in subsets}

    # Bottom boundary threshold for out-of-frame hand filtering (95% of frame height)
    if img_size is not None and len(img_size) >= 2:
        max_y = 0.95 * float(img_size[0])
    else:
        max_y = 0.95 * float(np.max(gt_joints[:, :, 1])) if gt_joints.size > 0 else 465.0

    for t in range(T):
        Y_t = gt_joints[t]
        X_t = pred_joints[t]

        if confidences is not None:
            valid_mask = confidences[t] >= conf_thresh
            if np.sum(valid_mask) < 3:
                valid_mask = np.ones(J, dtype=bool)
        else:
            valid_mask = np.ones(J, dtype=bool)

        # 1. Person-level reference scale for PCK thresholding
        scale = None
        if bboxes is not None and t < len(bboxes) and bboxes[t] is not None:
            bbox = bboxes[t]
            if len(bbox) >= 4:
                scale = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
        if scale is None or scale <= 1e-4:
            Y_val = Y_t[valid_mask]
            scale = float(np.max(np.ptp(Y_val, axis=0))) if len(Y_val) > 1 else 100.0
        if scale <= 1e-4:
            scale = 100.0

        thresh_1 = alpha_1 * scale
        thresh_2 = alpha_2 * scale

        # 2. Global Procrustes alignment (aligns torso, head and camera perspective based on non-hand joints)
        # Using torso/head reference prevents hand gestures or hand swaps from skewing the whole body frame.
        ref_mask = np.ones(J, dtype=bool)
        ref_mask[HAND_ALL_INDICES] = False
        align_mask = ref_mask & valid_mask
        if np.sum(align_mask) < 3:
            align_mask = valid_mask

        s, R, t_vec = get_procrustes_alignment(Y_t[align_mask], X_t[align_mask])
        X_aligned_t = apply_procrustes(X_t, s, R, t_vec)

        # 3. Evaluate Pose and Face
        for part, indices in [("Pose", BODY_INDICES), ("Face", FACE_INDICES)]:
            part_mask = np.zeros(J, dtype=bool)
            part_mask[[i for i in indices if i < J]] = True
            eval_mask = part_mask & valid_mask

            if not np.any(eval_mask):
                continue

            y_sub = Y_t[eval_mask]
            x_sub = X_aligned_t[eval_mask]

            diff = np.linalg.norm(x_sub - y_sub, axis=-1)
            part_errors[part].extend(diff.tolist())
            part_c_a1[part] += int(np.sum(diff <= thresh_1))
            part_c_a2[part] += int(np.sum(diff <= thresh_2))
            part_total[part] += len(diff)

            # Add to Overall
            part_errors["Overall"].extend(diff.tolist())
            part_c_a1["Overall"] += int(np.sum(diff <= thresh_1))
            part_c_a2["Overall"] += int(np.sum(diff <= thresh_2))
            part_total["Overall"] += len(diff)

        # 4. Hands evaluation with Hungarian Swap Matching & Out-of-Frame Filtering
        Y_L = Y_t[HAND_LEFT_INDICES]
        Y_R = Y_t[HAND_RIGHT_INDICES]
        X_L = X_aligned_t[HAND_LEFT_INDICES]
        X_R = X_aligned_t[HAND_RIGHT_INDICES]

        c_L = confidences[t, HAND_LEFT_INDICES] if confidences is not None else np.ones(len(HAND_LEFT_INDICES), dtype=np.float32)
        c_R = confidences[t, HAND_RIGHT_INDICES] if confidences is not None else np.ones(len(HAND_RIGHT_INDICES), dtype=np.float32)

        in_f_L = is_hand_in_frame(Y_L, c_L, conf_thresh=conf_thresh, max_y=max_y)
        in_f_R = is_hand_in_frame(Y_R, c_R, conf_thresh=conf_thresh, max_y=max_y)

        val_L = c_L >= conf_thresh
        val_R = c_R >= conf_thresh

        m_X_L, m_X_R, _ = match_hands_optimal(Y_L, Y_R, X_L, X_R, val_L, val_R, in_f_L, in_f_R)

        for is_in, y_h, x_h, val_h in [(in_f_L, Y_L, m_X_L, val_L), (in_f_R, Y_R, m_X_R, val_R)]:
            if is_in and np.sum(val_h) >= 3:
                # 4A. Global Hands Error (under whole-body alignment, swap-corrected)
                diff_h = np.linalg.norm(x_h[val_h] - y_h[val_h], axis=-1)
                part_errors["Hands"].extend(diff_h.tolist())
                part_c_a1["Hands"] += int(np.sum(diff_h <= thresh_1))
                part_c_a2["Hands"] += int(np.sum(diff_h <= thresh_2))
                part_total["Hands"] += len(diff_h)

                # Add valid in-frame hands to Overall
                part_errors["Overall"].extend(diff_h.tolist())
                part_c_a1["Overall"] += int(np.sum(diff_h <= thresh_1))
                part_c_a2["Overall"] += int(np.sum(diff_h <= thresh_2))
                part_total["Overall"] += len(diff_h)

                # 4B. Local Hand Articulation Error (pure finger shape / configuration)
                s_loc, R_loc, t_loc = get_procrustes_alignment(y_h[val_h], x_h[val_h])
                x_local = apply_procrustes(x_h[val_h], s_loc, R_loc, t_loc)
                diff_local = np.linalg.norm(x_local - y_h[val_h], axis=-1)

                part_errors["Hands_Shape"].extend(diff_local.tolist())
                part_c_a1["Hands_Shape"] += int(np.sum(diff_local <= thresh_1))
                part_c_a2["Hands_Shape"] += int(np.sum(diff_local <= thresh_2))
                part_total["Hands_Shape"] += len(diff_local)

    results = {}
    for part in subsets:
        errs = part_errors[part]
        tot = part_total[part]
        results[part] = {
            "PA-MPJPE": round(float(np.mean(errs)), 3) if errs else 0.0,
            f"PA-PCK@{alpha_1:.2f}": round(float(part_c_a1[part] / tot * 100.0), 2) if tot > 0 else 0.0,
            f"PA-PCK@{alpha_2:.2f}": round(float(part_c_a2[part] / tot * 100.0), 2) if tot > 0 else 0.0
        }

    return results


def compute_kinematic_metrics(gt_joints, pred_joints, confidences=None, bboxes=None,
                              alpha_1=0.05, alpha_2=0.10, conf_thresh=0.3, joint_indices=None,
                              global_scale=None):
    """
    Backward-compatible function evaluating a single subset of joints.
    """
    T, J, _ = gt_joints.shape
    if joint_indices is None:
        joint_indices = list(range(J))

    all_pa_errors = []
    correct_pa_a1 = 0
    correct_pa_a2 = 0
    total_valid = 0

    for t in range(T):
        Y = gt_joints[t, joint_indices]
        X = pred_joints[t, joint_indices]

        if confidences is not None:
            mask = confidences[t, joint_indices] >= conf_thresh
            if np.sum(mask) < 3:
                mask = np.ones(len(joint_indices), dtype=bool)
        else:
            mask = np.ones(len(joint_indices), dtype=bool)

        Y_m = Y[mask]
        X_m = X[mask]
        if len(Y_m) < 3:
            continue

        scale = global_scale
        if scale is None and bboxes is not None and t < len(bboxes) and bboxes[t] is not None:
            bbox = bboxes[t]
            if len(bbox) >= 4:
                scale = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
        if scale is None or scale <= 1e-4:
            scale = float(np.max(np.ptp(Y_m, axis=0))) if len(Y_m) > 1 else 100.0
        if scale <= 1e-4:
            scale = 100.0

        thresh_1 = alpha_1 * scale
        thresh_2 = alpha_2 * scale

        X_aligned = align_procrustes(Y_m, X_m)
        pa_diff = np.linalg.norm(X_aligned - Y_m, axis=-1)
        all_pa_errors.extend(pa_diff.tolist())

        correct_pa_a1 += int(np.sum(pa_diff <= thresh_1))
        correct_pa_a2 += int(np.sum(pa_diff <= thresh_2))
        total_valid += len(Y_m)

    return {
        "PA_MPJPE": float(np.mean(all_pa_errors)) if all_pa_errors else 0.0,
        f"PA_PCK@{alpha_1:.2f}": float(correct_pa_a1 / total_valid * 100.0) if total_valid > 0 else 0.0,
        f"PA_PCK@{alpha_2:.2f}": float(correct_pa_a2 / total_valid * 100.0) if total_valid > 0 else 0.0,
        "valid_joints": total_valid
    }


def evaluate_pair(source_path, generated_path,
                  resnet_extractor=None, vitpose_detector=None,
                  source_pose_path=None, generated_pose_path=None,
                  alpha_1=0.05, alpha_2=0.10, max_frames=None, device="cuda"):
    eff_max_frames = int(max_frames) if (max_frames is not None and max_frames > 0) else None
    src_t = load_video_frames(source_path, max_frames=eff_max_frames)
    gen_t = load_video_frames(generated_path, max_frames=eff_max_frames)

    min_f = min(len(src_t), len(gen_t))
    src_t = src_t[:min_f]
    gen_t = gen_t[:min_f]

    fvd = compute_fvd(src_t, gen_t, resnet_extractor=resnet_extractor, device=device)
    result = {
        "source": source_path,
        "generated": generated_path,
        "frames": min_f,
        "FVD": round(fvd, 2),
    }

    gt_kps, gt_c = None, None
    pr_kps, pr_c = None, None

    if source_pose_path and os.path.exists(source_pose_path):
        gt_kps, gt_c = parse_vitpose_file(source_pose_path)
    if generated_pose_path and os.path.exists(generated_pose_path):
        pr_kps, pr_c = parse_vitpose_file(generated_pose_path)

    if (gt_kps is None or pr_kps is None) and vitpose_detector and vitpose_detector.is_available():
        if gt_kps is None:
            gt_kps, gt_c = vitpose_detector.detect_video_keypoints(source_path, max_frames=min_f)
        if pr_kps is None:
            pr_kps, pr_c = vitpose_detector.detect_video_keypoints(generated_path, max_frames=min_f)

    if gt_kps is not None and pr_kps is not None:
        T = min(min_f, len(gt_kps), len(pr_kps))
        gt_kps = gt_kps[:T]
        pr_kps = pr_kps[:T]
        gt_c = gt_c[:T] if gt_c is not None else None

        result["kinematics_vitpose"] = compute_wholebody_kinematics(
            gt_kps, pr_kps, confidences=gt_c, alpha_1=alpha_1, alpha_2=alpha_2,
            img_size=(src_t.shape[1], src_t.shape[2]) if (src_t is not None and len(src_t.shape) >= 3) else None
        )
    else:
        result["kinematics_vitpose"] = "ViTPose model / pose files not loaded (Place vitpose-l-wholebody.onnx in models/ or pass --vitpose_model)"

    return result


def main():
    parser = argparse.ArgumentParser(description="Standalone Sign Language Video Evaluation with ViTPose WholeBody & 3D ResNet-18")
    parser.add_argument("--source", "-s", type=str, help="Source original video path")
    parser.add_argument("--generated", "-g", type=str, help="Generated video path")
    parser.add_argument("--source_dir", type=str, default=None, help="Directory containing source videos")
    parser.add_argument("--generated_dir", type=str, default=None, help="Directory containing generated videos")
    parser.add_argument("--source_pose", type=str, default=None, help="Pre-extracted source ViTPose keypoints (.json, .npy, .npz)")
    parser.add_argument("--generated_pose", type=str, default=None, help="Pre-extracted generated ViTPose keypoints (.json, .npy, .npz)")
    parser.add_argument("--vitpose_model", type=str, default=None, help="Path to vitpose-l-wholebody.onnx (default: models/vitpose-l-wholebody.onnx)")
    parser.add_argument("--yolo_model", type=str, default=None, help="Path to yolov10m.onnx (default: models/yolov10m.onnx)")
    parser.add_argument("--resnet_model", type=str, default=None, help="Path to r3d_18.pth (default: models/r3d_18.pth)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device (default: cuda)")
    parser.add_argument("--alpha_1", type=float, default=0.05, help="Strict PCK threshold (default: 0.05)")
    parser.add_argument("--alpha_2", type=float, default=0.10, help="Standard PCK threshold (default: 0.10)")
    parser.add_argument("--job_json", type=str, default=None, help="Path to existing job JSON file (e.g. outputs/jobs/job_xxx.json) to re-evaluate directly")
    parser.add_argument("--update_job", action="store_true", help="Update the job JSON file in-place with new evaluation results")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Max frames to evaluate")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        args.max_frames = None

    pairs = []
    job_data = None
    if args.job_json:
        if not os.path.exists(args.job_json):
            print(f"Error: Job file not found: {args.job_json}")
            return
        with open(args.job_json, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        job_dir = os.path.dirname(os.path.abspath(args.job_json))
        job_base = os.path.splitext(os.path.basename(args.job_json))[0]
        possible_subdirs = [
            os.path.join(job_dir, job_base),
            job_dir
        ]

        for item in job_data.get("results", []):
            v_name = item.get("video", "")
            base = os.path.splitext(v_name)[0]
            sp = item.get("source_path")
            gp = item.get("generated_path")

            # Check / resolve source path
            if not sp or not os.path.exists(sp):
                candidates = [
                    os.path.join("data", "kaggle_vsl", "videos", v_name),
                    os.path.join("data", "videos", v_name),
                    os.path.join("data", v_name),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        sp = c
                        break

            # Check / resolve generated path
            if not gp or not os.path.exists(gp):
                for sd in possible_subdirs:
                    cand_gp = os.path.join(sd, f"{base}_gen.mp4")
                    if os.path.exists(cand_gp):
                        gp = cand_gp
                        break
                    cand_gp2 = os.path.join(sd, v_name)
                    if os.path.exists(cand_gp2):
                        gp = cand_gp2
                        break

            if sp and gp and os.path.exists(sp) and os.path.exists(gp):
                pairs.append((sp, gp))
            else:
                print(f"Warning: Skipping {v_name} (source or generated file not found)")

    elif args.source and args.generated:
        pairs.append((args.source, args.generated))
    elif args.source_dir and args.generated_dir:
        gen_files = glob.glob(os.path.join(args.generated_dir, "*.mp4"))
        for gf in gen_files:
            base = os.path.splitext(os.path.basename(gf))[0]
            clean_base = base[:-4] if base.endswith("_gen") else base
            cand_sources = [
                os.path.join(args.source_dir, clean_base + ".mp4"),
                os.path.join(args.source_dir, base + ".mp4"),
            ]
            for sf in cand_sources:
                if os.path.exists(sf):
                    pairs.append((sf, gf))
                    break

    if not pairs:
        print("Error: Please provide --job_json, or --source and --generated, or --source_dir and --generated_dir")
        return

    detector = ViTPoseWholeBodyDetector(
        vitpose_onnx_path=args.vitpose_model,
        yolo_onnx_path=args.yolo_model,
        device=args.device
    )
    resnet_extractor = ResNetVideoFeatureExtractor(
        model_path=args.resnet_model,
        device=args.device
    )

    print("=" * 82)
    print("STANDALONE BENCHMARK: VITPOSE WHOLEBODY (133 J) & 3D RESNET-18 (R3D-18)")
    print("=" * 82)
    print(f"Device Target                : {args.device.upper()}")
    print(f"Temporal Model (FVD)         : 3D ResNet-18 (R3D-18 pooled spatio-temporal features)")
    print(f"Kinematic Model (PA-MPJPE)   : ViTPose WholeBody (133 Keypoints: Body 17, Face 68, Hands 42)")
    print(f"Total video pairs to evaluate: {len(pairs)}")
    print("-" * 82)

    results = []
    for s_path, g_path in pairs:
        print(f"\nEvaluating: {os.path.basename(s_path)} vs {os.path.basename(g_path)}")
        res = evaluate_pair(
            s_path, g_path,
            resnet_extractor=resnet_extractor,
            vitpose_detector=detector,
            source_pose_path=args.source_pose,
            generated_pose_path=args.generated_pose,
            alpha_1=args.alpha_1,
            alpha_2=args.alpha_2,
            max_frames=args.max_frames,
            device=args.device
        )
        results.append(res)
        print(f"  Frames: {res['frames']} | FVD (3D ResNet): {res['FVD']}")
        if isinstance(res.get("kinematics_vitpose"), dict):
            print("  ViTPose WholeBody Kinematics (PA-MPJPE / PA-PCK):")
            for part, vals in res["kinematics_vitpose"].items():
                print(f"    [{part:<11}] PA-MPJPE: {vals['PA-MPJPE']:>7.3f} px | PA-PCK@{args.alpha_1:.2f}: {vals[f'PA-PCK@{args.alpha_1:.2f}']:>6.2f}% | PA-PCK@{args.alpha_2:.2f}: {vals[f'PA-PCK@{args.alpha_2:.2f}']:>6.2f}%")

    # Print formatted benchmark summary table across all evaluated video pairs
    if len(results) > 0:
        print("\n" + "=" * 86)
        print(f"BENCHMARK SUMMARY (N = {len(results)} videos)")
        print("=" * 86)
        fvd_list = [r["FVD"] for r in results if r.get("FVD") is not None]
        if fvd_list:
            print(f"{'FVD (3D ResNet-18)':<22}: Mean: {np.mean(fvd_list):>7.2f} ± {np.std(fvd_list):<6.2f} | Median: {np.median(fvd_list):>7.2f} | Min-Max: [{np.min(fvd_list):.2f}, {np.max(fvd_list):.2f}]")

        kin_res_list = [r["kinematics_vitpose"] for r in results if isinstance(r.get("kinematics_vitpose"), dict)]
        if kin_res_list:
            print("-" * 86)
            print(f"{'Subset':<14} | {'PA-MPJPE (px)':<30} | {'PA-PCK@' + f'{args.alpha_1:.2f} (%)':<16} | {'PA-PCK@' + f'{args.alpha_2:.2f} (%)':<16}")
            print("-" * 86)
            for part in ["Overall", "Hands", "Hands_Shape", "Pose", "Face"]:
                m_list = [k[part]["PA-MPJPE"] for k in kin_res_list if part in k and "PA-MPJPE" in k[part]]
                p1_list = [k[part][f"PA-PCK@{args.alpha_1:.2f}"] for k in kin_res_list if part in k and f"PA-PCK@{args.alpha_1:.2f}" in k[part]]
                p2_list = [k[part][f"PA-PCK@{args.alpha_2:.2f}"] for k in kin_res_list if part in k and f"PA-PCK@{args.alpha_2:.2f}" in k[part]]
                if m_list:
                    m_str = f"{np.mean(m_list):.2f} ± {np.std(m_list):.2f} (med {np.median(m_list):.2f})"
                    p1_str = f"{np.mean(p1_list):.2f}%"
                    p2_str = f"{np.mean(p2_list):.2f}%"
                    print(f"{part:<14} | {m_str:<30} | {p1_str:<16} | {p2_str:<16}")
        print("=" * 86)

    if job_data is not None and args.update_job:
        res_map = {os.path.basename(r["source"]): r for r in results}
        for item in job_data.get("results", []):
            v = item.get("video")
            if v in res_map:
                r = res_map[v]
                item["frames"] = r.get("frames")
                item["FVD"] = r.get("FVD")
                item["kinematics"] = r.get("kinematics_vitpose")

        # Recalculate summary in job_data
        summary = {}
        fvd_scores = [r.get("FVD") for r in results if r.get("FVD") is not None]
        if fvd_scores:
            summary["mean_FVD"] = round(float(np.mean(fvd_scores)), 2)

        kin_results = [r.get("kinematics_vitpose") for r in results if isinstance(r.get("kinematics_vitpose"), dict)]
        if kin_results:
            parts = ["Overall", "Hands", "Hands_Shape", "Pose", "Face"]
            for part in parts:
                mpjpe_vals = [k[part]["PA-MPJPE"] for k in kin_results if part in k and "PA-MPJPE" in k[part]]
                pck_a1_vals = [k[part][f"PA-PCK@{args.alpha_1:.2f}"] for k in kin_results if part in k and f"PA-PCK@{args.alpha_1:.2f}" in k[part]]
                pck_a2_vals = [k[part][f"PA-PCK@{args.alpha_2:.2f}"] for k in kin_results if part in k and f"PA-PCK@{args.alpha_2:.2f}" in k[part]]
                if mpjpe_vals:
                    summary[f"mean_{part.lower()}_PA_MPJPE"] = round(float(np.mean(mpjpe_vals)), 3)
                if pck_a1_vals:
                    summary[f"mean_{part.lower()}_PA_PCK@{args.alpha_1:.2f}"] = round(float(np.mean(pck_a1_vals)), 2)
                if pck_a2_vals:
                    summary[f"mean_{part.lower()}_PA_PCK@{args.alpha_2:.2f}"] = round(float(np.mean(pck_a2_vals)), 2)
        job_data["summary"] = summary

        with open(args.job_json, "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Done] Updated job file and summary in-place: {args.job_json}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] All evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
