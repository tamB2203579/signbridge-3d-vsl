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

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
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

        clip_starts = list(range(0, T - clip_len + 1, stride))
        if not clip_starts:
            clip_starts = [0]

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
        return float(np.linalg.norm(mu_real - mu_fake)) * 100.0

    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_fake = np.mean(feats_fake, axis=0)
    sigma_fake = np.cov(feats_fake, rowvar=False)

    diff = mu_real - mu_fake
    eps = 1e-4
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    res = scipy.linalg.sqrtm(sigma_real.dot(sigma_fake))
    covmean = res[0] if isinstance(res, tuple) else res
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-3
        res = scipy.linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        covmean = res[0] if isinstance(res, tuple) else res

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = float(diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean))
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
            if max_frames and frame_idx >= max_frames:
                break

            h, w = frame.shape[:2]
            img_resized = cv2.resize(frame, (in_w, in_h))
            img_input = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            outputs = self.vitpose_session.run(None, {input_name: img_input})
            heatmaps = outputs[0][0]

            frame_kps = []
            frame_conf = []
            scale_x = w / float(heatmaps.shape[2])
            scale_y = h / float(heatmaps.shape[1])

            for j in range(heatmaps.shape[0]):
                hm = heatmaps[j]
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                conf = float(hm[max_idx])
                kx = float(max_idx[1]) * scale_x
                ky = float(max_idx[0]) * scale_y
                frame_kps.append([kx, ky])
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


def align_procrustes(Y, X):
    if len(Y) < 3 or len(X) < 3:
        return X
    mu_Y = np.mean(Y, axis=0, keepdims=True)
    mu_X = np.mean(X, axis=0, keepdims=True)
    Y_c = Y - mu_Y
    X_c = X - mu_X

    norm_Y = np.linalg.norm(Y_c)
    norm_X = np.linalg.norm(X_c)
    if norm_X < 1e-6 or norm_Y < 1e-6:
        return X

    H = (X_c / norm_X).T @ (Y_c / norm_Y)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    s = np.sum(S) * (norm_Y / norm_X)
    t_vec = mu_Y - s * (mu_X @ R)
    return s * (X @ R) + t_vec


def compute_kinematic_metrics(gt_joints, pred_joints, confidences=None, bboxes=None,
                              alpha_1=0.05, alpha_2=0.10, conf_thresh=0.3, joint_indices=None):
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

        scale = None
        if bboxes is not None and t < len(bboxes) and bboxes[t] is not None:
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
    src_t = load_video_frames(source_path, max_frames=max_frames)
    gen_t = load_video_frames(generated_path, max_frames=max_frames)

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
        num_j = gt_kps.shape[1]

        subsets = {
            "Overall": list(range(num_j)),
            "Pose": [i for i in BODY_INDICES if i < num_j],
            "Face": [i for i in FACE_INDICES if i < num_j],
            "Hands": [i for i in HAND_ALL_INDICES if i < num_j]
        }

        kin_res = {}
        for part, indices in subsets.items():
            if len(indices) >= 3:
                res = compute_kinematic_metrics(gt_kps, pr_kps, gt_c, None, alpha_1, alpha_2, joint_indices=indices)
                kin_res[part] = {
                    "PA-MPJPE": round(res["PA_MPJPE"], 3),
                    f"PA-PCK@{alpha_1:.2f}": round(res[f"PA_PCK@{alpha_1:.2f}"], 2),
                    f"PA-PCK@{alpha_2:.2f}": round(res[f"PA_PCK@{alpha_2:.2f}"], 2)
                }
        result["kinematics_vitpose"] = kin_res
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
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Max frames to evaluate")
    parser.add_argument("--output", "-o", type=str, default="evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    pairs = []
    if args.source and args.generated:
        pairs.append((args.source, args.generated))
    elif args.source_dir and args.generated_dir:
        gen_files = glob.glob(os.path.join(args.generated_dir, "*.mp4"))
        for gf in gen_files:
            base = os.path.splitext(os.path.basename(gf))[0]
            sf = os.path.join(args.source_dir, base + ".mp4")
            if os.path.exists(sf):
                pairs.append((sf, gf))

    if not pairs:
        print("Error: Please provide --source and --generated, or --source_dir and --generated_dir")
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
                print(f"    [{part:<8}] PA-MPJPE: {vals['PA-MPJPE']:.3f} px | PA-PCK@{args.alpha_1:.2f}: {vals[f'PA-PCK@{args.alpha_1:.2f}']}% | PA-PCK@{args.alpha_2:.2f}: {vals[f'PA-PCK@{args.alpha_2:.2f}']}%")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] All evaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
