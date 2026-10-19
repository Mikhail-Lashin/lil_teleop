import json
import contextlib
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import default_collate
from scipy.spatial.transform import Rotation as R
import tyro
import warnings

warnings.filterwarnings("ignore")

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset

BATCH_SIZE = 32


def read_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def run_hamer_on_clip(model, model_cfg, frames, device, fnull):
    all_poses = []
    all_shapes = []

    for i in range(0, len(frames), BATCH_SIZE):
        batch_frames = frames[i : i + BATCH_SIZE]
        batch_samples = []

        for f in batch_frames:
            h, w, _ = f.shape
            bboxes = np.array([[0, 0, w, h]])
            is_right = np.array([1])
            # suppress HaMeR downsampling_factor prints
            with contextlib.redirect_stdout(fnull):
                ds = ViTDetDataset(model_cfg, f, bboxes, is_right, rescale_factor=1.0)
            batch_samples.append(ds[0])

        # robust collation handling ints, arrays, and tensors
        batch = default_collate(batch_samples)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(batch)

        # hand_pose (B, 15, 3, 3) -> axis-angle (B, 45)
        hand_pose_rotm = out["pred_mano_params"]["hand_pose"].cpu().float().numpy()
        b_sz = hand_pose_rotm.shape[0]
        rotvecs = R.from_matrix(hand_pose_rotm.reshape(-1, 3, 3)).as_rotvec().reshape(b_sz, 45)

        # right_pose: 3 zeros for wrist + 45 joints -> (B, 48)
        right_pose = np.zeros((b_sz, 48), dtype=np.float32)
        right_pose[:, 3:] = rotvecs
        all_poses.append(right_pose)

        # right_shape: betas -> (B, 10)
        betas = out["pred_mano_params"]["betas"].cpu().float().numpy().astype(np.float32)
        all_shapes.append(betas)

    return np.concatenate(all_poses, axis=0), np.concatenate(all_shapes, axis=0)


def main(
    output_dir: Path,
    input_dir: Path = Path("./recorded_clips"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    output_npz = output_dir / "train_mano.npz"
    output_types_json = output_dir / "clip_types.json"

    # load existing master npz if present
    dataset_dict = {}
    if output_npz.exists():
        print(f">>> Found existing {output_npz}, loading to append new clips...")
        loaded = np.load(output_npz, allow_pickle=True)
        dataset_dict = {k: loaded[k] for k in loaded.files}

    # load existing clip types mapping if present
    clip_types_dict = {}
    if output_types_json.exists():
        with open(output_types_json, "r", encoding="utf-8") as f:
            clip_types_dict = json.load(f)

    # determine starting clip ID
    existing_ids = [int(k) for k in dataset_dict.keys() if k.isdigit()]
    start_id = max(existing_ids) + 1 if existing_ids else 1
    print(f">>> Next clip ID will start from: {start_id:04d}")

    video_files = sorted(list(input_dir.glob("*/*.mp4")))
    if not video_files:
        print(f">>> No .mp4 files found in {input_dir}")
        return

    print(f">>> Found {len(video_files)} clips in {input_dir}")
    print(f">>> Loading HaMeR on {device}...")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    fnull = open(os.devnull, "w")
    try:
        for idx, vid_path in enumerate(video_files):
            clip_id = start_id + idx
            clip_key = str(clip_id)
            clip_type = vid_path.parent.name

            frames = read_video_frames(vid_path)
            if len(frames) == 0:
                print(f">>> Skip empty video: {vid_path}")
                continue

            right_pose, right_shape = run_hamer_on_clip(model, model_cfg, frames, device, fnull)

            # save individual clip npz without overwriting existing
            clip_npz_path = clips_dir / f"clip_{clip_id:04d}.npz"
            np.savez_compressed(
                clip_npz_path,
                right_pose=right_pose,
                right_shape=right_shape,
                clip_type=clip_type,
            )

            # add to master dataset dictionary
            clip_item = {
                "right_pose": right_pose,
                "right_shape": right_shape,
            }
            dataset_dict[clip_key] = np.array(clip_item, dtype=object)
            clip_types_dict[clip_key] = clip_type

            print(f">>> [{idx + 1:04d}/{len(video_files):04d}] Saved clip_{clip_id:04d}.npz ('{clip_type}', {len(frames)} frames)")

    finally:
        fnull.close()

    # save/update master npz
    np.savez_compressed(output_npz, **dataset_dict)
    print(f"\n>>> Updated master dataset: {output_npz} (Total: {len(dataset_dict)} clips)")

    # save/update types json
    with open(output_types_json, "w", encoding="utf-8") as f:
        json.dump(clip_types_dict, f, indent=2, ensure_ascii=False)
    print(f">>> Updated clip types mapping: {output_types_json}")


if __name__ == "__main__":
    tyro.cli(main)