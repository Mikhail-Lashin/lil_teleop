from pathlib import Path
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.depthcor.sampler import DepthSampler
from src.utils.frame_processing import crop_to_square


def main(
    bag_file: str,
    frame_idx: int = 0,
    num_samples: int = 300,
):
    # load realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}

    # get frames
    for _ in range(frame_idx):
        pipeline.wait_for_frames()
    frames = align.process(pipeline.wait_for_frames())
    pipeline.stop()

    color = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_RGB2BGR)
    depth = np.asanyarray(frames.get_depth_frame().get_data())
    color_crop, depth_crop, offset_x = crop_to_square(color, depth)

    # sampling
    sampler = DepthSampler(num_samples=num_samples)
    pts_3d = sampler.sample(
        depth_crop, intrinsics, offset_x=offset_x, depth_scale=depth_scale
    )

    # draw hand mask
    depth_m = depth_crop.astype(np.float32) * depth_scale
    valid = (depth_m > sampler.min_depth) & (depth_m < sampler.hard_max_depth)
    if np.any(valid):
        hand_mask = valid
        color_crop[hand_mask] = (
            color_crop[hand_mask] * 0.6 + np.array([255, 0, 0]) * 0.4
        ).astype(np.uint8)

    # draw points
    if len(pts_3d) > 0:
        u = np.round(
            pts_3d[:, 0] * intrinsics["fx"] / pts_3d[:, 2]
            + intrinsics["cx"]
            - offset_x
        ).astype(int)
        v = np.round(
            pts_3d[:, 1] * intrinsics["fy"] / pts_3d[:, 2] + intrinsics["cy"]
        ).astype(int)
        
        # gradient colors for points
        zs = pts_3d[:, 2]
        z_min = np.min(zs)
        z_max = np.max(zs)
        t = np.clip((zs - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
        c_near = np.array([80, 80, 255], dtype=np.float32)
        c_far = np.array([255, 0, 0], dtype=np.float32)
        colors = ((1.0 - t[:, None]) * c_near + t[:, None] * c_far).astype(np.uint8)

        r = 1
        for x, y, c in zip(u, v, colors):
            color_bgr = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(color_crop, (x - r, y - r), (x + r, y + r), color_bgr, -1)

    cv2.imshow("Sampler Result", color_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)