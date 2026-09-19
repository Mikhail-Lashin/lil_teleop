from typing import Dict
import numpy as np

class DepthSampler:

  def __init__(
      self,
      num_samples: int = 300,
      min_depth: float = 0.07,
      hard_max_depth: float = 0.35,
  ):
    self.num_samples = num_samples
    self.min_depth = min_depth
    self.hard_max_depth = hard_max_depth

  def sample(
      self,
      depth_crop: np.ndarray,
      intrinsics: Dict[str, float],
      offset_x: int = 0,
      depth_scale: float = 0.001,
  ) -> np.ndarray:
    # convert depth to m
    if depth_crop.dtype == np.uint16:
      depth_m = depth_crop.astype(np.float32) * depth_scale
    else:
      depth_m = depth_crop

    # depth mask
    mask = (depth_m > self.min_depth) & (depth_m < self.hard_max_depth)

    if np.count_nonzero(mask) < self.num_samples:
      return np.empty((0, 3), dtype=np.float16)

    ys_crop, xs_crop = np.where(mask)
    total_hand_pts = len(xs_crop)

    if total_hand_pts < self.num_samples:
      return np.empty((0, 3), dtype=np.float16)

    # sample points
    indices = np.random.choice(total_hand_pts, size=self.num_samples, replace=False)
    xs_sel_crop = xs_crop[indices]
    ys_sel_crop = ys_crop[indices]
    zs = depth_m[ys_sel_crop, xs_sel_crop]

    # restore orig ccords
    xs_orig = xs_sel_crop + offset_x
    ys_orig = ys_sel_crop

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    xs_3d = (xs_orig - cx) * zs / fx
    ys_3d = (ys_orig - cy) * zs / fy
    zs_3d = zs

    points_3d = np.stack([xs_3d, ys_3d, zs_3d], axis=-1).astype(np.float16)
    return points_3d