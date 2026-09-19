from typing import Optional, Tuple
import numpy as np

def crop_to_square(
    color: np.ndarray, depth: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
  h, w = color.shape[:2]
  offset_x = 0

  if w > h:
    offset_x = (w - h) // 2
    color_cropped = color[:, offset_x : offset_x + h]
    depth_cropped = depth[:, offset_x : offset_x + h] if depth is not None else None
  elif h > w:
    offset_y = (h - w) // 2
    color_cropped = color[offset_y : offset_y + w, :]
    depth_cropped = depth[offset_y : offset_y + w, :] if depth is not None else None
  else:
    color_cropped = color
    depth_cropped = depth

  return color_cropped, depth_cropped, offset_x