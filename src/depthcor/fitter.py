"""src/depth/fitter.py

Server-side GPU module for refining HaMeR hand mesh using sparse depth points
via Laplacian surface deformation and 2D projection gating.
"""

from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def get_hand_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
  """Канонический расчет базиса руки (из dex-retargeting)."""
  assert keypoint_3d_array.shape == (21, 3)
  points = keypoint_3d_array[[0, 5, 9], :]

  x_vector = points[0] - points[2]
  points = points - np.mean(points, axis=0, keepdims=True)
  _, _, v = np.linalg.svd(points)
  normal = v[2, :]

  x = x_vector - np.sum(x_vector * normal) * normal
  x = x / (np.linalg.norm(x) + 1e-8)
  z = np.cross(x, normal)

  if np.sum(z * (points[1] - points[2])) < 0:
    normal *= -1
    z *= -1
  return np.stack([x, normal, z], axis=1)


class DepthFitter:

  def __init__(
      self,
      mano_path: Union[str, Path] = "_DATA/data/mano/MANO_RIGHT.pkl",
      device: Union[str, torch.device] = "cuda",
      w_anchor: float = 5.0,  # Вес притяжения к точкам глубины
      max_pixel_dist: float = 12.0,  # 2D-радиус отсечки выбросов (в пикселях 256x256)
  ):
    self.device = torch.device(device)
    self.w_anchor = w_anchor
    self.max_pixel_dist = max_pixel_dist

    # 1. Загрузка данных MANO
    with open(mano_path, "rb") as f:
      mano_data = pickle.load(f, encoding="latin1")

    self.faces = torch.tensor(
        mano_data["f"].astype(np.int64), device=self.device
    )  # [1538, 3]

    # Обработка J_regressor (приводим к [21, 778])
    j_reg = mano_data["J_regressor"]
    if hasattr(j_reg, "toarray"):
      j_reg = j_reg.toarray()
    j_reg = torch.tensor(j_reg, dtype=torch.float32, device=self.device)

    if j_reg.shape[0] == 16:
      # Добавляем 5 кончиков пальцев MANO (Thumb: 745, Index: 333, Middle: 444, Ring: 555, Pinky: 678)
      tips_idx = [745, 333, 444, 555, 678]
      extra_rows = torch.zeros((5, 778), dtype=torch.float32, device=self.device)
      for i, v_idx in enumerate(tips_idx):
        extra_rows[i, v_idx] = 1.0
      j_reg = torch.cat([j_reg, extra_rows], dim=0)

    self.J_regressor = j_reg  # [21, 778]

    # 2. Предрасчет топологической матрицы Лапласа (L)
    self.L = self._build_laplacian(self.faces, num_verts=778)  # [778, 778]
    self.LtL = torch.matmul(self.L.T, self.L)

  def _build_laplacian(
      self, faces: torch.Tensor, num_verts: int
  ) -> torch.Tensor:
    """Строит стандартный комбинаторный Лапласиан графа L = D - A."""
    adj = torch.zeros(
        (num_verts, num_verts), dtype=torch.float32, device=self.device
    )
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]

    adj[v0, v1] = 1.0
    adj[v1, v0] = 1.0
    adj[v1, v2] = 1.0
    adj[v2, v1] = 1.0
    adj[v2, v0] = 1.0
    adj[v0, v2] = 1.0

    deg = torch.diag(torch.sum(adj, dim=1))
    return deg - adj

  def _compute_vertex_normals(self, verts: torch.Tensor) -> torch.Tensor:
    """Вычисляет векторные нормали для всех 778 вершин меша."""
    v0 = verts[self.faces[:, 0]]
    v1 = verts[self.faces[:, 1]]
    v2 = verts[self.faces[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # [1538, 3]

    v_normals = torch.zeros_like(verts)
    v_normals.index_add_(0, self.faces[:, 0], face_normals)
    v_normals.index_add_(1, self.faces[:, 1], face_normals)
    v_normals.index_add_(2, self.faces[:, 2], face_normals)

    return v_normals / (torch.norm(v_normals, dim=1, keepdim=True) + 1e-8)

  def refine(
      self,
      mesh_verts: torch.Tensor,
      depth_points: Optional[Union[np.ndarray, torch.Tensor]],
      intrinsics_256: Dict[str, float],
  ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Уточняет положение меша и вычисляет суставы в базисе руки.

    Args:
        mesh_verts: [778, 3] тензор вершин от HaMeR (в метрах)
        depth_points: [N, 3] массив точек RealSense (в метрах) или None
        intrinsics_256: словарь {'fx', 'fy', 'cx', 'cy'} для кадра 256x256

    Returns:
        joints3d_local: [21, 3] суставы в локальном базисе руки
        wrist_rot: [3, 3] матрица ориентации кисти
        mesh_refined: [778, 3] деформированный меш в пространстве камеры
    """
    if not isinstance(mesh_verts, torch.Tensor):
      mesh_verts = torch.tensor(
          mesh_verts, dtype=torch.float32, device=self.device
      )
    else:
      mesh_verts = mesh_verts.to(self.device).float()

    # --- FALLBACK: Если точек глубины нет, возвращаем результат чистого HaMeR ---
    if depth_points is None or len(depth_points) < 5:
      j_glob = (
          torch.matmul(self.J_regressor, mesh_verts).detach().cpu().numpy()
      )
      j_glob = j_glob - j_glob[0]
      w_rot = get_hand_frame(j_glob)
      return np.round(j_glob @ w_rot, 4), np.round(w_rot, 4), mesh_verts

    if not isinstance(depth_points, torch.Tensor):
      pts = torch.tensor(depth_points, dtype=torch.float32, device=self.device)
    else:
      pts = depth_points.to(self.device).float()

    fx, fy = intrinsics_256["fx"], intrinsics_256["fy"]
    cx, cy = intrinsics_256["cx"], intrinsics_256["cy"]

    # --- ШАГ 1: Грубая инициализация по оси Z (по медиане сенсора) ---
    z_target = torch.median(pts[:, 2])
    z_mesh_curr = torch.median(mesh_verts[:, 2])
    mesh_verts[:, 2] += z_target - z_mesh_curr

    # --- ШАГ 2: Фильтр нормалей (только фронтальные вершины) ---
    normals = self._compute_vertex_normals(mesh_verts)
    seen_mask = normals[:, 2] < -0.05  # Нормаль смотрит на камеру (ось -Z)
    seen_indices = torch.where(seen_mask)[0]

    if len(seen_indices) < 10:
      seen_indices = torch.arange(778, device=self.device)

    # --- ШАГ 3: 2D-контроль по проекции (Outlier Rejection) ---
    v_seen = mesh_verts[seen_indices]
    u_v = (v_seen[:, 0] * fx / v_seen[:, 2]) + cx
    v_v = (v_seen[:, 1] * fy / v_seen[:, 2]) + cy
    uv_seen = torch.stack([u_v, v_v], dim=-1)  # [K, 2]

    u_p = (pts[:, 0] * fx / pts[:, 2]) + cx
    v_p = (pts[:, 1] * fy / pts[:, 2]) + cy
    uv_pts = torch.stack([u_p, v_p], dim=-1)  # [N, 2]

    # Считаем попарные 2D-расстояния [N, K]
    dist_matrix_2d = torch.cdist(uv_pts, uv_seen)
    min_dist, nearest_seen_idx = torch.min(dist_matrix_2d, dim=1)

    # Оставляем только те точки сенсора, которые попали в 2D-проекцию кисти
    valid_pts_mask = min_dist <= self.max_pixel_dist
    if torch.sum(valid_pts_mask) < 5:
      valid_pts_mask = min_dist <= (self.max_pixel_dist * 2.0)

    chosen_pts = pts[valid_pts_mask]
    matched_vert_indices = seen_indices[nearest_seen_idx[valid_pts_mask]]

    # Если несколько точек привязались к одной вершине, усредняем таргеты
    unique_indices, inv_map = torch.unique(
        matched_vert_indices, return_inverse=True
    )
    M = len(unique_indices)

    target_P = torch.zeros(
        (M, 3), dtype=torch.float32, device=self.device
    ).scatter_add_(
        0, inv_map.unsqueeze(-1).expand(-1, 3), chosen_pts
    ) / torch.bincount(
        inv_map, minlength=M
    ).unsqueeze(
        -1
    ).float()

    # --- ШАГ 4: Лапласовская деформация меша (СЛАУ на GPU) ---
    # delta_0 сохраняет исходную анатомическую форму HaMeR
    delta_0 = torch.matmul(self.L, mesh_verts)

    # Матрица системы: A = L^T * L + w * I_anchors
    A = self.LtL.clone()
    A[unique_indices, unique_indices] += self.w_anchor

    # Правая часть: B = L^T * delta_0 + w * Target_P
    B = torch.matmul(self.L.T, delta_0)
    B[unique_indices] += self.w_anchor * target_P

    # Решаем A * V_prime = B
    mesh_refined = torch.linalg.solve(A, B)

    # --- ШАГ 5: Пересчет 21 сустава и вывод для ретаргетинга ---
    joints_cam = (
        torch.matmul(self.J_regressor, mesh_refined).detach().cpu().numpy()
    )
    joints_rel = joints_cam - joints_cam[0]  # Центрирование по запястью

    wrist_rot = get_hand_frame(joints_rel)
    joints3d_local = np.round(joints_rel @ wrist_rot, 4)
    wrist_rot = np.round(wrist_rot, 4)

    return joints3d_local, wrist_rot, mesh_refined