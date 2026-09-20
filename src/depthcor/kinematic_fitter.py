"""
src/depthcor/kinematic_fitter.py

Modular, kinematic-aware hand mesh refinement using MANO skeletal forward kinematics.
Optimizes finger joint angles (Delta Theta) and wrist orientation (Delta Wrist)
using L-BFGS with Strong Wolfe line search, rigid scale locking (1.0), and analytic Z-prealignment.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch


def rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """
    Численно устойчивый дифференцируемый перевод axis-angle в матрицу поворота 3x3.
    Использует ряд Тейлора при theta -> 0 для устранения сингулярности деления на ноль.
    """
    theta_sq = torch.sum(rot_vecs ** 2, dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq + 1e-12)

    use_taylor = theta_sq < 1e-6
    sin_over_theta = torch.where(use_taylor, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
    one_minus_cos_over_theta_sq = torch.where(
        use_taylor, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / (theta_sq + 1e-12)
    )

    rx, ry, rz = rot_vecs[..., 0], rot_vecs[..., 1], rot_vecs[..., 2]
    zeros = torch.zeros_like(rx)

    K = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=-1).reshape(rot_vecs.shape[:-1] + (3, 3))

    eye = torch.eye(3, dtype=rot_vecs.dtype, device=rot_vecs.device)
    return eye + sin_over_theta.unsqueeze(-1) * K + one_minus_cos_over_theta_sq.unsqueeze(-1) * torch.matmul(K, K)


class KinematicDepthFitter:
    def __init__(
        self,
        mano_model: torch.nn.Module,
        faces: Union[np.ndarray, torch.Tensor],
        device: Union[str, torch.device] = "cuda",
        mask_erosion: int = 2,
        mask_dilation: int = 0,
        ik_iters: int = 12,
        ik_lr: float = 0.8,
        lambda_reg: float = 0.01,
        max_grad_norm: float = 0.05,
    ):
        self.device = torch.device(device)
        self.mano = mano_model.to(self.device).eval()
        self.mask_erosion = mask_erosion
        self.mask_dilation = mask_dilation
        self.ik_iters = ik_iters
        self.ik_lr = ik_lr
        self.lambda_reg = lambda_reg
        self.max_grad_norm = max_grad_norm

        if not isinstance(faces, torch.Tensor):
            self.faces = torch.tensor(faces, dtype=torch.long, device=self.device)
        else:
            self.faces = faces.to(self.device).long()

    # =========================================================================
    # 1. 2D ПРОЕКЦИЯ
    # =========================================================================
    def project_mesh_2d(
        self,
        mesh_verts: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_cam_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Точная пинхол-проекция меша HaMeR на кадр 256x256 (f=5000, cx=128, cy=128)."""
        if pred_cam_t is not None:
            t = pred_cam_t.to(self.device).float()
            tx, ty, tz = t[0], t[1], t[2]
        else:
            cam = pred_cam.to(self.device).float()
            s_cam, tx, ty = cam[0], cam[1], cam[2]
            tz = 2.0 * 5000.0 / (256.0 * s_cam + 1e-8)

        u = 5000.0 * (mesh_verts[:, 0] + tx) / (mesh_verts[:, 2] + tz) + 128.0
        v = 5000.0 * (mesh_verts[:, 1] + ty) / (mesh_verts[:, 2] + tz) + 128.0
        return torch.stack([u, v], dim=-1)

    def project_depth_points_2d(
        self, pts: torch.Tensor, intrinsics_256: Dict[str, float]
    ) -> torch.Tensor:
        """Проекция точек RealSense на кадр 256x256 через интринсики."""
        fx, fy = intrinsics_256["fx"], intrinsics_256["fy"]
        cx, cy = intrinsics_256["cx"], intrinsics_256["cy"]

        u = (pts[:, 0] * fx / pts[:, 2]) + cx
        v = (pts[:, 1] * fy / pts[:, 2]) + cy
        return torch.stack([u, v], dim=-1)

    # =========================================================================
    # 2. РАСТЕРИЗАЦИЯ И ЭРОЗИЯ МАСКИ (ОТСЕЧКА КРАЕВ ПАЛЬЦЕВ)
    # =========================================================================
    def rasterize_silhouette_mask(self, uv_mesh: torch.Tensor) -> np.ndarray:
        """Заливает силуэт меша и делает эрозию, срезая шумные боковины пальцев."""
        uv_np = uv_mesh.detach().cpu().numpy().astype(np.int32)
        faces_np = self.faces.detach().cpu().numpy()
        triangles = uv_np[faces_np]

        mask = np.zeros((256, 256), dtype=np.uint8)
        for tri in triangles:
            cv2.fillConvexPoly(mask, tri, 255)

        if self.mask_erosion > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.mask_erosion + 1, 2 * self.mask_erosion + 1)
            )
            mask = cv2.erode(mask, kernel)

        return mask

    def filter_depth_points(
        self, uv_pts: torch.Tensor, pts: torch.Tensor, silhouette_mask: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Оставляет только точки, попавшие внутрь маски."""
        u_int = torch.round(uv_pts[:, 0]).long()
        v_int = torch.round(uv_pts[:, 1]).long()

        in_bounds = (u_int >= 0) & (u_int < 256) & (v_int >= 0) & (v_int < 256)
        mask_t = torch.from_numpy(silhouette_mask).to(self.device) > 0

        valid_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
        valid_idx = in_bounds.nonzero(as_tuple=True)[0]
        valid_mask[valid_idx] = mask_t[v_int[valid_idx], u_int[valid_idx]]

        return pts[valid_mask], uv_pts[valid_mask], valid_mask

    # =========================================================================
    # 3. ОТБОР ФРОНТАЛЬНЫХ ВЕРШИН (BACK-FACE CULLING)
    # =========================================================================
    def compute_vertex_normals(self, verts: torch.Tensor) -> torch.Tensor:
        """Вычисляет векторные 3D-нормали вершин меша."""
        v0 = verts[self.faces[:, 0]]
        v1 = verts[self.faces[:, 1]]
        v2 = verts[self.faces[:, 2]]

        face_normals = torch.linalg.cross(v1 - v0, v2 - v0)
        v_normals = torch.zeros_like(verts)
        v_normals.index_add_(0, self.faces[:, 0], face_normals)
        v_normals.index_add_(0, self.faces[:, 1], face_normals)
        v_normals.index_add_(0, self.faces[:, 2], face_normals)

        return v_normals / (torch.norm(v_normals, dim=1, keepdim=True) + 1e-8)

    def get_front_facing_vertices(self, mesh_verts: torch.Tensor) -> torch.Tensor:
        """Отбирает только центральные фронтальные вершины (Nz < -0.30)."""
        v_normals = self.compute_vertex_normals(mesh_verts)
        view_dirs = mesh_verts / (torch.norm(mesh_verts, dim=1, keepdim=True) + 1e-8)

        is_front = torch.sum(v_normals * view_dirs, dim=1) < -0.30
        front_indices = torch.where(is_front)[0]

        if len(front_indices) < 50:
            front_indices = torch.arange(mesh_verts.shape[0], device=self.device)

        return front_indices

    def match_points_to_vertices(
        self,
        chosen_uv: torch.Tensor,
        uv_mesh: torch.Tensor,
        mesh_verts: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Связывает каждую точку глубины с конкретной вершиной ладони/пальцев."""
        cand_uv = uv_mesh[candidate_indices]
        dists = torch.cdist(chosen_uv, cand_uv)
        matched_local_ids = torch.argmin(dists, dim=1)
        matched_vert_ids = candidate_indices[matched_local_ids]

        return mesh_verts[matched_vert_ids], matched_vert_ids

    # =========================================================================
    # 4. ГЛОБАЛЬНАЯ МЕТРИЧЕСКАЯ ПОСАДКА (ФИКСИРОВАННЫЙ МАСШТАБ 1.0 + Z-SHIFT)
    # =========================================================================
    def estimate_global_alignment(
        self, src_pts: torch.Tensor, dst_pts: torch.Tensor, fix_scale_to_one: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Находит поворот R и сдвиг t (алгоритм Кабша при scale=1.0)."""
        p_mean = torch.mean(src_pts, dim=0)
        q_mean = torch.mean(dst_pts, dim=0)

        P_c = src_pts - p_mean
        Q_c = dst_pts - q_mean
        num_pts = src_pts.shape[0]

        H = torch.matmul(P_c.T, Q_c) / num_pts
        U, S, V_t = torch.linalg.svd(H)
        det = torch.linalg.det(V_t.T) * torch.linalg.det(U.T)
        d = torch.diag(
            torch.tensor([1.0, 1.0, 1.0 if det > 0 else -1.0], device=self.device)
        )

        R = torch.matmul(torch.matmul(V_t.T, d), U.T)

        if fix_scale_to_one:
            # ЖЕСТКО держим масштаб 1:1, запрещая алгоритму уменьшать руку!
            scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        else:
            var_p = torch.sum(P_c ** 2) / num_pts
            scale = (1.0 / (var_p + 1e-8)) * torch.sum(S * torch.diag(d))

        t = q_mean - scale * torch.matmul(R, p_mean)

        return scale, R, t

    def align_depth_offset(
        self,
        verts: torch.Tensor,
        scale: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        chosen_pts: torch.Tensor,
        matched_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Аналитически сажает меш в ноль по Z через медиану разности глубин."""
        v_metric = scale * torch.matmul(verts, R.T) + t
        z_diff = chosen_pts[:, 2] - v_metric[matched_ids, 2]
        delta_z = torch.median(z_diff)

        t_corrected = t.clone()
        t_corrected[2] += delta_z
        return t_corrected

    # =========================================================================
    # 5. КИНЕМАТИЧЕСКАЯ ОПТИМИЗАЦИЯ (L-BFGS С ПОИСКОМ ПО ЛУЧУ)
    # =========================================================================
    def optimize_kinematics(
        self,
        pred_mano_params: Dict[str, torch.Tensor],
        scale: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        chosen_pts: torch.Tensor,
        matched_vert_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Квази-ньютоновский догиб суставов MANO через L-BFGS:
        - Линейный поиск Strong Wolfe гарантирует отсутствие овершота («перелета»).
        - Оптимизируются углы пальцев delta_theta и микро-наклон запястья delta_wrist.
        """
        init_global_orient = pred_mano_params["global_orient"].to(self.device).clone()
        init_hand_pose = pred_mano_params["hand_pose"].to(self.device).clone()
        betas = pred_mano_params["betas"].to(self.device).clone()

        delta_theta = torch.zeros((1, 15, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        delta_wrist = torch.zeros((1, 1, 3), dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [delta_theta, delta_wrist],
            lr=self.ik_lr,
            max_iter=self.ik_iters,
            history_size=5,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            cur_hand_pose = torch.matmul(init_hand_pose, rodrigues(delta_theta))
            cur_global_orient = torch.matmul(init_global_orient, rodrigues(delta_wrist))

            mano_out = self.mano(
                global_orient=cur_global_orient,
                hand_pose=cur_hand_pose,
                betas=betas,
                pose2rot=False,
            )
            v_metric = scale * torch.matmul(mano_out.vertices[0], R.T) + t

            matched_v = v_metric[matched_vert_ids]
            loss_depth = torch.mean((chosen_pts - matched_v) ** 2)

            # Регуляризация удерживает позу вблизи предсказания HaMeR
            loss_reg = torch.mean(delta_theta ** 2) + 2.0 * torch.mean(delta_wrist ** 2)
            total_loss = loss_depth + self.lambda_reg * loss_reg
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([delta_theta, delta_wrist], max_norm=self.max_grad_norm)
            return total_loss

        optimizer.step(closure)

        # Финальный проход
        with torch.no_grad():
            final_pose = torch.matmul(init_hand_pose, rodrigues(delta_theta))
            final_orient = torch.matmul(init_global_orient, rodrigues(delta_wrist))
            final_out = self.mano(
                global_orient=final_orient,
                hand_pose=final_pose,
                betas=betas,
                pose2rot=False,
            )
            v_final = scale * torch.matmul(final_out.vertices[0], R.T) + t
            j_final = scale * torch.matmul(final_out.joints[0], R.T) + t

        return v_final, j_final.detach().cpu().numpy(), final_pose.detach().cpu().numpy()

    # =========================================================================
    # 6. КАНОНИЧЕСКИЙ БАЗИС КИСТИ (DEX-RETARGETING)
    # =========================================================================
    def compute_hand_frame(self, joints_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert joints_cam.shape == (21, 3)
        points = joints_cam[[0, 5, 9], :]

        x_vector = points[0] - points[2]
        pts_centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(pts_centered)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / (np.linalg.norm(x) + 1e-8)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1

        wrist_rot = np.stack([x, normal, z], axis=1)
        joints_rel = joints_cam - joints_cam[0]
        joints_local = np.round(joints_rel @ wrist_rot, 4)

        return joints_local, np.round(wrist_rot, 4)

    # =========================================================================
    # ГЛАВНЫЙ СЦЕНАРИЙ (ORCHESTRATOR)
    # =========================================================================
    def refine(
        self,
        pred_mano_params: Dict[str, torch.Tensor],
        mesh_verts: torch.Tensor,
        depth_points: Optional[Union[np.ndarray, torch.Tensor]],
        pred_cam: torch.Tensor,
        intrinsics_256: Dict[str, float],
        pred_cam_t: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, torch.Tensor],
        Tuple[np.ndarray, np.ndarray, torch.Tensor, Dict[str, Any]],
    ]:
        verts = mesh_verts.to(self.device).float() if isinstance(mesh_verts, torch.Tensor) \
            else torch.tensor(mesh_verts, dtype=torch.float32, device=self.device)

        if depth_points is None or len(depth_points) < 10:
            return self._fallback(pred_mano_params, verts, return_details)

        pts = depth_points.to(self.device).float() if isinstance(depth_points, torch.Tensor) \
            else torch.tensor(depth_points, dtype=torch.float32, device=self.device)

        # 1. 2D Проекция
        uv_mesh = self.project_mesh_2d(verts, pred_cam, pred_cam_t)
        uv_pts = self.project_depth_points_2d(pts, intrinsics_256)

        # 2. Фильтрация выбросов маской силуэта с эрозией (срезаем боковины пальцев)
        mask = self.rasterize_silhouette_mask(uv_mesh)
        chosen_pts, chosen_uv, valid_mask = self.filter_depth_points(uv_pts, pts, mask)

        if torch.sum(valid_mask) < 8:
            return self._fallback(pred_mano_params, verts, return_details, uv_pts, uv_mesh, mask)

        # 3. Отбор вершин строго по фронтальному хребту
        front_idx = self.get_front_facing_vertices(verts)
        src_verts, matched_ids = self.match_points_to_vertices(chosen_uv, uv_mesh, verts, front_idx)

        # 4. Глобальная метрическая посадка скелета (масштаб зафиксирован на 1.0)
        scale, R, t = self.estimate_global_alignment(src_verts, chosen_pts, fix_scale_to_one=True)

        # 4.1. Аналитическая посадка в ноль по Z перед догибом
        t = self.align_depth_offset(verts, scale, R, t, chosen_pts, matched_ids)

        # Исходный меш HaMeR в ТОЧНО ТАКОЙ ЖЕ глобальной позе (R, t) со scale=1.0 и Z-offset
        v_orig_aligned = scale * torch.matmul(verts, R.T) + t

        # 5. Кинематический квази-ньютоновский догиб (L-BFGS)
        mesh_refined, joints_cam, final_pose = self.optimize_kinematics(
            pred_mano_params, scale, R, t, chosen_pts, matched_ids
        )

        # 6. Локализация для контроллера робота
        joints_local, wrist_rot = self.compute_hand_frame(joints_cam)

        if return_details:
            details = {
                "valid_mask": valid_mask.detach().cpu().numpy(),
                "uv_pts": uv_pts.detach().cpu().numpy(),
                "uv_mesh": uv_mesh.detach().cpu().numpy(),
                "silhouette_mask": mask,
                "scale_est": float(scale.detach().cpu().item()),
                "optimized_hand_pose": final_pose,
                "verts_orig_aligned": v_orig_aligned.detach().cpu().numpy(),
            }
            return joints_local, wrist_rot, mesh_refined, details

        return joints_local, wrist_rot, mesh_refined

    def _fallback(
        self,
        pred_mano_params: Dict[str, torch.Tensor],
        verts: torch.Tensor,
        return_details: bool,
        uv_pts: Optional[torch.Tensor] = None,
        uv_mesh: Optional[torch.Tensor] = None,
        mask: Optional[np.ndarray] = None,
    ):
        with torch.no_grad():
            out = self.mano(
                global_orient=pred_mano_params["global_orient"].to(self.device),
                hand_pose=pred_mano_params["hand_pose"].to(self.device),
                betas=pred_mano_params["betas"].to(self.device),
                pose2rot=False,
            )
        j_cam = out.joints[0].detach().cpu().numpy()
        joints_local, wrist_rot = self.compute_hand_frame(j_cam)

        if return_details:
            details = {
                "valid_mask": np.zeros(0, dtype=bool),
                "uv_pts": uv_pts.detach().cpu().numpy() if uv_pts is not None else np.zeros((0, 2)),
                "uv_mesh": uv_mesh.detach().cpu().numpy() if uv_mesh is not None else np.zeros((778, 2)),
                "silhouette_mask": mask if mask is not None else np.zeros((256, 256), dtype=np.uint8),
                "verts_orig_aligned": verts.detach().cpu().numpy(),
            }
            return joints_local, wrist_rot, verts, details
        return joints_local, wrist_rot, verts