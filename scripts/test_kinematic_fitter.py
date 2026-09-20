"""
scripts/test_kinematic_fitter.py
python scripts/test_kinematic_fitter.py --bag-file test_data/test.bag --port 8080

Interactive Video Player & 3D Visualizer for HaMeR Kinematic IK with RealSense BAG streams.

Keyboard Controls in browser:
    [→] (Arrow Right) : Step forward
    [←] (Arrow Left)  : Step backward
    [Space]           : Play / Pause stream

Usage:
    python scripts/test_kinematic_fitter.py --bag-file test_data/test.bag --start-frame 1301 --rescale-factor 0.77 --port 8080
"""

from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import tyro
import viser

sys.path.append(str(Path(__file__).resolve().parent.parent))
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from src.depthcor.kinematic_fitter import KinematicDepthFitter
from src.depthcor.sampler import DepthSampler
from src.utils.frame_processing import crop_to_square


class BagStreamReader:
    """Обертка над pyrealsense2 с поддержкой перемотки назад через кольцевой кэш."""
    def __init__(self, bag_file: str, max_cache_size: int = 400):
        self.bag_file = bag_file
        self.max_cache_size = max_cache_size
        self._init_pipeline()

    def _init_pipeline(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(self.bag_file, repeat_playback=False)
        self.profile = self.pipeline.start(self.config)

        self.playback = self.profile.get_device().as_playback()
        self.playback.set_real_time(False)
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.color_fmt = color_stream.format()

        self.current_idx = 0
        self.cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def get_frame(self, target_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # 1. Если кадр уже в кэше — отдаем мгновенно (0 мс)
        if target_idx in self.cache:
            return self.cache[target_idx]

        # 2. Если нужно прыгнуть назад дальше кэша — перезапускаем поток
        if target_idx < self.current_idx:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self._init_pipeline()

        # 3. Читаем вперед до target_idx
        while self.current_idx <= target_idx:
            success, raw = self.pipeline.try_wait_for_frames(timeout_ms=3000)
            if not success:
                break

            aligned = self.align.process(raw)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            raw_c = np.asanyarray(c_frame.get_data())
            raw_d = np.asanyarray(d_frame.get_data())

            if self.color_fmt == rs.format.bgr8:
                color_rgb = cv2.cvtColor(raw_c, cv2.COLOR_BGR2RGB)
            else:
                color_rgb = raw_c.copy()

            depth_raw = raw_d.copy()

            # Сохраняем в кольцевой кэш
            self.cache[self.current_idx] = (color_rgb, depth_raw)
            if len(self.cache) > self.max_cache_size:
                furthest_key = max(self.cache.keys(), key=lambda k: abs(k - target_idx))
                del self.cache[furthest_key]

            if self.current_idx == target_idx:
                self.current_idx += 1
                return color_rgb, depth_raw

            self.current_idx += 1

        if len(self.cache) > 0:
            latest = max(self.cache.keys())
            return self.cache[latest]

        return None, None

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def create_2d_debug_overlay(
    img_crop_rgb: np.ndarray,
    uv_mesh: np.ndarray,
    uv_pts: np.ndarray,
    valid_mask: np.ndarray,
    faces: np.ndarray,
    silhouette_mask: Optional[np.ndarray] = None,
    scale_factor: float = 2.0,
    frame_idx: int = 0,
) -> np.ndarray:
    h, w, _ = img_crop_rgb.shape
    out_w, out_h = int(w * scale_factor), int(h * scale_factor)
    base_img = cv2.resize(img_crop_rgb, (out_w, out_h))
    overlay = base_img.copy()

    if silhouette_mask is not None:
        mask_scaled = cv2.resize(silhouette_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        overlay[mask_scaled > 0] = [60, 180, 80]
        cv2.addWeighted(overlay, 0.30, base_img, 0.70, 0, base_img)

    overlay_wire = base_img.copy()
    mesh_scaled = (uv_mesh * scale_factor).astype(np.int32)
    for f in faces:
        pt1 = tuple(mesh_scaled[f[0]])
        pt2 = tuple(mesh_scaled[f[1]])
        pt3 = tuple(mesh_scaled[f[2]])
        cv2.line(overlay_wire, pt1, pt2, (150, 255, 150), 1, cv2.LINE_AA)
        cv2.line(overlay_wire, pt2, pt3, (150, 255, 150), 1, cv2.LINE_AA)
        cv2.line(overlay_wire, pt3, pt1, (150, 255, 150), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay_wire, 0.40, base_img, 0.60, 0, base_img)

    pts_scaled = (uv_pts * scale_factor).astype(np.int32)

    # Красные крестики (отброшенные)
    outlier_pts = pts_scaled[~valid_mask]
    for pt in outlier_pts:
        x, y = pt[0], pt[1]
        sz = 4
        cv2.line(base_img, (x - sz, y - sz), (x + sz, y + sz), (255, 40, 40), 2, cv2.LINE_AA)
        cv2.line(base_img, (x - sz, y + sz), (x + sz, y - sz), (255, 40, 40), 2, cv2.LINE_AA)

    # Зеленые кружки (inliers)
    inlier_pts = pts_scaled[valid_mask]
    for pt in inlier_pts:
        cv2.circle(base_img, tuple(pt), 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(base_img, tuple(pt), 3, (0, 255, 100), -1, cv2.LINE_AA)

    n_in = int(np.sum(valid_mask))
    n_out = int(np.sum(~valid_mask))
    total = len(valid_mask)

    cv2.rectangle(base_img, (10, 10), (330, 115), (20, 20, 20), -1)
    cv2.rectangle(base_img, (10, 10), (330, 115), (100, 100, 100), 1)

    cv2.putText(base_img, f"Frame: #{frame_idx}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(base_img, f"Depth Points Total: {total}", (20, 57),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(base_img, f"[O] Inliers (Masked): {n_in}", (20, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(base_img, f"[X] Outliers (Filtered): {n_out}", (20, 107),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 60), 1, cv2.LINE_AA)

    return base_img


def main(
    bag_file: str = "test_data/test.bag",
    start_frame: int = 1301,
    port: int = 8080,
    rescale_factor: float = 0.77,
    mask_erosion: int = 2,
    ik_iters: int = 12,
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f">>> [INIT] Running Kinematic Stream Player on {dev}")

    reader = BagStreamReader(bag_file=bag_file)
    sampler = DepthSampler(num_samples=300)

    # Загрузка модели HaMeR
    print(">>> [HAMER] Loading model...")
    hamer_model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    hamer_model = hamer_model.to(dev).eval()

    fitter = KinematicDepthFitter(
        mano_model=hamer_model.mano,
        faces=hamer_model.mano.faces,
        device=dev,
        mask_erosion=mask_erosion,
        ik_iters=ik_iters,
    )

    faces_np = fitter.faces.detach().cpu().numpy()

    # Viser Web Server
    print(f"\n>>> [VISER] Open player in browser: http://localhost:{port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)

    # Состояние плеера
    state = {
        "current_frame": start_frame,
        "is_playing": False,
        "step_size": 1,
        "needs_update": True,
    }

    # Панель управления видеоплеером
    with server.gui.add_folder("🎬 Управление видеопотоком"):
        btn_prev = server.gui.add_button("◀ Назад (Left Arrow)")
        btn_play = server.gui.add_button("▶ Play / ⏸ Pause (Space)")
        btn_next = server.gui.add_button("Вперед ▶ (Right Arrow)")
        slider_frame = server.gui.add_slider("Кадр", min=0, max=5000, step=1, initial_value=start_frame)
        gui_step = server.gui.add_slider("Шаг перемотки", min=1, max=20, step=1, initial_value=1)
        txt_status = server.gui.add_markdown(f"**Статус**: Пауза | Кадр: {start_frame}")

    # Горячие клавиши в браузере
    cmd_next = server.gui.add_command("След. кадр", hotkey="arrowright")
    cmd_prev = server.gui.add_command("Пред. кадр", hotkey="arrowleft")
    cmd_play = server.gui.add_command("Воспроизведение/Пауза", hotkey="space")

    def step_forward():
        state["current_frame"] += state["step_size"]
        slider_frame.value = state["current_frame"]
        state["needs_update"] = True

    def step_backward():
        state["current_frame"] = max(0, state["current_frame"] - state["step_size"])
        slider_frame.value = state["current_frame"]
        state["needs_update"] = True

    def toggle_play():
        state["is_playing"] = not state["is_playing"]
        status_text = "Воспроизведение ▶" if state["is_playing"] else "Пауза ⏸"
        txt_status.content = f"**Статус**: {status_text} | Кадр: {state['current_frame']}"

    @btn_next.on_click
    def _(_):
        step_forward()

    @btn_prev.on_click
    def _(_):
        step_backward()

    @btn_play.on_click
    def _(_):
        toggle_play()

    @cmd_next.on_trigger
    def _(_):
        step_forward()

    @cmd_prev.on_trigger
    def _(_):
        step_backward()

    @cmd_play.on_trigger
    def _(_):
        toggle_play()

    @slider_frame.on_update
    def _(_):
        if state["current_frame"] != slider_frame.value:
            state["current_frame"] = slider_frame.value
            state["needs_update"] = True

    @gui_step.on_update
    def _(_):
        state["step_size"] = gui_step.value

    # Слои визуализации 3D сцены
    with server.gui.add_folder("🕹️ Управление 3D слоями"):
        cb_inliers = server.gui.add_checkbox("Точки внутри силуэта (Зеленые)", True)
        cb_outliers = server.gui.add_checkbox("Выбросы/Браслет (Красные)", True)
        cb_orig = server.gui.add_checkbox("Меш HaMeR ДО подтяжки (Серый)", True)
        cb_ref = server.gui.add_checkbox("Меш Kinematic IK (Зеленый)", True)

    # Инициализация дескрипторов 3D сцены
    dummy_pts = np.zeros((1, 3), dtype=np.float32)
    dummy_verts = np.zeros((778, 3), dtype=np.float32)

    h_inliers = server.scene.add_point_cloud(
        name="/depth/inliers",
        points=dummy_pts,
        colors=np.array([[0, 255, 100]], dtype=np.uint8),
        point_size=0.0005,
    )
    h_outliers = server.scene.add_point_cloud(
        name="/depth/outliers",
        points=dummy_pts,
        colors=np.array([[255, 40, 40]], dtype=np.uint8),
        point_size=0.0005,
    )
    h_mesh_orig = server.scene.add_mesh_simple(
        name="/mesh/original_hamer",
        vertices=dummy_verts,
        faces=faces_np,
        color=(180, 180, 180),
        opacity=0.35,
        wireframe=True,
    )
    h_mesh_ref = server.scene.add_mesh_simple(
        name="/mesh/kinematic_ik_refined",
        vertices=dummy_verts,
        faces=faces_np,
        color=(50, 220, 120),
        opacity=0.85,
        wireframe=False,
    )

    @cb_inliers.on_update
    def _(_):
        h_inliers.visible = cb_inliers.value

    @cb_outliers.on_update
    def _(_):
        h_outliers.visible = cb_outliers.value

    @cb_orig.on_update
    def _(_):
        h_mesh_orig.visible = cb_orig.value

    @cb_ref.on_update
    def _(_):
        h_mesh_ref.visible = cb_ref.value

    # 2D изображение
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    with server.gui.add_folder("📸 2D Проекция"):
        gui_img = server.gui.add_image(dummy_img, label="Exact Silhouette Overlay", format="jpeg")

    print("\n>>> [READY] Player started. Press Space to play, ← / → to step.")

    try:
        while True:
            # Если включен Play, автоматически шагаем вперед
            if state["is_playing"]:
                state["current_frame"] += state["step_size"]
                slider_frame.value = state["current_frame"]
                state["needs_update"] = True

            if not state["needs_update"]:
                time.sleep(0.01)
                continue

            state["needs_update"] = False
            cur_idx = state["current_frame"]
            t_start = time.perf_counter()

            color_rgb, depth_raw = reader.get_frame(cur_idx)
            if color_rgb is None or depth_raw is None:
                print(f">>> [STREAM] End of stream reached at frame {cur_idx}.")
                state["is_playing"] = False
                continue

            color_crop, depth_crop, offset_x = crop_to_square(color_rgb, depth_raw)
            scale_256 = 256.0 / color_crop.shape[0]

            intrinsics_256 = {
                "fx": reader.intr.fx * scale_256,
                "fy": reader.intr.fy * scale_256,
                "cx": (reader.intr.ppx - offset_x) * scale_256,
                "cy": reader.intr.ppy * scale_256,
            }
            intrinsics_orig = {
                "fx": reader.intr.fx,
                "fy": reader.intr.fy,
                "cx": reader.intr.ppx,
                "cy": reader.intr.ppy,
            }

            pts_3d = sampler.sample(
                depth_crop, intrinsics_orig, offset_x=offset_x, depth_scale=reader.depth_scale
            )

            img_256 = cv2.resize(color_crop, (256, 256))
            dataset = ViTDetDataset(
                model_cfg,
                img_256,
                np.array([[0, 0, 256, 256]]),
                np.array([1]),
                rescale_factor=rescale_factor,
            )
            batch = {
                k: (torch.from_numpy(v).unsqueeze(0).to(dev) if isinstance(v, np.ndarray) else v)
                for k, v in dataset[0].items()
            }

            with torch.no_grad():
                out = hamer_model(batch)

            pred_cam = out["pred_cam"][0]
            pred_cam_t = out["pred_cam_t"][0] if "pred_cam_t" in out else None
            mesh_verts_raw = out["pred_vertices"][0]
            pred_mano_params = out["pred_mano_params"]

            joints_local, wrist_rot, verts_refined, details = fitter.refine(
                pred_mano_params=pred_mano_params,
                mesh_verts=mesh_verts_raw,
                depth_points=pts_3d,
                pred_cam=pred_cam,
                intrinsics_256=intrinsics_256,
                pred_cam_t=pred_cam_t,
                return_details=True,
            )

            valid_mask = details.get("valid_mask", np.ones(len(pts_3d), dtype=bool))
            uv_pts = details.get("uv_pts", np.zeros((len(pts_3d), 2), dtype=np.float32))
            uv_mesh = details.get("uv_mesh", np.zeros((778, 2), dtype=np.float32))
            silhouette_mask = details.get("silhouette_mask", None)

            pts_inliers = pts_3d[valid_mask]
            pts_outliers = pts_3d[~valid_mask]

            verts_ref_np = verts_refined.detach().cpu().numpy()
            if "verts_orig_aligned" in details:
                verts_orig_np = details["verts_orig_aligned"]
            else:
                verts_orig_np = mesh_verts_raw.detach().cpu().numpy()

            # Обновление 3D объектов в Viser
            if len(pts_inliers) > 0:
                h_inliers.points = pts_inliers.astype(np.float32)
                h_inliers.colors = np.array([[0, 255, 100]] * len(pts_inliers), dtype=np.uint8)
            else:
                h_inliers.points = dummy_pts

            if len(pts_outliers) > 0:
                h_outliers.points = pts_outliers.astype(np.float32)
                h_outliers.colors = np.array([[255, 40, 40]] * len(pts_outliers), dtype=np.uint8)
            else:
                h_outliers.points = dummy_pts

            h_mesh_orig.vertices = verts_orig_np
            h_mesh_ref.vertices = verts_ref_np

            # 2D визуализация
            debug_2d_img = create_2d_debug_overlay(
                img_crop_rgb=img_256,
                uv_mesh=uv_mesh,
                uv_pts=uv_pts,
                valid_mask=valid_mask,
                faces=faces_np,
                silhouette_mask=silhouette_mask,
                scale_factor=2.0,
                frame_idx=cur_idx,
            )
            gui_img.image = debug_2d_img

            elapsed = time.perf_counter() - t_start
            fps = 1.0 / max(elapsed, 1e-4)
            status_text = "Воспроизведение ▶" if state["is_playing"] else "Пауза ⏸"
            txt_status.content = f"**Статус**: {status_text} | Кадр: #{cur_idx} | FPS: {fps:.1f} | Inliers: {len(pts_inliers)}"

    except KeyboardInterrupt:
        print("\n>>> Shutting down player.")
    finally:
        reader.close()


if __name__ == "__main__":
    tyro.cli(main)