import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional
import cv2
import numpy as np
import tyro

try:
    import pyrealsense2 as rs
    HAS_REALSENSE_LIB = True
except ImportError:
    HAS_REALSENSE_LIB = False

try:
    from utils.frame_processing import crop_to_square
except ImportError:
    def crop_to_square(frame):
        h, w = frame.shape[:2]
        dim = min(h, w)
        y, x = (h - dim) // 2, (w - dim) // 2
        return frame[y:y + dim, x:x + dim], 0, 0

RESOLUTION = (256, 256)
STREAM_PORT = 8080

# global preview frame state
latest_jpeg = None
jpeg_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class StreamHandler(BaseHTTPRequestHandler):
    """Serve MJPEG stream for VS Code Simple Browser."""
    def log_message(self, format, *args):
        pass  # suppress HTTP request logging

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = (
                "<html><body style='margin:0;background:#1e1e1e;display:flex;"
                "justify-content:center;align-items:center;height:100vh;'>"
                "<img src='/stream.mjpg' style='width:384px;height:384px;'/></body></html>"
            )
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with jpeg_lock:
                        jpg = latest_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.016)
            except (ConnectionResetError, BrokenPipeError):
                pass


def init_camera(bag_file: Optional[str] = None, fps: int = 60):
    if HAS_REALSENSE_LIB:
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            if bag_file:
                config.enable_device_from_file(bag_file, repeat_playback=True)
            else:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
            
            profile = pipeline.start(config)
            if bag_file:
                playback = profile.get_device().as_playback()
                playback.set_real_time(True)
                print(f">>> [CAMERA] Opened BAG file: {bag_file}")
            else:
                print(">>> [CAMERA] Connected to Intel RealSense!")
            return "realsense", pipeline
        except Exception as e:
            print(f">>> [CAMERA] RealSense init failed ({e}). Falling back to webcam...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(">>> [CAMERA] ERROR: No camera available!")
    print(">>> [CAMERA] Using standard webcam")
    return "opencv", cap


def save_clip_chunk(frames, save_path: Path, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(save_path), fourcc, fps, RESOLUTION)
    for f in frames:
        out.write(f)
    out.release()


def main(
    clip_type: str,
    clip_time: float = 5.0,
    fps: int = 60,
    bag: Optional[str] = None,
    output_dir: Path = Path("./recorded_clips"),
    http_port: int = STREAM_PORT,
):
    save_dir = output_dir / clip_type
    save_dir.mkdir(parents=True, exist_ok=True)

    cam_type, cam_obj = init_camera(bag, fps)

    # start preview HTTP server
    httpd = ThreadedHTTPServer(("0.0.0.0", http_port), StreamHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    frames_per_clip = int(clip_time * fps)
    print(f"\n>>> Stream URL for VS Code: http://localhost:{http_port}")
    print(f">>> (VS Code: Ctrl+Shift+P -> 'Simple Browser: Show' -> http://localhost:{http_port})")
    print(f">>> Session: '{clip_type}' | Target clip: {clip_time}s ({frames_per_clip} frames)")
    print(">>> Controls: [Enter] - Start/Stop session | 'q' + [Enter] - Exit\n")

    is_recording = [False]
    recorded_frames = []
    running = [True]

    def capture_worker():
        global latest_jpeg
        while running[0]:
            if cam_type == "realsense":
                fs = cam_obj.wait_for_frames()
                cf = fs.get_color_frame()
                if not cf:
                    continue
                frame = np.asanyarray(cf.get_data())
                if bag:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cam_obj.read()
                if not ret:
                    time.sleep(0.01)
                    continue

            # crop to square & resize
            cropped, _, _ = crop_to_square(frame)
            resized = cv2.resize(cropped, RESOLUTION)

            if is_recording[0]:
                recorded_frames.append(resized)

            # preview frame with REC overlay
            preview = resized.copy()
            if is_recording[0]:
                cv2.circle(preview, (20, 20), 8, (0, 0, 255), -1)
                cv2.putText(preview, "REC", (35, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            _, enc = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            with jpeg_lock:
                latest_jpeg = enc.tobytes()

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()

    try:
        while True:
            cmd = input("Ready to record. Press [Enter] to start ('q' to quit): ").strip().lower()
            if cmd == "q":
                break

            recorded_frames.clear()
            is_recording[0] = True

            input(">>> [RECORDING IN PROGRESS] Press [Enter] to STOP... ")
            is_recording[0] = False

            total = len(recorded_frames)
            num_clips = total // frames_per_clip
            print(f">>> Captured {total} frames ({total / fps:.1f}s). Slicing into {num_clips} clips...")

            ts = int(time.time() * 1000)
            for idx in range(num_clips):
                start = idx * frames_per_clip
                chunk = recorded_frames[start : start + frames_per_clip]
                clip_path = save_dir / f"clip_{ts}_{idx:02d}.mp4"
                save_clip_chunk(chunk, clip_path, fps)
                print(f"  + Saved: {clip_path.name} ({len(chunk)} frames)")

            if total % frames_per_clip != 0:
                print(f"  - Dropped {total % frames_per_clip} leftover frames.")
            print()

    except KeyboardInterrupt:
        print("\n>>> Interrupted by user.")
    finally:
        running[0] = False
        capture_thread.join(timeout=1.0)
        httpd.shutdown()
        if cam_type == "realsense":
            cam_obj.stop()
        else:
            cam_obj.release()
        print(">>> Clean shutdown complete.")


if __name__ == "__main__":
    tyro.cli(main)