import cv2
import json
import socket
import time
import warnings
import threading
import traceback
import contextlib
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset

warnings.filterwarnings("ignore")

INPUT_VIDEO_PORT = 49102
PROCESSING_IP = "10.144.113.127"
PROCESSING_PORT = 50006

FPS_LOG_INTERVAL = 1.0


class UDPStreamReceiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # daemon receiver thread
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        while self.running:
            try:
                packet, _ = self.sock.recvfrom(65535)
                nparr = np.frombuffer(packet, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.sock.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Initialising HaMeR on {device}.")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    
    receiver = UDPStreamReceiver(INPUT_VIDEO_PORT)
    sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    processing_addr = (PROCESSING_IP, PROCESSING_PORT)
    
    print(f">>> Server started.")
    print(f">>> Listening video on port: {INPUT_VIDEO_PORT}")
    print(f">>> Sending MANO params to client: {PROCESSING_IP}:{PROCESSING_PORT}")
    
    t_start = time.perf_counter()
    counter = 0
    
    try:
        while True:
            frame = receiver.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            
            # frame preparation
            img_rgb = frame[:, :, ::-1]
            h, w, _ = img_rgb.shape
            bboxes = np.array([[0, 0, w, h]])
            is_right = np.array([1])
            
            # inference
            fnull = open(os.devnull, 'w')
            with contextlib.redirect_stdout(fnull):
                dataset = ViTDetDataset(model_cfg, img_rgb, bboxes, is_right, rescale_factor=1.0)
                sample = dataset[0]
            
            batch = {}
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if isinstance(v, torch.Tensor):
                    v = v.unsqueeze(0).to(device)
                batch[k] = v
                
            with torch.no_grad():
                out = model(batch)
            
            # mano params
            mano_pose_mats = out['pred_mano_params']['hand_pose'][0].cpu().numpy()
            mano_pose_aa = [R.from_matrix(m.reshape(3, 3)).as_rotvec().tolist() for m in mano_pose_mats]
            
            # Отправка JSON по UDP
            msg = json.dumps({
                'mano_params': mano_pose_aa,
                'timestamp': time.time()
            }).encode('utf-8')
            sock_out.sendto(msg, processing_addr)
            
            # FPS
            counter += 1
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            
            if elapsed >= FPS_LOG_INTERVAL:
                fps = counter / elapsed
                print(f">>> FPS: {fps:4.1f}", end='\r')
                counter = 0
                t_start = t_now

    except KeyboardInterrupt:
        print("\n>>> Server stopped by user.")
    except Exception as e:
        print(f"\n>>> Error: {e}")
        traceback.print_exc()
    finally:
        receiver.stop()
        sock_out.close()
        fnull.close()

if __name__ == '__main__':
    main()