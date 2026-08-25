import cv2
import numpy as np
import rerun as rr

class CameraView:
    """Render video stream from camera in Rerun"""
    def __init__(self, entity_path):
        self.entity_path = entity_path

    def update(self, image_bytes):
        if image_bytes is None:
            return
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rr.log(self.entity_path, rr.Image(frame_rgb))