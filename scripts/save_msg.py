import os
import cv2
import torch
import json
import time
import numpy as np
import contextlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset

def get_hand_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoints detected with HaMeR. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initialising HaMeR on {device}.")
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
                
    # frame preparation
    path = Path(__file__).resolve().parent.parent / "test_data" / "arm.jpg"
    frame = cv2.imread(str(path))[:, :, ::-1] # read & convert from BGR to RGB
    h, w, _ = frame.shape
    bboxes = np.array([[0, 0, w, h]])
    is_right = np.array([1])
    
    # inference
    fnull = open(os.devnull, 'w')
    with contextlib.redirect_stdout(fnull):
        dataset = ViTDetDataset(model_cfg, frame, bboxes, is_right, rescale_factor=1.0)
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
            
    fnull.close()
        
    # get joint positions & wrist rotation matrix
    joints3d_global = out['pred_keypoints_3d'][0].cpu().numpy() # joints in camera frame
    joints3d_global = joints3d_global - joints3d_global[0]
    
    wrist_rot = get_hand_frame(joints3d_global)
    
    joints3d = np.round(joints3d_global @ wrist_rot, 4) # # joints in hand frame
    wrist_rot = np.round(wrist_rot, 4)
    
    # save msg
    msg = {
        'joints': joints3d.tolist(),
        'hand_rotation': wrist_rot.tolist(),
        'timestamp': time.time()
    }
    
    msg_path = Path(__file__).resolve().parent.parent / "test_data" / "debug_msg.json"
    
    with open(msg_path, "w") as f:
        json.dump(msg, f, indent=4)
    
    print(f"\n>>> Saved msg: {msg_path}")

if __name__ == '__main__':
    main()