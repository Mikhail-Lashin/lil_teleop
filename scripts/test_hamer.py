import os
import cv2
import torch
import numpy as np
import time
import contextlib
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset

N_iter = 100

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
    
    # inference (1 cold startup + N_iter iterations)
    fnull = open(os.devnull, 'w')
    with contextlib.redirect_stdout(fnull):
        for i in range(N_iter + 1):
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
                
            # mano parameters
            mano_pose_mats = out['pred_mano_params']['hand_pose'][0].cpu().numpy()
            mano_pose_aa = [R.from_matrix(m.reshape(3,3)).as_rotvec() for m in mano_pose_mats]
            mano_params_list = [v.tolist() for v in mano_pose_aa]
                
            if i==0:
                t_start = time.perf_counter()
    
    print("\nMANO parameters:")
    for param in mano_params_list:
        print(param)
    
    # stats
    t_end = time.perf_counter()
    fps = N_iter / (t_end - t_start)
    print(f"\nFPS: {fps:4.1f}")       

if __name__ == '__main__':
    main()