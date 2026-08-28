import numpy as np
from pathlib import Path
import tyro
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from retargeting.constants import RetargetingType
from retargeting.retargeting_config import RetargetingConfig

def main(
    robot_dir: str,
    config_path: str,
    retargeting_type: RetargetingType,
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config = RetargetingConfig.load_from_file(config_path)
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    
    print(f"\n>>> Hand:\n{robot_name}")
    
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retargeting_type = retargeting.optimizer.retargeting_type
    indices = retargeting.optimizer.target_link_human_indices
    
    # dummy operator hand joints
    joint_pos = np.array([
        [0.00,  0.00,  0.00],  # 0: wrist
        [0.02,  0.02,  0.00], [0.04,  0.04,  0.00], [0.06,  0.05,  0.00], [0.08,  0.05,  0.00], # 1-4: thumb
        [0.03,  0.00,  0.01], [0.07,  0.00,  0.02], [0.11,  0.00,  0.02], [0.15,  0.00,  0.02], # 5-8: index
        [0.03, -0.01,  0.00], [0.08, -0.02,  0.00], [0.13, -0.02,  0.00], [0.17, -0.02,  0.00], # 9-12: middle
        [0.03, -0.02, -0.01], [0.07, -0.03, -0.01], [0.11, -0.03, -0.01], [0.14, -0.03, -0.01], # 13-16: ring
        [0.02, -0.03, -0.02], [0.05, -0.04, -0.02], [0.08, -0.04, -0.02], [0.10, -0.04, -0.02]  # 17-20: little
    ])
    
    if retargeting_type == "POSITION":
        indices = indices
        ref_value = joint_pos[indices, :]
    else:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    
    # all robot hand joints rotations
    qpos = retargeting.retarget(ref_value)
    
    print(f"\n>>> Retargeting result: \n{qpos}")

if __name__ == "__main__":
    tyro.cli(main)