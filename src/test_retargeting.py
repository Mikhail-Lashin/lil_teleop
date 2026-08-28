from pathlib import Path
import tyro

from retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from retargeting.retargeting_config import RetargetingConfig

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(
    robot_dir: str,
    config_path: str,
    retargeting_type: RetargetingType,
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    config = RetargetingConfig.load_from_file(config_path)
    filepath = Path(config.urdf_path)
    
    hand_type = "Right" if "right" in config_path.lower() else "Left"
    robot_name = filepath.stem
    
    
    print(f">>> Hand: {robot_name}, type: {hand_type}")
    
    retargeting = RetargetingConfig.load_from_file(config_path).build()


if __name__ == "__main__":
    tyro.cli(main)
    
'''
def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):    
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos[retargeting_to_sapien])

'''