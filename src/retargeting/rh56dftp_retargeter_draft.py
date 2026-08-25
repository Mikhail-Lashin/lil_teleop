import os
import numpy as np
from yourdfpy import URDF

class RH56DFTPRetargeter:
    """operator hand -> robot hand"""
    
    def __init__(self, urdf_path):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        self.robot = URDF.load(urdf_path)
        self.finger_map = {
            "right_index_1_joint":  [0, 1, 2],
            "right_middle_1_joint": [3, 4, 5],
            "right_little_1_joint": [6, 7, 8],
            "right_ring_1_joint":   [9, 10, 11]
        }
        self.mimic_weights = self._get_weights()

    def _get_weights(self):
        weights = {}
        for name in self.finger_map.keys():
            w = 1.0
            for j_name, joint in self.robot.joint_map.items():
                if hasattr(joint, 'mimic') and joint.mimic and joint.mimic.joint == name:
                    w += joint.mimic.multiplier
            weights[name] = w
        return weights

    def compute_robot_angles(self, mano_aa_pose):
        commands = {}
        for r_joint, m_indices in self.finger_map.items():
            total_flexion = sum([mano_aa_pose[i][2] for i in m_indices if i < len(mano_aa_pose)])
            commands[r_joint] = float(total_flexion / self.mimic_weights[r_joint])
        
        if len(mano_aa_pose) >= 15:
            commands["right_thumb_1_joint"] = float(mano_aa_pose[14][0])
            commands["right_thumb_2_joint"] = float(mano_aa_pose[12][0])
        else:
            commands["right_thumb_1_joint"] = 0.0
            commands["right_thumb_2_joint"] = 0.0

        for name, val in commands.items():
            if name in self.robot.joint_map:
                limit = self.robot.joint_map[name].limit
                if limit:
                    commands[name] = float(np.clip(val, limit.lower, limit.upper))
                    
        return commands