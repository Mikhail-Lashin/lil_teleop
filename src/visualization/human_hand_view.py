import numpy as np
import rerun as rr
from utils.one_euro import OneEuroFilter

class HumanHandView:
    """Render operator hand skeleton in Rerun"""
    
    HUMAN_SKELETON_LINKS = [
        (0, 1), (1, 2), (2, 3), (3, 4),             # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),             # index
        (0, 9), (9, 10), (10, 11), (11, 12),        # middle
        (0, 13), (13, 14), (14, 15), (15, 16),      # ring
        (0, 17), (17, 18), (18, 19), (19, 20)       # pinky
    ]
    
    DEFAULT_POSE = np.array([
        [0.00,  0.00,  0.00],                                                                   #     0: wrist
        [0.02,  0.02,  0.00], [0.04,  0.04,  0.00], [0.06,  0.05,  0.00], [0.08,  0.05,  0.00], #   1-4: thumb
        [0.03,  0.00,  0.01], [0.07,  0.00,  0.02], [0.11,  0.00,  0.02], [0.15,  0.00,  0.02], #   5-8: index
        [0.03, -0.01,  0.00], [0.08, -0.02,  0.00], [0.13, -0.02,  0.00], [0.17, -0.02,  0.00], #  9-12: middle
        [0.03, -0.02, -0.01], [0.07, -0.03, -0.01], [0.11, -0.03, -0.01], [0.14, -0.03, -0.01], # 13-16: ring
        [0.02, -0.03, -0.02], [0.05, -0.04, -0.02], [0.08, -0.04, -0.02], [0.10, -0.04, -0.02]  # 17-20: pinky
    ])
    
    def __init__(self, root_entity, freq=50, min_cutoff=2.0, beta=0.03, d_cutoff=1.0):
        self.root_entity = root_entity
        self.filter = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def update(self, joints):
        if joints is not None:
            raw_joints = np.array(joints) - np.array(joints)[0]
        else:
            raw_joints = self.DEFAULT_POSE

        # filter
        filtered_joints = self.filter.filter(raw_joints)

        # joint points
        rr.log(
            f"{self.root_entity}/Joints",
            rr.Points3D(
                filtered_joints,
                colors=[[0, 200, 255] for _ in range(len(filtered_joints))],
                radii=0.005
            )
        )

        # links
        bone_strips = [[filtered_joints[start], filtered_joints[end]] for start, end in self.HUMAN_SKELETON_LINKS]
        rr.log(
            f"{self.root_entity}/Skeleton_Bones",
            rr.LineStrips3D(
                bone_strips,
                colors=[[255, 255, 255] for _ in range(len(self.HUMAN_SKELETON_LINKS))],
                radii=0.001
            )
        )