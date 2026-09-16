import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import pybvh
import numpy as np
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
from visualization.rerun_views import HumanHandView, RobotHandView
from retargeting.retargeting_config import RetargetingConfig

ASSETS_DIR = PROJECT_ROOT / "assets"
BVH_FILEPATH = PROJECT_ROOT / "test_data" / "bvh_misha_no_xyz_chr01_MAYA.bvh"

CONFIG_PATH_RIGHT = PROJECT_ROOT / "src" / "retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
CONFIG_PATH_LEFT = PROJECT_ROOT / "src" / "retargeting" / "configs" / "teleop" / "rh56dftp_left.yml"

URDF_PATH_RIGHT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_right.urdf"
URDF_PATH_LEFT = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_left.urdf"

# bvh nodes (right + left hand)
RIGHT_HAND_NODES = [
    "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "EndSiteRightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "EndSiteRightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "EndSiteRightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "EndSiteRightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "EndSiteRightHandPinky3",
]

LEFT_HAND_NODES = [
    "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "EndSiteLeftHandThumb3",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "EndSiteLeftHandIndex3",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "EndSiteLeftHandMiddle3",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "EndSiteLeftHandRing3",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "EndSiteLeftHandPinky3",
]

SCALE = 0.01 # scale [cm] to [m]

def get_hand_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Originates from dex-retargeting repo
        
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
    # load bvh data
    print(f">>> Loading BVH: {BVH_FILEPATH}")
    bvh = pybvh.read_bvh_file(str(BVH_FILEPATH))
    bvh = bvh.reorient_world_up('+z')
    bvh = bvh.rotate_vertical(np.pi / 2)

    poses = bvh.node_positions(centered="skeleton")
    r_indices = [bvh.node_index[name] for name in RIGHT_HAND_NODES]
    l_indices = [bvh.node_index[name] for name in LEFT_HAND_NODES]
    total_frames = poses.shape[0]
    dt = getattr(bvh, "frame_time", 1.0 / 240.0)

    # set rerun windows
    origin_bvh_left = "Operator_Hand_Left"
    origin_bvh_right = "Operator_Hand_Right"
    origin_robot_left = "Robot_Hand_Left"
    origin_robot_right = "Robot_Hand_Right"

    blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial3DView(origin=origin_bvh_left, name="BVH Left Hand"),
            rrb.Spatial3DView(origin=origin_bvh_right, name="BVH Right Hand"),
            rrb.Spatial3DView(origin=origin_robot_left, name="Robot Left Hand URDF"),
            rrb.Spatial3DView(origin=origin_robot_right, name="Robot Right Hand URDF"),
            grid_columns=2
        )
    )
    rr.init("bimanual_bvh_retargeting", spawn=True, default_blueprint=blueprint)

    fps = int(round(1.0 / dt)) if dt > 0 else 60
    bvh_view_right = HumanHandView(root_entity=origin_bvh_right, freq=fps)
    bvh_view_left = HumanHandView(root_entity=origin_bvh_left, freq=fps)

    RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
    retargeter_right = RetargetingConfig.load_from_file(str(CONFIG_PATH_RIGHT)).build()
    robot_view_right = RobotHandView(
        urdf_path=URDF_PATH_RIGHT,
        retargeter=retargeter_right,
        root_entity=origin_robot_right
    )

    retargeter_left = RetargetingConfig.load_from_file(str(CONFIG_PATH_LEFT)).build()
    robot_view_left = RobotHandView(
        urdf_path=URDF_PATH_LEFT,
        retargeter=retargeter_left,
        root_entity=origin_robot_left
    )

    print(f">>> Start: {total_frames} frames, FPS: {fps}.")

    try:
        frame = 0
        while True:
            t_start = time.time()

            rr.set_time("frame_idx", sequence=frame)
            rr.set_time("time", duration=frame * dt)

            # right
            joints_r = poses[frame, r_indices, :] * SCALE
            joints_r = joints_r - joints_r[0]
            rot_r = get_hand_frame(joints_r)
            joints_r_canonical = joints_r @ rot_r

            bvh_view_right.update(joints_r_canonical)
            robot_view_right.update(joints_r_canonical)

            # left
            joints_l = poses[frame, l_indices, :] * SCALE
            joints_l = joints_l - joints_l[0]
            rot_l = get_hand_frame(joints_l)
            joints_l_canonical = joints_l @ rot_l

            bvh_view_left.update(joints_l_canonical)
            robot_view_left.update(joints_l_canonical)

            frame = (frame + 1) % total_frames

            elapsed = time.time() - t_start
            delay = dt - elapsed
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n>>> Dashboard Manager stopped.")
    finally:
        robot_view_right.close()
        robot_view_left.close()


if __name__ == '__main__':
    main()