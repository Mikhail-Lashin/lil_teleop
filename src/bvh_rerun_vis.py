import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import pybvh
import numpy as np
from pathlib import Path
import rerun as rr
import rerun.blueprint as rrb

from visualization.human_hand_view import HumanHandView
from visualization.robot_hand_view import RobotHandView
from retargeting.retargeting_config import RetargetingConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
ASSETS_DIR = PROJECT_ROOT / "assets"
URDF_PATH = ASSETS_DIR / "rh56dftp" / "rh56dftp_modified_right.urdf"

BVH_FILEPATH = PROJECT_ROOT / "test_data" / "bvh_misha_no_xyz_chr01_MAYA.bvh" # fix

# bvh nodes
RIGHT_HAND_NODES = [
    "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "EndSiteRightHandThumb3",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "EndSiteRightHandIndex3",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "EndSiteRightHandMiddle3",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "EndSiteRightHandRing3",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "EndSiteRightHandPinky3",
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
    hand_indices = [bvh.node_index[name] for name in RIGHT_HAND_NODES]
    total_frames = poses.shape[0]
    dt = getattr(bvh, "frame_time", 1.0 / 60.0)

    # set rerun windows
    human_origin = "Operator_Hand"
    robot_origin = "Robot_Hand"

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=human_origin, name="BVH Right Hand"),
            rrb.Spatial3DView(origin=robot_origin, name="Robot hand URDF")
        )
    )
    rr.init("bvh_retargeting_vis", spawn=True, default_blueprint=blueprint)

    fps = int(round(1.0 / dt)) if dt > 0 else 60
    bvh_view = HumanHandView(root_entity=human_origin, freq=fps)

    RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
    retargeter = RetargetingConfig.load_from_file(str(CONFIG_PATH)).build()
    robot_view = RobotHandView(
        urdf_path=URDF_PATH,
        retargeter=retargeter,
        root_entity=robot_origin
    )

    print(f">>> Start: {total_frames} frames, FPS: {fps}.")

    try:
        frame = 0
        while True:
            t_start = time.time()

            rr.set_time("frame_idx", sequence=frame)
            rr.set_time("time", duration=frame * dt)

            joints = poses[frame, hand_indices, :] * SCALE
            joints = joints - joints[0]
            wrist_rot = get_hand_frame(joints)
            joints = joints @ wrist_rot

            bvh_view.update(joints)
            robot_view.update(joints)

            frame = (frame + 1) % total_frames

            elapsed = time.time() - t_start
            delay = dt - elapsed
            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print("\n>>> Dashboard Manager stopped.")
    finally:
        robot_view.close()


if __name__ == '__main__':
    main()