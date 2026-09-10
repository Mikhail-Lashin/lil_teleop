import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import zmq
import numpy as np
from pathlib import Path

from retargeting.retargeting_config import RetargetingConfig

from RH56DFTP.RH56DFTP_TCP import RH56DFTP_TCP
from Register.RegisterKey.ftp_registers_keys import (
    POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5,
    DEFAULT_FORCE_SET_0, DEFAULT_FORCE_SET_1, DEFAULT_FORCE_SET_2,
    DEFAULT_FORCE_SET_3, DEFAULT_FORCE_SET_4, DEFAULT_FORCE_SET_5
)

ZMQ_PORT = 5555
ROBOT_IP = "192.168.123.211"
ROBOT_PORT = 6000

SAFE_FORCE_LIMIT = 50 # force lim (min=0, max=1000)
CHANGE_THRESHOLD = 6  # position change threshold (min_pos=0, max_pos=1800)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "retargeting" / "configs" / "teleop" / "rh56dftp_right.yml"
ASSETS_DIR = PROJECT_ROOT / "assets"

POS_KEYS = [POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5]
FORCE_KEYS = [
    DEFAULT_FORCE_SET_0, DEFAULT_FORCE_SET_1, DEFAULT_FORCE_SET_2,
    DEFAULT_FORCE_SET_3, DEFAULT_FORCE_SET_4, DEFAULT_FORCE_SET_5
]

R_hamer2urdf = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1]
])


def set_safe_force(robot_client, force_limit):
    print(f">>> Setting force limit: {force_limit}/1000 ({int(force_limit/10)}%)...")
    for i, key in enumerate(FORCE_KEYS):
        ok = robot_client.set(key, force_limit)
        if not ok:
            print(f">>> WARNING: can't set force for finger {i}!")


def qpos_to_modbus(qpos_dict, joint_limits):
    joint_mapping = [
        "right_little_1_joint",  # 0: Little
        "right_ring_1_joint",    # 1: Ring
        "right_middle_1_joint",  # 2: Middle
        "right_index_1_joint",   # 3: Index
        "right_thumb_2_joint",   # 4: Thumb flexion
        "right_thumb_1_joint",   # 5: Thumb rotation
    ]

    modbus_positions = []
    for joint_name in joint_mapping:
        if joint_name in qpos_dict:
            q = qpos_dict[joint_name]
            q_min, q_max = joint_limits.get(joint_name, (0.0, 1.4))
            norm = np.clip((q - q_min) / (q_max - q_min + 1e-6), 0.0, 1.0)
            pos = int(norm * 1800)
        else:
            pos = 0
        modbus_positions.append(pos)
        
    return modbus_positions


def main():
    # retargeter init
    print(">>> Initialising retargeter...")
    RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
    retargeter = RetargetingConfig.load_from_file(str(CONFIG_PATH)).build()
    optimizer = retargeter.optimizer
    
    # get joint names & joint limits from URDF
    joint_names = optimizer.robot.dof_joint_names
    raw_limits = optimizer.robot.joint_limits
    joint_limits = {
        name: (raw_limits[i, 0], raw_limits[i, 1]) 
        for i, name in enumerate(joint_names)
    }

    # connect to hand
    print(f">>> Connecting to hand: {ROBOT_IP}:{ROBOT_PORT}...")
    try:
        robot_client = RH56DFTP_TCP(host=ROBOT_IP, port=ROBOT_PORT)
        print(">>> Connected!")
    except Exception as e:
        print(f">>> ERROR: failed connection ({e})")
        return

    # set force lim
    set_safe_force(robot_client, SAFE_FORCE_LIMIT)

    # connect to ZMQ bus
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f">>> Teleop active. Connected to ZMQ: tcp://127.0.0.1:{ZMQ_PORT}")
    print(">>> Ctrl+C to stop.")

    last_sent_positions = [None] * 6

    try:
        while True:
            data = None
            while True:
                try:
                    meta = sub_socket.recv_json(flags=zmq.NOBLOCK)
                    img_bytes = sub_socket.recv(flags=zmq.NOBLOCK)
                    data = meta
                except zmq.Again:
                    break

            if data is None or "joints" not in data or data["joints"] is None:
                time.sleep(0.001)
                continue

            # human keypoints
            joints3d = np.array(data["joints"])
            joints_array = joints3d - joints3d[0]
            
            # local URDF frame
            joints_array = joints_array @ R_hamer2urdf.T

            # target vectors
            if optimizer.retargeting_type == "POSITION":
                ref_value = joints_array[optimizer.target_link_human_indices, :]
            else:
                origin_idx = optimizer.target_link_human_indices[0, :]
                task_idx = optimizer.target_link_human_indices[1, :]
                ref_value = joints_array[task_idx, :] - joints_array[origin_idx, :]

            # compute joint angles & set positions
            qpos = retargeter.retarget(ref_value)
            qpos_dict = dict(zip(joint_names, qpos))

            positions = qpos_to_modbus(qpos_dict, joint_limits)

            for i in range(6):
                if (last_sent_positions[i] is None or 
                    abs(positions[i] - last_sent_positions[i]) >= CHANGE_THRESHOLD):
                    
                    robot_client.set(POS_KEYS[i], positions[i])
                    last_sent_positions[i] = positions[i]

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n>>> Stopped by user.")
    except Exception as e:
        print(f"\n>>> ERROR: {e}")
    finally:
        print(">>> Finishing...")
        try:
            # reset hand pos to zero
            for i in range(6):
                robot_client.set(POS_KEYS[i], 0)
            robot_client.close()
        except:
            pass
        sub_socket.close()
        context.term()


if __name__ == "__main__":
    main()