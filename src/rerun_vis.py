import cv2
import numpy as np
import zmq
import rerun as rr

from utils.one_euro import OneEuroFilter

ZMQ_PORT = 5555

SKELETON_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20)   # little
]


def default_pose():
    """Default hand pose in viewer when stream is waiting."""
    j = np.zeros((21, 3))
    j[1] = [0.02, 0.02, 0]; j[2] = [0.04, 0.04, 0]; j[3] = [0.06, 0.05, 0]; j[4] = [0.08, 0.05, 0]
    j[5] = [0.03, 0, 0.01]; j[6] = [0.07, 0, 0.02]; j[7] = [0.11, 0, 0.02]; j[8] = [0.15, 0, 0.02]
    j[9] = [0.03, -0.01, 0]; j[10] = [0.08, -0.02, 0]; j[11] = [0.13, -0.02, 0]; j[12] = [0.17, -0.02, 0]
    j[13] = [0.03, -0.02, -0.01]; j[14] = [0.07, -0.03, -0.01]; j[15] = [0.11, -0.03, -0.01]; j[16] = [0.14, -0.03, -0.01]
    j[17] = [0.02, -0.03, -0.02]; j[18] = [0.05, -0.04, -0.02]; j[19] = [0.08, -0.04, -0.02]; j[20] = [0.10, -0.04, -0.02]
    return j


def main():
    # rerun dashboard init
    rr.init("lil_teleop_dashboard", spawn=True)

    # subscribe to local ZeroMQ bus
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f">>> Rerun Visualizer connected to ZMQ bus: tcp://127.0.0.1:{ZMQ_PORT}")
    print(">>> Press Ctrl+C to stop")
    
    # def filter
    one_euro = OneEuroFilter(freq=50, beta=0.03, min_cutoff=2.0, d_cutoff=1.0)

    try:
        while True:
            metadata = sub_socket.recv_json()
            image_bytes = sub_socket.recv()

            # draw video
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rr.log("Camera/Video", rr.Image(frame_rgb))

            # draw hand
            joints = metadata.get("joints")
            if joints is not None:
                joints3d = np.array(joints)
            else:
                joints3d = default_pose()

            joints3d = joints3d - joints3d[0]
            joints_3d_filtered = one_euro.filter(joints3d)

            # joints
            rr.log(
                "Hand/Joint_Rotations",
                rr.Points3D(
                    joints_3d_filtered,
                    colors=[[0, 200, 255] for _ in range(len(joints_3d_filtered))],
                    radii=0.005
                )
            )

            # links
            bone_strips = [[joints_3d_filtered[start], joints_3d_filtered[end]] for start, end in SKELETON_LINKS]
            rr.log(
                "Hand/Skeleton_Bones",
                rr.LineStrips3D(
                    bone_strips,
                    colors=[[255, 255, 255] for _ in range(len(SKELETON_LINKS))],
                    radii=0.001
                )
            )

    except KeyboardInterrupt:
        print("\n>>> Rerun Visualizer stopped.")
    finally:
        sub_socket.close()
        context.term()


if __name__ == '__main__':
    main()