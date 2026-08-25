import time
import zmq
import rerun as rr
import rerun.blueprint as rrb

from visualization.camera_view import CameraView
from visualization.human_hand_view import HumanHandView
from visualization.robot_hand_view import RobotHandView

from configs.rh56dftp_right_config import URDF_PATH, FINGER_JOINTS
from retargeting.rh56dftp_retargeter_draft import RH56DFTPRetargeter

ZMQ_PORT = 5555

def main():
    # set rerun windows
    cam_origin, human_hand_origin, robot_hand_origin = "Camera/Video", "Human_Hand", "Robot_Hand"
    
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=cam_origin, name="Video"),
            rrb.Spatial3DView(origin=human_hand_origin, name="Operator hand"),
            rrb.Spatial3DView(origin=robot_hand_origin, name="Robot hand URDF")
        )
    )
    rr.init("lil_teleop_dashboard", spawn=True, default_blueprint=blueprint)

    camera_view = CameraView(entity_path=cam_origin)
    human_view = HumanHandView(freq=50, min_cutoff=2.0, beta=0.03,
                               root_entity=human_hand_origin)
    robot_view = RobotHandView(urdf_path=URDF_PATH,
                               retargeter=RH56DFTPRetargeter(URDF_PATH),
                               joints=FINGER_JOINTS,
                               freq=50, min_cutoff=2.0, beta=0.03,
                               root_entity=robot_hand_origin)

    # connect to ZeroMQ bus
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{ZMQ_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f">>> Dashboard Manager running. Connected to ZMQ bus: tcp://127.0.0.1:{ZMQ_PORT}")
    print(">>> Ctrl+C to stop")

    try:
        while True:
            data = None # reset buffer for zero latensy
            while True:
                try:
                    meta = sub_socket.recv_json(flags=zmq.NOBLOCK)
                    img_bytes = sub_socket.recv(flags=zmq.NOBLOCK)
                    data = (meta, img_bytes)
                except zmq.Again:
                    break

            if data is None:
                time.sleep(0.001)
                continue

            # refresh windows
            metadata, image_bytes = data
            camera_view.update(image_bytes)
            human_view.update(metadata.get("joints"))
            robot_view.update(metadata.get("mano_params"))

    except KeyboardInterrupt:
        print("\n>>> Dashboard Manager stopped.")
    finally:
        robot_view.close()
        sub_socket.close()
        context.term()

if __name__ == '__main__':
    main()