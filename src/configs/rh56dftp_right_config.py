import numpy as np
import sys
from pathlib import Path

HAND_IP = "192.168.123.211"
HAND_PORT = 6000

MIN_ANGLE = 0
MAX_ANGLE = 1800
SEND_INTERVAL = 0.005
MAX_ANGLE_STEP = 300

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
URDF_PATH = PROJECT_ROOT / "assets" / "rh56dftp" / "rh56dftp_right.urdf"

FINGER_JOINTS = [
    "right_little_1_joint",   # flexion
    "right_ring_1_joint",     # flexion
    "right_middle_1_joint",   # flexion
    "right_index_1_joint",    # flexion
    "right_thumb_1_joint",    # flexion
    "right_thumb_2_joint"     # rotation
]
# joint limits in rads
MIN_RAD_LIMITS = np.array([ 0.0,  0.0,  0.0,  0.0,  0.5,  0.35]) 
MAX_RAD_LIMITS = np.array([ 1.2,  1.2,  1.2,  1.2,  0.9,  0.6])

# joint limits for registers of phythical hand
MIN_HAND_UNITS = np.array([0, 0, 0, 0, 0, 0])
MAX_HAND_UNITS = np.array([1800, 1800, 1800, 1800, 1800, 1800])