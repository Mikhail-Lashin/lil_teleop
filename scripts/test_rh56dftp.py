# Test for Inspire Hand RH56DFTP

from RH56DFTP.RH56DFTP_TCP import RH56DFTP_TCP
from Register.RegisterKey.ftp_registers_keys import (
    POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5
)

import logging
logging.getLogger("RH56DFTP").setLevel(logging.ERROR)
logging.getLogger("pymodbus").setLevel(logging.ERROR)

import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

IP = "192.168.123.211"
PORT = 6000
MOVEMENT_DELAY = 0.2

FINGER_NAMES = {
    0: "Little",
    1: "Ring", 
    2: "Middle",
    3: "Index",
    4: "Thumb (flexion)",
    5: "Thumb (rotation)"
}

# finger limits: min~0 (straight), max~1800 (bent)
POSES = {
    "1": {
        "name": "🖐️  Open hand",
        "description": "All fingers straight",
        "positions": [0, 0, 0, 0, 0, 0]
    },
    "2": {
        "name": "✊️  Fist",
        "description": "All fingers bent",
        "positions": [1800, 1800, 1800, 1800, 1800, 0]
    },
    "3": {
        "name": "✌️   V",
        "description": "Index and middle fingers pointing upwards, others bent",
        "positions": [1800, 1800, 0, 0, 1800, 0]
    },
    "4": {
        "name": "🤘️  Rock",
        "description": "Index and little fingers raised, the middle and ring fingers are bent",
        "positions": [0, 1800, 1800, 0, 1800, 0]
    },
    "5": {
        "name": "👍️  Super",
        "description": "Thumb up, others bent",
        "positions": [1800, 1800, 1800, 1800, 0, 0]
    },
    "6": {
        "name": "👆️  Pointing",
        "description": "Index finger extended",
        "positions": [1800, 1800, 1800, 0, 1800, 0]
    }
}

def print_menu():
    print("\n" + "=" * 50)
    print("INSPIRE HAND RH56DFTP TEST")
    print("=" * 50)
    print("\nTest poses:")
    
    for key in ["1", "2", "3", "4", "5", "6"]:
        pose = POSES[key]
        print(f"  {key:>2}. {pose['name']}")
    
    print("\n  R. Reset (open hand)")
    print("  X. Exit.")
    print("-" * 50)

def move_finger(client, finger_id, position, finger_name):
    ok = client.set(finger_id, position)
    time.sleep(MOVEMENT_DELAY)
    return ok

def execute_pose(client, pose_key):
    if pose_key not in POSES:
        print("\n>>> Unknown pose!")
        return False
    
    pose = POSES[pose_key]
    
    finger_registers = [POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5]
    positions = pose["positions"]
    
    for i in range(6):
        move_finger(client, finger_registers[i], positions[i], FINGER_NAMES[i])
    
    print(f"\n>>> Set pose '{pose['name']}'")
    return True

def reset_pose(client):
    pose = POSES["1"]
    
    finger_registers = [POS_SET_0, POS_SET_1, POS_SET_2, POS_SET_3, POS_SET_4, POS_SET_5]
    positions = pose["positions"]
    
    for i in range(6):
        move_finger(client, finger_registers[i], positions[i], FINGER_NAMES[i])

def main():    
    try:
        client = RH56DFTP_TCP(host=IP, port=PORT)
        reset_pose(client)
        time.sleep(1)
        
        print(f"\n>>> Connected to {IP}:{PORT}")
        
        while True:
            print_menu()
            choice = input("\n>>> Select pose: ").strip().upper()
            
            if choice == "X":
                print("\n>>> Exiting...")
                break
            elif choice == "R":
                reset_pose(client)
            else:
                execute_pose(client, choice)
            
            time.sleep(0.5)
        
        client.close()
        print("\n>>> Connection closed.")
        
    except KeyboardInterrupt:
        print("\n\n>>> Terminated by user.")
    except Exception as e:
        print(f"\n>>> ERROR: {e}")
        print(f">>> Check connection to {IP}:{PORT}")
        
        subnet = IP.rsplit('.', 1)[0]
        print(f">>> Try set PC IPv4 manually in same subnet (e.g. ip {subnet}.100, subnet 255.255.255.0)")
    finally:
        try:
            client.close()
        except:
            pass

if __name__ == "__main__":
    main()