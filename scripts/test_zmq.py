import zmq

ZMQ_PUB_PORT = 5555

def main():
    print(f">>> Connecting to ZMQ Bus at tcp://127.0.0.1:{ZMQ_PUB_PORT}...")

    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{ZMQ_PUB_PORT}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print(">>> Waiting for 1 package from the bus...")

    metadata = sub_socket.recv_json()
    image_bytes = sub_socket.recv()

    print("\n>>> SUCCESS! Received 1 packet from ZeroMQ bus:")
    print(f">>> Timestamp    : {metadata.get('timestamp')}")
    print(f">>> 3D Joints    : {len(metadata['joints']) if metadata.get('joints') else 'None (Waiting for GPU server)'}")
    print(f">>> MANO Params  : {len(metadata['mano_params']) if metadata.get('mano_params') else 'None (Waiting for GPU server)'}")
    print(f">>> Image Frame  : {len(image_bytes)} bytes (JPEG)")

    sub_socket.close()
    context.term()

if __name__ == "__main__":
    main()