import cv2
import json
import socket
import time
import threading
import zmq

SERVER_IP = "10.144.208.248" 
SERVER_PORT = 49102           # for video (client -> server)
CLIENT_PORT = 50006           # for json msgs (server -> client)

RESOLUTION = (256, 256)
JPEG_QUALITY = 60

ZMQ_PUB_PORT = 5555

class Receiver:
    """Recheive json messages from server in daemon process."""
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(0.5)
        self.latest_data = None
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65535)
                msg = json.loads(data.decode('utf-8'))
                with self.lock:
                    self.latest_data = msg
            except socket.timeout:
                pass
            except Exception:
                pass

    def get_data(self):
        with self.lock:
            return self.latest_data

    def stop(self):
        self.running = False
        self.sock.close()


def start_streamer():
    zmq_context = zmq.Context()
    pub_socket = zmq_context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://127.0.0.1:{ZMQ_PUB_PORT}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    sock_gpu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_receiver = Receiver(CLIENT_PORT)
    
    print(f">>> Client Streamer & ZMQ Hub Started!")
    print(f">>> Streaming UDP video to Server: {SERVER_IP}:{SERVER_PORT}")
    print(f">>> Receiving server data on port: {CLIENT_PORT}")
    print(f">>> Publishing ZeroMQ stream on: tcp://127.0.0.1:{ZMQ_PUB_PORT}")
    print(">>> Ctrl+C to stop")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            # compress frame to jpeg & send to server
            frame_resized = cv2.resize(frame, RESOLUTION)
            _, frame_encoded = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            frame_bytes = frame_encoded.tobytes()

            if len(frame_bytes) < 65000:
                sock_gpu.sendto(frame_bytes, (SERVER_IP, SERVER_PORT))

            # get json message from server
            data = server_receiver.get_data()
            
            meta = {
                'joints': data.get('joints') if data else None,
                'mano_params': data.get('mano_params') if data else None,
                'timestamp': time.time()
            }

            # publish data to local zmq bus (metadata + jpeg)
            pub_socket.send_json(meta, flags=zmq.SNDMORE)
            pub_socket.send(frame_bytes)

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n>>> Streamer stopped by user.")
    finally:
        cap.release()
        sock_gpu.close()
        server_receiver.stop()
        pub_socket.close()
        zmq_context.term()

if __name__ == "__main__":
    start_streamer()