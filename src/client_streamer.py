import cv2
import socket

SERVER_IP = "10.144.208.248"
SERVER_PORT = 49102
RESOLUTION = (256, 256)

def start_sender():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f">>> Streaming UDP to {SERVER_IP}:{SERVER_PORT}...")
    print(">>> Ctrl+C to stop")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # resizing and compression
            frame_resized = cv2.resize(frame, RESOLUTION)
            _, frame_encoded = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            data = frame_encoded.tobytes()
            
            # frames demo
            cv2.imshow('Sending to HaMeR', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(data) > 65000:
                continue

            sock.sendto(data, (SERVER_IP, SERVER_PORT))

    except KeyboardInterrupt:
        print(">>> Stopped.")
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    start_sender()