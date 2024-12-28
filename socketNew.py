import socket
import struct
import threading
import os
import whisper

from message_producer import transcribe_video

# Directory to save received files
UPLOAD_DIR = "uploads"
TRANSCRIPTION_DIR = "transcriptions"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(TRANSCRIPTION_DIR):
    os.makedirs(TRANSCRIPTION_DIR)

# Load Whisper Model
print("Loading Whisper model...")
model = whisper.load_model("base")  # Change to "medium" or "large" for better accuracy
print("Whisper model loaded.")

def handle_client(client_socket, client_address):
    global file_path
    try:
        print(f"Connection established with {client_address}. Receiving metadata and file...")

        # Receive the lesson ID (8 bytes for a long)
        lesson_id_bytes = client_socket.recv(8)
        if len(lesson_id_bytes) < 8:
            print("Error: Failed to receive lesson ID.")
            return
        lesson_id = struct.unpack('>q', lesson_id_bytes)[0]
        print(f"Received Lesson ID: {lesson_id}")

        # Save the file with a unique name based on the lesson ID
        file_path = os.path.join(UPLOAD_DIR, f"lesson_{lesson_id}.mp4")
        with open(file_path, 'wb') as output_file:
            while True:
                chunk = client_socket.recv(1024)  # Receive 1 KB at a time
                if not chunk:
                    break
                output_file.write(chunk)

        print(f"File for Lesson ID {lesson_id} received successfully and saved to {file_path}.")

        # Transcribe the video
        transcribe_video(file_path, lesson_id)


    except Exception as e:
        print(f"Error handling client {client_address}: {e}")
    finally:
        os.remove(file_path)
        client_socket.close()
        print(f"Connection with {client_address} closed.")

def start_server(host='localhost', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(10)  # Allow up to 10 concurrent connections
    print(f"Server listening on {host}:{port}...")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            client_handler = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address)
            )
            client_handler.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()

