import json
import os
import whisper
from gpt4all import GPT4All
import stomp


model = whisper.load_model("base")

def transcribe_video(file_path,lesson_id):
    print(f"Starting transcription for Lesson ID {lesson_id} with file {file_path}...")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    result = model.transcribe(file_path)
    summary = get_summary(result["text"])
    send_to_the_queue(summary,lesson_id)

def get_summary(result):
    prompt = "Provide a summary of this text: " + result
    model_gpt = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf", allow_download=True)
    result = model_gpt.generate(prompt)
    return result

def send_to_the_queue(summary,lesson_id):
    print(f"Sending summary: {summary}")
    broker_host = 'localhost'
    broker_port = 61613
    data = {"summary": summary, "lessonId": lesson_id}
    conn = stomp.Connection([(broker_host,broker_port)])
    conn.connect()
    conn.send('/queue/Queue.example',json.dumps(data))
    print("message sent")
    conn.disconnect()

