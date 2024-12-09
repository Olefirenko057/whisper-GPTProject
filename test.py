import os
import tempfile
from gpt4all import GPT4All
from whisper import load_model
from flask import Flask, jsonify, request

app = Flask(__name__)

model = load_model("base")

# Allowed MIME types for video/audio files
ALLOWED_MIME_TYPES = {"video/mp4", "audio/mpeg", "video/quicktime"}  # mp4, mp3, mov

@app.route("/upload-video/", methods = ['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400
    # Extract video file from request
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if file and file.mimetype in ALLOWED_MIME_TYPES :
        # Save the file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            file.save(temp_video.name)
            temp_video_path = temp_video.name

        try:
            # Transcribe the file using Whisper
            transcription = transcribe_file(temp_video_path)
            # Get the summary of the text through GPT4All
            result = get_summary(transcription)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temporary file
            os.remove(temp_video_path)

        return result, 200
    else:
        return jsonify({"error": "Invalid file type. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"}), 400


def transcribe_file(file_path):
    # Load the model and transcribe the video
    result = model.transcribe(file_path)
    return result['text']

def get_summary(text):
    print(text)
    prompt = "Provide a brief summary of the text: " + text + ". Format should be: you begin with here is the summary (on the first line): and then below just write your brief but informative summary of the given text."
    model_gpt = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf", allow_download=True)
    output = model_gpt.generate(prompt)
    return output

if __name__ == '__main__':
    app.run(debug=True)
