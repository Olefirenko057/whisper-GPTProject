import whisper
import ssl
from gpt4all import GPT4All

ssl._create_default_https_context = ssl.create_default_context()
try:
    model = whisper.load_model("base")
    result = model.transcribe("733ec6f5-f063-4df5-9e08-4687bf2cbe56.mp4")
    print(f' The text in video: \n {result["text"]}')

    data = result["text"]
    prompt = f'''Provide me point wise summary of text "{data}".'''

    model = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf", allow_download=True)

    output = model.generate(prompt)

    print(f' The summary of meeting : \n {output}')
except Exception as e:
    print(f'Error: {e}')

