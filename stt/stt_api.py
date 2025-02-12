from flask import Flask, request, jsonify
import whisper

app = Flask(__name__)

model = whisper.load_model("small")

@app.route('/stt', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "/app/input_audio.mp3"
    file.save(file_path)

    result = model.transcribe(file_path)
    return jsonify({"text": result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
