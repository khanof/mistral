from flask import Flask, request, send_file
import asyncio
import edge_tts
import os

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get("text", "Hello, this is an AI-generated voice.")
    voice = data.get("voice", "en-US-JennyNeural")
    output_file = "/app/output.mp3"

    async def generate_speech():
        tts = edge_tts.Communicate(text, voice)
        await tts.save(output_file)

    asyncio.run(generate_speech())

    # Check if the file exists before returning it
    if os.path.exists(output_file):
        return send_file(output_file, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")
    else:
        return {"error": "TTS generation failed"}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
