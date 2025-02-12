from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import json
import logging
import random

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MISTRAL_URL = "http://mistral-api:8001/generate"  # Docker service name
TTS_URL = "http://tts-api:5001/tts"              # Docker service name
WHISPER_URL = "http://stt-api:5002/stt"      # Docker service name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stt', methods=['POST'])
def speech_to_text():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        logger.info("Received audio file, sending to Whisper API")
        
        # Send to Whisper API
        files = {'file': ('audio.wav', audio_file.read(), 'audio/wav')}
        whisper_response = requests.post(WHISPER_URL, files=files)

        if whisper_response.status_code != 200:
            logger.error(f"Whisper API error: {whisper_response.text}")
            return jsonify({"error": "Whisper API error"}), 500

        text = whisper_response.json().get('text', '')
        logger.info(f"Received transcription from Whisper API: {text[:100]}...")
        
        return jsonify({"text": text})

    except Exception as e:
        logger.error(f"Error processing STT request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_text():
    try:
        # Get input from request
        data = request.json
        prompt = data.get('prompt', '')
        voice = data.get('voice', 'en-US-JennyNeural')
        
        logger.info(f"Received request with prompt length: {len(prompt)} and voice: {voice}")

        if not prompt:
            logger.warning("No prompt provided in request")
            return jsonify({"error": "No prompt provided"}), 400

        # Call Mistral API
        logger.info("Calling Mistral API...")
        mistral_payload = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        mistral_response = requests.post(MISTRAL_URL, json=mistral_payload)
        mistral_response_json = mistral_response.json()
        generated_text = mistral_response_json['text'][0]
        if mistral_response.status_code != 200:
            logger.error(f"Mistral API error: {mistral_response.text}")
            return jsonify({"error": "Mistral API error", "details": mistral_response.text}), 500

        
        logger.info(f"Received response from Mistral API, length: {len(generated_text)}")

        # Call TTS API
        logger.info("Calling TTS API...")
        tts_payload = {
            "text": generated_text,
            "voice": voice
        }
        
        tts_response = requests.post(TTS_URL, json=tts_payload)
        if tts_response.status_code != 200:
            logger.error(f"TTS API error: {tts_response.text}")
            return jsonify({"error": "TTS API error", "details": tts_response.text}), 500

        logger.info("Successfully processed request")
        # Return both the generated text and audio
        return tts_response.content, 200, {
            'Content-Type': 'audio/mpeg',
            'Generated-Text': generated_text.replace('\n', ' ').replace('\r', '')
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/wait/<filename>')
def serve_wait_file(filename):
    return send_from_directory('wait', filename)

@app.route('/text-to-speech')
def text_to_speech_page():
    return render_template('text-to-speech.html')

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    try:
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'en-US-JennyNeural')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        logger.info(f"Generating speech for text length: {len(text)}")
        
        # Call TTS API
        tts_payload = {
            "text": text,
            "voice": voice
        }
        
        tts_response = requests.post(TTS_URL, json=tts_payload)
        if tts_response.status_code != 200:
            logger.error(f"TTS API error: {tts_response.text}")
            return jsonify({"error": "TTS API error"}), 500

        return tts_response.content, 200, {
            'Content-Type': 'audio/mpeg',
            'Content-Disposition': 'attachment; filename=generated_speech.mp3'
        }

    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/mimic/<filename>')
def serve_mimic_file(filename):
    return send_from_directory('mimic', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000) 