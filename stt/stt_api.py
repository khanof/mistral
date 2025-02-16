from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import whisper
import numpy as np
import io
import wave
import torch
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

print("Loading Whisper model...")
model = whisper.load_model("tiny")
print("Whisper model loaded successfully")

class AudioBuffer:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000
        self.min_duration = 1.0    # 1 second minimum
        self.max_duration = 12     # 12 seconds maximum
        self.is_processing = False
        self.processing_start_time = None
        self.processing_timeout = 10
        self.last_active_time = time.time()
        self.silence_threshold = 0.001
        self.silence_timeout = 1.5
        self.transcription_threshold = 0.002
        self.last_transcription = ""
        logger.info("Initialized AudioBuffer")

    def add_chunk(self, chunk):
        try:
            if not chunk or not isinstance(chunk, (list, np.ndarray)):
                return True

            # Add new chunk to buffer
            self.buffer.extend(chunk)
            
            # Keep only the last N seconds of audio
            max_samples = int(self.sample_rate * self.max_duration)
            if len(self.buffer) > max_samples:
                self.buffer = self.buffer[-max_samples:]
            
            # Check if we have enough data to process
            if len(self.buffer) >= int(self.sample_rate * self.min_duration) and not self.is_processing:
                return self.process_buffer()
                
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunk to buffer: {str(e)}")
            return True

    def process_buffer(self):
        try:
            self.is_processing = True
            audio_data = self.get_buffer()
            
            if audio_data is not None and len(audio_data) > 0:
                # Normalize the audio data
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Check if audio level is too low
                if np.max(np.abs(audio_data)) < self.transcription_threshold:
                    logger.info("Audio level too low for transcription")
                    self.reset()
                    return True

                # Transcribe
                result = model.transcribe(
                    audio_data,
                    language='en',
                    temperature=0.0,
                    best_of=5,
                    beam_size=5,
                    fp16=False,
                    compression_ratio_threshold=2.0,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    condition_on_previous_text=True,
                    prompt=self.last_transcription
                )
                
                transcribed_text = result['text'].strip()
                if transcribed_text and not transcribed_text.isspace():
                    logger.info(f"Transcribed: {transcribed_text}")
                    socketio.emit('transcription', {'text': transcribed_text})
                    self.last_transcription = transcribed_text
                    self.clear()
                else:
                    self.reset()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing buffer: {str(e)}")
            self.reset()
            return True
        finally:
            self.is_processing = False

    def get_buffer(self):
        if len(self.buffer) == 0:
            return None
            
        try:
            audio_data = np.array(self.buffer, dtype=np.float32)
            
            # Normalize the audio data
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
        except Exception as e:
            logger.error(f"Error converting buffer to numpy array: {e}")
            return None

    def clear(self):
        self.buffer = []
        self.reset()

    def reset(self):
        self.is_processing = False
        self.processing_start_time = None

audio_buffer = AudioBuffer()

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    audio_buffer.clear()

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")
    audio_buffer.clear()

@socketio.on('audio_stream')
def handle_audio_stream(data):
    try:
        if not audio_buffer.add_chunk(data):
            return

        if audio_buffer.has_enough_data() and not audio_buffer.is_processing:
            audio_buffer.start_processing()
            
            try:
                audio_data = audio_buffer.get_buffer()
                if audio_data is not None and len(audio_data) > 0:
                    # Transcribe with improved parameters
                    result = model.transcribe(
                        audio_data,
                        language='en',
                        temperature=0.0,
                        best_of=5,
                        beam_size=5,
                        fp16=False,
                        compression_ratio_threshold=2.0,
                        no_speech_threshold=0.6,
                        logprob_threshold=-1.0,
                        condition_on_previous_text=True,
                        prompt=audio_buffer.last_transcription  # Use last transcription as prompt
                    )
                    
                    transcribed_text = result['text'].strip()
                    if transcribed_text and not transcribed_text.isspace():
                        # Simple repetition check
                        if transcribed_text != audio_buffer.last_transcription:
                            logger.info(f"Transcribed: {transcribed_text}")
                            socketio.emit('transcription', {'text': transcribed_text})
                            audio_buffer.last_transcription = transcribed_text
                        audio_buffer.clear()
                    else:
                        audio_buffer.reset()
                        
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                audio_buffer.reset()
                
    except Exception as e:
        logger.error(f"Stream handling error: {str(e)}")
        audio_buffer.reset()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, allow_unsafe_werkzeug=True)
