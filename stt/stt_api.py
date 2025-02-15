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
        self.min_duration = 1  # 1 second minimum
        self.max_duration = 3  # 3 seconds maximum
        self.is_processing = False
        self.processing_start_time = None
        self.processing_timeout = 5  # 5 seconds timeout
        logger.info(f"Initialized AudioBuffer with sample rate: {self.sample_rate}")

    def add_chunk(self, chunk):
        try:
            # Check for processing timeout
            if self.is_processing and self.processing_start_time:
                if time.time() - self.processing_start_time > self.processing_timeout:
                    logger.warning("Processing timeout reached, resetting state")
                    self.reset()
                    return

            # Skip if we're currently processing
            if self.is_processing:
                return

            # Validate incoming chunk
            if not chunk or not isinstance(chunk, (list, np.ndarray)):
                return

            # Add the chunk to our buffer
            self.buffer.extend(chunk)
            
            # Keep only the last N seconds of audio
            max_samples = self.sample_rate * self.max_duration
            if len(self.buffer) > max_samples:
                excess = len(self.buffer) - max_samples
                self.buffer = self.buffer[excess:]
            
            logger.debug(f"Buffer size after adding chunk: {len(self.buffer)} samples")
            
        except Exception as e:
            logger.error(f"Error adding chunk to buffer: {e}")

    def get_buffer(self):
        if len(self.buffer) == 0:
            return None
            
        try:
            audio_data = np.array(self.buffer, dtype=np.float32)
            return audio_data
        except Exception as e:
            logger.error(f"Error converting buffer to numpy array: {e}")
            return None

    def clear(self):
        buffer_size = len(self.buffer)
        self.buffer = []
        self.reset()
        logger.debug(f"Cleared buffer of size {buffer_size}")

    def reset(self):
        self.is_processing = False
        self.processing_start_time = None

    def has_enough_data(self):
        min_samples = self.sample_rate * self.min_duration
        return len(self.buffer) >= min_samples

    def start_processing(self):
        self.is_processing = True
        self.processing_start_time = time.time()

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
        # Add the chunk to the buffer
        audio_buffer.add_chunk(data)
        
        # Only process if we have enough data and aren't already processing
        if audio_buffer.has_enough_data() and not audio_buffer.is_processing:
            audio_buffer.start_processing()
            
            try:
                audio_data = audio_buffer.get_buffer()
                if audio_data is not None and len(audio_data) > 0:
                    # Normalize audio between -1 and 1
                    audio_data = np.nan_to_num(audio_data)
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    
                    logger.info(f"Processing audio: shape={audio_data.shape}, range=[{np.min(audio_data):.2f}, {np.max(audio_data):.2f}]")
                    
                    # Transcribe
                    result = model.transcribe(
                        audio_data,
                        language='en',
                        temperature=0.0
                    )
                    
                    transcribed_text = result['text'].strip()
                    if transcribed_text:
                        logger.info(f"Transcribed: {transcribed_text}")
                        socketio.emit('transcription', {'text': transcribed_text})
                        audio_buffer.clear()  # Clear buffer after successful transcription
                    else:
                        logger.warning("No text transcribed")
                        audio_buffer.reset()  # Just reset processing state, keep buffer
                        
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                socketio.emit('error', {'error': str(e)})
                audio_buffer.reset()
                
    except Exception as e:
        logger.error(f"Stream handling error: {str(e)}")
        socketio.emit('error', {'error': str(e)})
        audio_buffer.reset()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, allow_unsafe_werkzeug=True)
