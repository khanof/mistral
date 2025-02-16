from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
import whisper
import numpy as np
import io
import wave
import torch
import time
import logging
from scipy import signal  # Add this import

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration flags
ENABLE_NOISE_CANCELLATION = False  # Disable by default to test raw audio first
ENABLE_SILENCE_DETECTION = True

def apply_noise_reduction(audio_data, sample_rate=16000):
    """
    Apply simple noise reduction using spectral subtraction
    """
    try:
        logger.debug("Applying noise reduction...")
        
        # Ensure audio data is the right shape and type
        audio_data = np.array(audio_data, dtype=np.float32)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
            
        # Parameters for STFT
        nperseg = 256  # Window size
        noverlap = 128  # Overlap between windows
        
        # Compute STFT
        f, t, Zxx = signal.stft(audio_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Compute magnitude spectrogram
        mag = np.abs(Zxx)
        
        # Estimate noise profile from the first few frames
        noise_profile = np.mean(mag[:, :3], axis=1)
        
        # Apply spectral subtraction
        mag_cleaned = np.maximum(mag - noise_profile[:, np.newaxis], 0)
        
        # Reconstruct complex spectrogram
        Zxx_cleaned = mag_cleaned * np.exp(1j * np.angle(Zxx))
        
        # Inverse STFT
        _, audio_cleaned = signal.istft(Zxx_cleaned, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Ensure same length as input
        audio_cleaned = audio_cleaned[:len(audio_data)]
        
        # Normalize
        if np.max(np.abs(audio_cleaned)) > 0:
            audio_cleaned = audio_cleaned / np.max(np.abs(audio_cleaned))
        
        logger.info(f"Noise reduction applied successfully. Input shape: {audio_data.shape}, Output shape: {audio_cleaned.shape}")
        return audio_cleaned
        
    except Exception as e:
        logger.error(f"Error in noise reduction: {str(e)}")
        logger.error(f"Input audio shape: {audio_data.shape if isinstance(audio_data, np.ndarray) else 'not numpy array'}")
        return audio_data  # Return original audio if noise reduction fails

print("Loading Whisper model...")
model = whisper.load_model("base")  # or "small" for even better results
print("Whisper model loaded successfully")

class AudioBuffer:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000
        self.min_duration = 0.5    # Reduced to catch shorter phrases
        self.max_duration = 8      # Increased to handle longer phrases
        self.is_processing = False
        self.processing_start_time = None
        self.processing_timeout = 10  # Increased timeout for longer processing
        self.last_active_time = time.time()
        self.silence_threshold = 0.001
        self.silence_timeout = 5.0
        
        # Keep the lower threshold for better speech detection
        self.transcription_threshold = 0.002
        logger.info("Initialized AudioBuffer")
        logger.info(f"Silence detection is {'enabled' if ENABLE_SILENCE_DETECTION else 'disabled'}")

    def add_chunk(self, chunk):
        try:
            if ENABLE_SILENCE_DETECTION:
                chunk_array = np.array(chunk, dtype=np.float32)
                max_amplitude = np.max(np.abs(chunk_array))
                
                if max_amplitude > self.silence_threshold:
                    self.last_active_time = time.time()
                else:
                    silence_duration = time.time() - self.last_active_time
                    
                    if silence_duration >= self.silence_timeout:
                        logger.info(f"Extended silence detected: {silence_duration:.2f}s of silence")
                        socketio.emit('silence_timeout', {'message': 'Recording stopped due to silence'})
                        self.clear()
                        disconnect()
                        return False

            if self.is_processing and self.processing_start_time:
                if time.time() - self.processing_start_time > self.processing_timeout:
                    logger.warning("Processing timeout reached, resetting state")
                    self.reset()
                    return True

            if self.is_processing:
                return True

            if not chunk or not isinstance(chunk, (list, np.ndarray)):
                return True

            # Add new chunk to buffer
            self.buffer.extend(chunk)
            
            # Calculate buffer duration in seconds
            current_duration = len(self.buffer) / self.sample_rate
            
            # Only trim if we're significantly over max_duration
            if current_duration > (self.max_duration + 1):
                samples_to_keep = int(self.max_duration * self.sample_rate)
                self.buffer = self.buffer[-samples_to_keep:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunk to buffer: {str(e)}")
            return True

    def get_buffer(self):
        if len(self.buffer) == 0:
            return None
            
        try:
            audio_data = np.array(self.buffer, dtype=np.float32)
            
            # Normalize the audio data
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Add debug logging for audio levels
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(np.square(audio_data)))
            logger.info(f"Audio stats - Max amplitude: {max_amplitude:.6f}, RMS: {rms:.6f}, Threshold: {self.transcription_threshold}")
            
            # Check if the audio has enough amplitude to be valid speech
            if max_amplitude < self.transcription_threshold:
                logger.info(f"Audio level too low for transcription (max amplitude: {max_amplitude:.6f})")
                return None
                
            if ENABLE_NOISE_CANCELLATION:
                audio_data = apply_noise_reduction(audio_data, self.sample_rate)
                
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
        if not audio_buffer.add_chunk(data):
            return

        if audio_buffer.has_enough_data() and not audio_buffer.is_processing:
            audio_buffer.start_processing()
            
            try:
                audio_data = audio_buffer.get_buffer()
                if audio_data is not None and len(audio_data) > 0:
                    # Add stronger validation of audio data
                    rms = np.sqrt(np.mean(np.square(audio_data)))
                    if rms < audio_buffer.transcription_threshold:
                        logger.info("Audio signal too weak, skipping transcription")
                        audio_buffer.clear()
                        return

                    # Removed initial_prompt parameter
                    result = model.transcribe(
                        audio_data,
                        language='en',
                        temperature=0.0,
                        beam_size=5,
                        best_of=5,
                        condition_on_previous_text=True
                    )
                    
                    transcribed_text = result['text'].strip()
                    if transcribed_text and not transcribed_text.isspace():
                        # Add basic validation of transcribed text
                        if not transcribed_text.count("I'm sorry") > 2:  # Avoid repetitive text
                            logger.info(f"Transcribed: {transcribed_text}")
                            socketio.emit('transcription', {'text': transcribed_text})
                        else:
                            logger.info("Rejected repetitive transcription")
                        audio_buffer.clear()
                    else:
                        audio_buffer.reset()
                        
            except Exception as e:
                logger.error(f"Transcription error: {str(e)}")
                audio_buffer.reset()
                
    except Exception as e:
        logger.error(f"Stream handling error: {str(e)}")
        audio_buffer.reset()

@socketio.on('config')
def handle_config(data):
    """Handle configuration updates from client"""
    global ENABLE_NOISE_CANCELLATION, ENABLE_SILENCE_DETECTION
    try:
        if 'noise_cancellation' in data:
            ENABLE_NOISE_CANCELLATION = bool(data['noise_cancellation'])
            logger.info(f"Noise cancellation {'enabled' if ENABLE_NOISE_CANCELLATION else 'disabled'}")
            
        if 'silence_detection' in data:
            ENABLE_SILENCE_DETECTION = bool(data['silence_detection'])
            logger.info(f"Silence detection {'enabled' if ENABLE_SILENCE_DETECTION else 'disabled'}")
            
        socketio.emit('config_update', {
            'noise_cancellation': ENABLE_NOISE_CANCELLATION,
            'silence_detection': ENABLE_SILENCE_DETECTION
        })
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        socketio.emit('error', {'error': str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5002, allow_unsafe_werkzeug=True)
