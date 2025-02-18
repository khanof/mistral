import whisper
import numpy as np
import tempfile
import logging
from pydub import AudioSegment
import io
import time
import psutil
import os

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_name="tiny"):
        logger.warning("Starting to load Whisper model...")
        logger.warning(f"Available memory before model load: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
        self.model = whisper.load_model(model_name)
        logger.warning("Whisper model loaded successfully!")
        logger.warning(f"Available memory after model load: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    
    def transcribe_audio(self, audio_data):
        try:
            pid = os.getpid()
            process = psutil.Process(pid)
            logger.warning(f"Memory usage before processing: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            logger.warning(f"Starting to process audio data of size: {len(audio_data)} bytes")
            
            # Convert WebM to WAV using pydub
            logger.warning("Converting WebM to AudioSegment...")
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
            logger.warning(f"Audio converted: duration={len(audio)}ms, channels={audio.channels}, sample_width={audio.sample_width}")
            
            # Export as WAV to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                logger.warning("Exporting to WAV format...")
                audio.export(temp_wav.name, format='wav')
                logger.warning("WAV export completed")
                
                # Convert to numpy array
                logger.warning("Converting to numpy array...")
                samples = np.array(audio.get_array_of_samples())
                logger.warning(f"Numpy array shape: {samples.shape}, dtype: {samples.dtype}")
                
                # Convert to float32 and normalize
                samples = samples.astype(np.float32) / 32768.0
                logger.warning("Starting Whisper transcription...")
                logger.warning(f"Memory usage before transcribe: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                
                try:
                    logger.warning("Just before model.transcribe()...")
                    # Add timeout to transcribe
                    start_time = time.time()
                    result = self.model.transcribe(samples, language='en')  # Specify language
                    end_time = time.time()
                    logger.warning("Immediately after model.transcribe()")
                    logger.warning(f"Transcription took {end_time - start_time:.2f} seconds")
                    logger.warning(f"Memory usage after transcribe: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    logger.warning(f"RAW RESULT FROM WHISPER: {result}")
                    
                    response = {
                        "text": result.get("text", "No text found"),
                        "segments": result.get("segments", [])
                    }
                    logger.warning(f"Returning response: {response}")
                    return response
                    
                except Exception as transcribe_error:
                    logger.error(f"Error during model.transcribe(): {str(transcribe_error)}")
                    logger.exception("Transcription error traceback:")
                    raise
                
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            logger.exception("Full traceback:")
            raise

# Create singleton instance
logger.warning("Initializing WhisperTranscriber...")
transcriber = WhisperTranscriber()
logger.warning("WhisperTranscriber initialized!") 