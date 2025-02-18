import whisper
import numpy as np
import io

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)
    
    def transcribe_audio(self, audio_data):
        # Convert audio data to format expected by Whisper
        audio_np = np.frombuffer(audio_data, np.float32)
        
        # Perform transcription
        result = self.model.transcribe(audio_np)
        
        return {
            "text": result["text"],
            "segments": result["segments"]
        }

# Create singleton instance
transcriber = WhisperTranscriber() 