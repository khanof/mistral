from celery import Celery
from whisper_worker import transcriber

# Initialize Celery
celery_app = Celery('tasks',
                    broker='redis://redis:6379/0',
                    backend='redis://redis:6379/0')

@celery_app.task
def transcribe_audio(audio_data):
    return transcriber.transcribe_audio(audio_data) 