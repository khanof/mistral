from celery import Celery
from whisper_worker import transcriber
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Celery with longer timeouts
celery = Celery('tasks', 
                broker='redis://redis:6379/0',
                backend='redis://redis:6379/0')

# Configure Celery
celery.conf.update(
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True
)

@celery.task(bind=True)
def transcribe_audio(self, audio_data):
    try:
        logger.warning(f"[Task {self.request.id}] Starting transcription of {len(audio_data)} bytes")
        result = transcriber.transcribe_audio(audio_data)
        logger.warning(f"[Task {self.request.id}] Got result from transcriber: {result}")
        logger.warning(f"[Task {self.request.id}] Transcription completed successfully")
        return result
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Transcription error: {str(e)}")
        logger.exception("Full traceback:")
        raise 