from fastapi import FastAPI
import asyncio
import websockets
import json
from gtts import gTTS
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

async def test_stt_with_audio():
    # Create test audio
    text = "This is a test message for the speech to text service. Hello world!"
    tts = gTTS(text=text, lang='en')
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
        logger.info("Creating MP3 file...")
        tts.save(mp3_file.name)
        # Convert to webm
        webm_path = mp3_file.name.replace('.mp3', '.webm')
        logger.info("Converting to WebM...")
        os.system(f"ffmpeg -i {mp3_file.name} -c:a libopus {webm_path}")
    
    # Connect to WebSocket server
    uri = "ws://nginx:80/ws"
    logger.info(f"Connecting to WebSocket at {uri}")
    
    try:
        async with websockets.connect(uri, ping_interval=None) as websocket:
            logger.info("WebSocket connected")
            
            # Read and send the audio file
            with open(webm_path, "rb") as audio_file:
                audio_data = audio_file.read()
                logger.info(f"Sending audio data of size: {len(audio_data)} bytes")
                await websocket.send(audio_data)
            
            logger.info("Waiting for response...")
            response = await websocket.recv()
            result = json.loads(response)
            
            # Clean up temporary files
            os.remove(mp3_file.name)
            os.remove(webm_path)
            
            return {
                "status": "success",
                "original_text": text,
                "transcribed_text": result['text']
            }
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "details": "Make sure the main STT service is running and accessible"
        }

@app.get("/")
async def root():
    return {"message": "Test API is running"}

@app.post("/run-test")
async def run_test():
    logger.info("Starting test run...")
    result = await test_stt_with_audio()
    logger.info(f"Test completed with result: {result}")
    return result 