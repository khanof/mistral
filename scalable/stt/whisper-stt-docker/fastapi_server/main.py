from fastapi import FastAPI, WebSocket
from celery_tasks import transcribe_audio
import asyncio
import base64
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data as base64 string
            data = await websocket.receive_text()
            audio_data = base64.b64decode(data)
            
            # Submit transcription task to Celery
            task = transcribe_audio.delay(audio_data)
            
            # Wait for result and send back to client
            while not task.ready():
                await asyncio.sleep(0.1)
            
            result = task.get()
            await websocket.send_json({
                "text": result["text"],
                "segments": result["segments"]
            })
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 