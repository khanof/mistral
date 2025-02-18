from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from celery_tasks import transcribe_audio
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.get("/test")
async def test():
    return {"message": "Server is running!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        audio_chunks = []
        while True:
            try:
                # Receive audio data
                data = await websocket.receive()
                
                if data.get("type") == "websocket.disconnect":
                    break
                    
                if "bytes" in data:
                    logger.info(f"Received audio chunk of size: {len(data['bytes'])} bytes")
                    audio_chunks.append(data['bytes'])
                    
                    # Process complete audio when we receive enough data
                    if len(audio_chunks) > 0:
                        # Combine all chunks
                        complete_audio = b''.join(audio_chunks)
                        logger.info(f"Processing complete audio of size: {len(complete_audio)} bytes")
                        
                        # Submit to Celery
                        task = transcribe_audio.delay(complete_audio)
                        logger.info("Transcription task submitted")
                        
                        # Wait for result with much longer timeout
                        start_time = asyncio.get_event_loop().time()
                        timeout_seconds = 120  # Increase to 120 seconds

                        while not task.ready():
                            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                                logger.error(f"Task {task.id} timed out after {timeout_seconds} seconds")
                                await websocket.send_json({
                                    "status": "error",
                                    "error": f"Transcription timed out after {timeout_seconds} seconds"
                                })
                                return
                            
                            # Log progress every 10 seconds
                            if int(asyncio.get_event_loop().time() - start_time) % 10 == 0:
                                logger.info(f"Still waiting for task {task.id} after {int(asyncio.get_event_loop().time() - start_time)} seconds")
                            
                            await asyncio.sleep(1)  # Changed from 0.1 to reduce CPU usage

                        try:
                            result = task.get(timeout=30)
                            logger.warning(f"Got transcription result from Celery: {result}")
                            
                            if result and "text" in result:
                                response = {
                                    "status": "success",
                                    "text": result["text"],
                                    "segments": result.get("segments", [])
                                }
                                logger.warning(f"Sending WebSocket response: {response}")
                                await websocket.send_json(response)
                                logger.warning("WebSocket response sent successfully")
                            else:
                                logger.error(f"Invalid result format: {result}")
                                await websocket.send_json({
                                    "status": "error",
                                    "error": "Invalid transcription result format"
                                })
                            
                        except Exception as e:
                            logger.error(f"Error getting task result: {str(e)}")
                            await websocket.send_json({
                                "status": "error",
                                "error": f"Error getting transcription result: {str(e)}"
                            })
                        
                        # Clear chunks for next audio
                        audio_chunks = []
                
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 