from gtts import gTTS
import os

def create_test_audio():
    # Create a test message
    text = "This is a test message for the speech to text service. Hello world!"
    
    # Create audio file
    tts = gTTS(text=text, lang='en')
    tts.save("test.mp3")
    
    # Convert to webm using ffmpeg
    os.system("ffmpeg -i test.mp3 -c:a libopus test.webm")
    
    print("Created test.webm file")

if __name__ == "__main__":
    create_test_audio()