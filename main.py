from collections import deque
import numpy as np
import openwakeword
from openwakeword.model import Model
import pyaudio
import webrtcvad
import redis

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
STREAM_NAME = "speech_pipeline"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

RATE = 16000
CHUNK = 1280 
VAD_MODE = 2
SILENCE_FRAMES = 20
WAKEWORD_THRESHOLD = 0.5
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)

openwakeword.utils.download_models()
oww_model = Model()
vad = webrtcvad.Vad(VAD_MODE)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

def frames_from_chunk(chunk):
    for i in range(0, len(chunk), FRAME_SIZE * 2):
        yield chunk[i:i + FRAME_SIZE * 2]

print("Listening for wakeword... Broadcasting to local Redis stream.")

recording = False
silence_counter = 0
prebuffer = deque(maxlen=20)
audio_to_save = []

try:
    while True:
        chunk = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(chunk, dtype=np.int16)

        if not recording:
            prebuffer.append(chunk)
            prediction = oww_model.predict(audio_data)
            
            for model_name, score in prediction.items():
                if score > WAKEWORD_THRESHOLD:
                    print(f"\n[!] Wakeword detected!")
                    print(f"Model detected: {model_name}")
                    recording = True
                    silence_counter = 0
                    
                    # 1. Message: Speech capture started
                    r.xadd(STREAM_NAME, {"event_type": "started"})
                    
                    audio_to_save = list(prebuffer)
                    break
        else:
            audio_to_save.append(chunk)

            for frame in frames_from_chunk(chunk):
                if len(frame) < FRAME_SIZE * 2: continue
                if vad.is_speech(frame, RATE):
                    silence_counter = 0
                else:
                    silence_counter += 1

            if silence_counter <= SILENCE_FRAMES:
                continue
            
            print("--> Speech ended.")
            
            # 2. Message: Speech capture finished
            r.xadd(STREAM_NAME, {"event_type": "finished"})
            
            # 3. Message: Full raw audio payload
            full_audio_bytes = b"".join(audio_to_save)
            r.xadd(STREAM_NAME, {"event_type": "content", "audio_data": full_audio_bytes})
            
            # Reset states
            recording = False
            silence_counter = 0
            audio_to_save = []
            oww_model.reset()

except KeyboardInterrupt:
    print("\nStopping listener...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()