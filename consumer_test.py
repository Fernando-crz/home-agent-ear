import time
import wave

import redis

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
STREAM_NAME = "speech_pipeline"


RATE = 16000
N_CHANNELS = 1
SAMPLE_WIDTH = 2

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

def save_wav(audio_bytes, filename):
    """Reconstructs the bytes back into a playable WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(N_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    print(f" Saved file successfully: {filename}")

print("Processor active. Awaiting local Redis stream messages...")

last_id = "$"

try:
    while True:
        response = r.xread({STREAM_NAME: last_id}, block=0)

        for stream, messages in response:
            for msg_id, data in messages:
                last_id = msg_id

                event_type = data.get(b"event_type", b"").decode("utf-8")

                if event_type == "started":
                    print("\n[Redis Event]: 🟢 Speech capture started (wakeword detected).")

                elif event_type == "finished":
                    print("[Redis Event]: 🔴 Speech capture is finished.")

                elif event_type == "content":
                    print("[Redis Event]: 📦 Speech content received (full audio payload downloaded).")

                    audio_bytes = data.get(b"audio_data")
                    if audio_bytes:
                        filename = f"received_speech_{int(time.time())}.wav"
                        save_wav(audio_bytes, filename)

except KeyboardInterrupt:
    print("\nStopping processor...")
