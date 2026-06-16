from collections import deque
from enum import Enum

import numpy as np
import openwakeword
import pyaudio
import redis
import webrtcvad
from openwakeword.model import Model

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
STREAM_NAME = "speech_pipeline"

RATE = 16000
CHUNK = 1280
VAD_MODE = 2
SILENCE_FRAMES = 15
WAKEWORD_THRESHOLD = 0.5
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)

openwakeword.utils.download_models()

def frames_from_chunk(chunk):
    for i in range(0, len(chunk), FRAME_SIZE * 2):
        yield chunk[i:i + FRAME_SIZE * 2]

class RedisBroadcaster:
    def __init__(self, host, port, stream_name):
        self.redis = redis.Redis(host=host, port=port)
        self.stream_name = stream_name

    def start(self):
        self.redis.xadd(self.stream_name, {"event_type": "started"})

    def finish(self):
        self.redis.xadd(self.stream_name, {"event_type": "finished"})

    def content(self, audio):
        full_audio_bytes = b"".join(audio)
        self.redis.xadd(self.stream_name, {"event_type": "content", "audio_data": full_audio_bytes})


class HomeAgentEarState(str, Enum):
    LISTENING = "listening"
    WAKEWORD_DETECTED = "wakeword_detected"
    CAPTURING = "capturing"
    DONE_CAPTURING = "done_capturing"

class HomeAgentEar:
    def __init__(self, pyaudio_instance, redis_broadcaster, vad_model, wakeword_model):
        self.recording = False
        self.silence_counter = 0
        self.prebuffer = deque(maxlen=20)
        self.audio_to_save = []

        self.pyaudio_instance = pyaudio_instance
        self.stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.redis_broadcaster = redis_broadcaster
        self.vad_model = vad_model
        self.wakeword_model = wakeword_model

    def _is_wakeword_detected(self, audio):
        prediction = self.wakeword_model.predict(audio)
        for _, score in prediction.items():
            if score > WAKEWORD_THRESHOLD:
                    return True

        return False

    def _update_silence_counter(self, chunk):
        has_speech = False

        for frame in frames_from_chunk(chunk): # Need to divide chunk into frames for VAD lib
            if len(frame) < FRAME_SIZE * 2:
                continue
            if self.vad_model.is_speech(frame, RATE):
                has_speech = True
                break

        if has_speech:
            self.silence_counter = 0
        else:
            self.silence_counter += 1

    def _reset_states(self):
        self.recording = False
        self.silence_counter = 0
        self.audio_to_save = []
        self.wakeword_model.reset()

    def process_chunk(self, chunk):
        audio = np.frombuffer(chunk, dtype=np.int16)

        if not self.recording:
            self.prebuffer.append(chunk)

            if self._is_wakeword_detected(audio):
                self.recording = True
                self.silence_counter = 0

                self.redis_broadcaster.start()

                self.audio_to_save = list(self.prebuffer)
                return HomeAgentEarState.WAKEWORD_DETECTED

            return HomeAgentEarState.LISTENING
        else:
            self.audio_to_save.append(chunk)

            self._update_silence_counter(chunk)

            if self.silence_counter <= SILENCE_FRAMES:
                return HomeAgentEarState.CAPTURING

            self.redis_broadcaster.finish()
            self.redis_broadcaster.content(self.audio_to_save)

            self._reset_states()

            return HomeAgentEarState.DONE_CAPTURING

    def run(self):
        try:
            while True:
                chunk = self.stream.read(CHUNK, exception_on_overflow=False)
                result = self.process_chunk(chunk)
                if result == HomeAgentEarState.WAKEWORD_DETECTED:
                    print("Wakeword Detected!")
                elif result == HomeAgentEarState.DONE_CAPTURING:
                    print("Done Capturing")

        except KeyboardInterrupt:
            print("\nStopping listener...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()


def main():
    pyaudio_instance = pyaudio.PyAudio()
    oww_model = Model()
    vad = webrtcvad.Vad(VAD_MODE)
    redis_broadcaster = RedisBroadcaster(REDIS_HOST, REDIS_PORT, STREAM_NAME)
    home_agent_ear = HomeAgentEar(pyaudio_instance, redis_broadcaster, vad, oww_model)

    home_agent_ear.run()

if __name__ == "__main__":
    main()
