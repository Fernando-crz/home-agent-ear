import os
from collections import deque
from enum import Enum

import numpy as np
import redis
import webrtcvad
from openwakeword.model import Model

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
CONSUMER_STREAM_NAME = os.environ.get("CONSUMER_STREAM_NAME", "live_audio_broadcast")
PRODUCER_STREAM_NAME = os.environ.get("PRODUCER_STREAM_NAME", "speech_pipeline")

VAD_MODE = int(os.environ.get("VAD_MODE", 2))
SILENCE_TIMEOUT_SECONDS = float(os.environ.get("SILENCE_TIMEOUT_SECONDS", 1.2))
WAKEWORD_THRESHOLD = float(os.environ.get("WAKEWORD_THRESHOLD", 0.5))

RATE = 16000
CHUNK = 1280
CHUNK_DURATION_SECONDS = CHUNK / RATE
SILENCE_FRAMES = round(SILENCE_TIMEOUT_SECONDS / CHUNK_DURATION_SECONDS)
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
EAR_PREBUFFER_CHUNK_SIZE = 15

def frames_from_chunk(chunk):
    for i in range(0, len(chunk), FRAME_SIZE * 2):
        yield chunk[i:i + FRAME_SIZE * 2]

class RedisProducer:
    def __init__(self, redis_provider, stream_name):
        self.redis_provider = redis_provider
        self.stream_name = stream_name

    def start(self):
        self.redis_provider.xadd(self.stream_name, {"event_type": "started"})

    def finish(self):
        self.redis_provider.xadd(self.stream_name, {"event_type": "finished"})

    def content(self, audio):
        full_audio_bytes = b"".join(audio)
        self.redis_provider.xadd(
            self.stream_name,
            {"event_type": "content", "audio_data": full_audio_bytes}
        )

class RedisConsumer:
    def __init__(self, redis_provider, stream_name):
        self.redis_provider = redis_provider
        self.stream_name = stream_name
        self.last_id = "$"

    def yield_chunks(self):
        response  = self.redis_provider.xread({self.stream_name: self.last_id}, block=2000)
        for _, message in response:
            for msg_id, payload in message:
                self.last_id = msg_id
                chunk = payload.get(b"audio_data")
                yield chunk

class HomeAgentEarState(str, Enum):
    LISTENING = "listening"
    WAKEWORD_DETECTED = "wakeword_detected"
    CAPTURING = "capturing"
    DONE_CAPTURING = "done_capturing"

class HomeAgentEar:
    def __init__(self, redis_consumer, redis_producer, vad_model, wakeword_model):
        self.recording = False
        self.silence_counter = 0
        self.prebuffer = deque(maxlen=EAR_PREBUFFER_CHUNK_SIZE)
        self.audio_to_save = []

        self.redis_consumer = redis_consumer
        self.redis_producer = redis_producer
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

        # Need to divide chunk into frames for VAD lib
        for frame in frames_from_chunk(chunk):
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

                self.redis_producer.start()

                self.audio_to_save = list(self.prebuffer)
                return HomeAgentEarState.WAKEWORD_DETECTED

            return HomeAgentEarState.LISTENING
        else:
            self.audio_to_save.append(chunk)

            self._update_silence_counter(chunk)

            if self.silence_counter <= SILENCE_FRAMES:
                return HomeAgentEarState.CAPTURING

            self.redis_producer.finish()
            self.redis_producer.content(self.audio_to_save)

            self._reset_states()

            return HomeAgentEarState.DONE_CAPTURING

    def run(self):
        try:
            while True:
                for chunk in self.redis_consumer.yield_chunks():
                    result = self.process_chunk(chunk)
                    if result == HomeAgentEarState.WAKEWORD_DETECTED:
                        print("Wakeword Detected!")
                    elif result == HomeAgentEarState.DONE_CAPTURING:
                        print("Done Capturing")

        except KeyboardInterrupt:
            print("\nStopping listener...")


def main():
    oww_model = Model()
    vad = webrtcvad.Vad(VAD_MODE)
    redis_provider = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
    redis_consumer = RedisConsumer(redis_provider, CONSUMER_STREAM_NAME)
    redis_producer = RedisProducer(redis_provider, PRODUCER_STREAM_NAME)
    home_agent_ear = HomeAgentEar(redis_consumer, redis_producer, vad, oww_model)

    home_agent_ear.run()

if __name__ == "__main__":
    main()
