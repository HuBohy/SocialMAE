import numpy as np
import pyaudio
from collections import deque

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioStream:
    def __init__(self, audio_rate=16000, audio_duration=0.27, audio_chunk=1024):
        # Audio recording setup
        self.audio_format = pyaudio.paFloat32
        self.audio_channels = 1
        self.audio_rate = audio_rate  # Sample rate
        self.audio_chunk = audio_chunk  # Buffer size
        self.pyaudio_instance = None
        self.audio_stream = None
        self.audio_duration = audio_duration
        self.audio_buffer = deque(maxlen=int(self.audio_rate * self.audio_duration))  # 0.27 seconds of audio

    def start_stream(self):
        """Initialize audio recording"""
        try:
            self.stop_stream()  # Ensure any existing stream is stopped
            logger.debug("Setting up audio recording")
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_rate,
                input=True,
                frames_per_buffer=self.audio_chunk
            )
            logger.debug("Audio recording initialized")
            self.is_running = True
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
            self.is_running = False

    def stop_stream(self):
        """Stop audio recording"""
        try:
            self.is_running = False

            if self.audio_stream:
                logger.debug("Stopping existing audio stream")
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None            
            if self.pyaudio_instance:
                logger.debug("Terminating existing PyAudio instance")
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            logger.debug("Clearing audio buffer")
            self.audio_buffer.clear()
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
            self.is_running = False

    def _record_audio(self):
        """Record audio in a separate thread"""
        while self.is_running and self.audio_stream:
            try:
                data = self.audio_stream.read(self.audio_chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.audio_buffer.extend(audio_data)
                
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                break