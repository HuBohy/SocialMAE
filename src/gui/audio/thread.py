import threading

from src.gui.audio.stream import AudioStream
from src.gui.audio.processing import AudioProcessor

class AudioThread(threading.Thread, AudioStream, AudioProcessor):
    def __init__(self, input_mode: str = "video", audio_rate: int = 16000, audio_duration: float = 0.27, audio_chunk: int = 1024, n_mels: int = 128):
        AudioStream.__init__(self, audio_rate, audio_duration, audio_chunk)
        AudioProcessor.__init__(self, audio_rate, audio_chunk, n_mels)
        self.is_running = False
        self.input_mode = input_mode

        super().__init__(target=self._record_audio, daemon=True)

    def start_thread(self):
        if not self.is_running:
            self.start_stream()
            
    def stop_thread(self):
        if self.is_running and self.is_alive():
            self.stop_stream()

    def get_audio_tensor(self):
        """Process audio from the buffer"""
        if (len(self.audio_buffer) < self.audio_rate * 0.1) or (self.input_mode == "webcam" and not self.is_running):  # Less than 0.1 seconds
            yield None
        yield self.get_audio_features(self.audio_buffer)
