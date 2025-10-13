import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.gui.video.processing import VideoProcessing
from src.gui.video.stream import VideoStream

class VideoThread(VideoStream, VideoProcessing):
    def __init__(self, buffer_size: int = 8, fps: int = 30, processing_duration: float = 2.0, input_mode: str = "video"):
        VideoStream.__init__(self, buffer_size, fps, processing_duration, input_mode)
        VideoProcessing.__init__(self, buffer_size, input_mode)
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.start_stream()

    def stop(self):
        if self.is_running:
            self.stop_stream()

    def get_video_tensor(self):
        """Process frames from the buffer"""
        self._capture_frames()
        if len(self.frame_buffer) < self.buffer_size:
            yield None  # Not enough frames yet
        yield self.process_frames(self.frame_buffer[-self.buffer_size:])
