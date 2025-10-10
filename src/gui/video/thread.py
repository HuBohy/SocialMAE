import threading
import torch
import tkinter as tk
from tkinter import ttk
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.gui.video.processing import VideoProcessing
from src.gui.video.stream import VideoStream

class VideoThread(threading.Thread, VideoStream, VideoProcessing):
    def __init__(self, buffer_size: int = 8, fps: int = 30, input_mode: str = "video"):
        VideoStream.__init__(self, buffer_size, fps, input_mode)
        VideoProcessing.__init__(self, buffer_size, input_mode)
        self.is_running = False

        super().__init__(target=self._capture_frames, daemon=True)

    def start_thread(self):
        if not self.is_running:
            self.start_stream()

    def stop_thread(self):
        if self.is_running and self.is_alive():
            self.stop_stream()

    def get_video_tensor(self):
        """Process frames from the buffer"""
        if len(self.frame_buffer) < self.buffer_size:
            self.current_frame = self.frame_buffer[-1] if self.frame_buffer else None
            yield None  # Not enough frames yet
        yield self.process_frames(self.frame_buffer[-self.buffer_size:])
