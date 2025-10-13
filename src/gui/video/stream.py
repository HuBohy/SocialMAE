
import glob
import random
from tkinter import messagebox
import cv2
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, buffer_size: int = 8, fps: int = 30, processing_duration: float = 2.0, input_mode: str = "video"):
        self.cap = None
        self.is_running = False
        self.frame_buffer = []
        self.current_frame = None
        self.has_processed = False  # Flag to indicate if frames have been processed

        self.buffer_size = buffer_size
        self.fps = fps
        self.input_mode = input_mode  # "video" or "webcam"
        self.processing_duration = processing_duration  # seconds to wait before processing frames

        # Video database path
        self.video_db_files = glob.glob(os.path.abspath("../../../databases/cremad/processed_cremad/*.flv"))
        self.current_video_path = random.choice(self.video_db_files) if self.video_db_files else None

    def start_stream(self):
        try:
            self.stop_stream()  # Ensure any existing stream is stopped
            self.cap = cv2.VideoCapture(0 if self.input_mode == "webcam" else self.current_video_path)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.is_running = True
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self.is_running = False

    def stop_stream(self):
        try:
            self.is_running = False
            self.frame_counter = 0
            self.loop_counter = 0
            if self.cap:
                self.cap.release()
                self.cap = None
            self.frame_buffer.clear()
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            self.is_running = False

    def _capture_frames(self):
        try:
            if self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    if self.input_mode == "webcam":
                        frame = cv2.flip(frame, 1)  # Mirror for webcam

                    self.current_frame = frame

                    if len(self.frame_buffer) >= self.buffer_size:
                        self.frame_buffer = self.frame_buffer[-self.buffer_size:]
                    if self.frame_counter % (max(self.fps // (self.buffer_size/self.processing_duration), 1)) == 0:
                        self.frame_buffer.append(frame)
                    if self.frame_counter % (self.buffer_size // 4) == 0:
                        self.has_processed = False  # Reset processing flag every buffer_size frames
                    self.frame_counter += 1
                else:
                    # Video ended, loop it or load next random video
                    if self.input_mode == "video":
                        self.loop_counter += 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if self.loop_counter >= 2:
                            self.load_random_video()
                            self.loop_counter = 0
        except Exception as e:
            logger.error(f"Error capturing frames: {e}")
            self.stop_stream()
        
    def load_random_video(self):
        """Load a random video from the database"""
        try:
            if not self.video_db_files:
                messagebox.showerror("Error", f"No video files found. Tried patterns:\n{self.video_db_files}")
                logger.error(f"No video files found in any pattern")
                return
            was_running = self.is_running
            if was_running:
                self.stop_stream()

            self.current_video_path = random.choice(self.video_db_files)
            video_name = os.path.basename(self.current_video_path)

            if was_running:
                self.start_stream()
                
        except Exception as e:
            logger.error(f"Error loading random video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")

        return video_name
