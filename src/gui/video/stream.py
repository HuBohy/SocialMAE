
import glob
import random
from tkinter import messagebox
import cv2
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, buffer_size: int = 8, fps: int = 30, input_mode: str = "video"):
        self.cap = None
        self.is_running = False
        self.frame_buffer = []

        self.buffer_size = buffer_size
        self.fps = fps
        self.input_mode = input_mode  # "video" or "webcam"

        # Video database path
        self.video_db_files = glob.glob(os.path.abspath("../../../databases/cremad/processed_cremad/*.flv"))
        self.current_video_path = random.choice(self.video_db_files) if self.video_db_files else None

    def start_stream(self):
        try:
            self.stop_stream()  # Ensure any existing stream is stopped
            logger.debug(f"Starting video stream in mode: {self.input_mode}")
            self.cap = cv2.VideoCapture(0 if self.input_mode == "webcam" else self.current_video_path)
            self.is_running = True
            logger.debug("Video stream started")
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            self.is_running = False

    def stop_stream(self):
        try:
            self.is_running = False
            logger.debug("Stopping video stream")
            if self.cap:
                self.cap.release()
                self.cap = None
            logger.debug("Clearing frame buffer")
            self.frame_buffer.clear()
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            self.is_running = False

    def _capture_frames(self):
        try:
            counter = 0
            while self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    if self.input_mode == "webcam":
                        frame = cv2.flip(frame, 1)  # Mirror for webcam
                    if len(self.frame_buffer) >= self.buffer_size:
                        self.frame_buffer.pop(0)
                    self.frame_buffer.append(frame)
                else:
                    # Video ended, loop it or load next random video
                    if self.input_mode == "video":
                        counter += 1
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        if counter >= 1000:  # After 1000 loops, load a new random video
                            self.load_random_video()
                            counter = 0
                    else:
                        break  # Webcam issue, break the loop
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
            
            self.current_video_path = random.choice(self.video_db_files)
            video_name = os.path.basename(self.current_video_path)
            logger.debug(f"Selected video: {self.current_video_path}")
                
        except Exception as e:
            logger.error(f"Error loading random video: {e}")
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")

        return video_name
