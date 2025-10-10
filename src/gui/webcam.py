import threading
import time
from PIL import Image
import cv2

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebCAM:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.thread = None

    def start_webcam(self):
        if not self.is_running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.error("Cannot open webcam")
                    return
                
                self.is_running = True
                self.thread = threading.Thread(target=self.update_frame, daemon=True)
                self.thread.start()
                
            except Exception as e:
                logger.error(f"Failed to start webcam: {str(e)}")

    def stop_webcam(self):
        if self.is_running:
            self.is_running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None

    
    def update_frame(self):
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit in the GUI
                height, width = frame_rgb.shape[:2]
                max_width, max_height = 640, 480
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
            
            time.sleep(0.033)  # ~30 FPS