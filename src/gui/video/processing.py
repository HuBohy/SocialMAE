import cv2
import torch
from torchvision import transforms

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessing:
    def __init__(self, buffer_size: int = 8, input_mode: str = "video"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using GPU for processing")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon GPU for processing")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for processing")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resize_transform = lambda img: cv2.resize(img, dsize=(224, 224))

        self.input_mode = input_mode  # "webcam" or "video"
        self.buffer_size = buffer_size
        self.face_coords = None  # To store the coordinates of the detected face
        self.has_processed = False  # Flag to indicate if frames have been processed
        self.last_result = None  # Store last result to avoid reprocessing

    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        del gray  # Free memory

        return faces
    
    def preprocess_frames(self, frames):
        """Preprocess 8 frames for SocialMAE model"""
        try:
            processed_frames = []
            for frame in frames:
                # Convert BGR to RGB
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                frame_tensor = self.transform(frame)
                processed_frames.append(frame_tensor)
            
            # Stack frames: (8, 3, 224, 224) -> (1, 3, 224, 224, 8)
            frame_stack = torch.stack(processed_frames)
            del processed_frames  # Free memory
            frame_stack = torch.einsum('tchw->chwt', frame_stack)
            frame_stack = frame_stack.unsqueeze(0)  # Add batch dimension
            
            return frame_stack.to(self.device)
        except Exception as e:
            logger.error(f"Error preprocessing frames: {e}")
            if 'processed_frames' in locals():
                del processed_frames
            return None

    def process_frames(self, frames):
        """Get the processing with a new frame"""
        if not self.has_processed:
            if self.input_mode == "webcam":
                faces = self.detect_faces(frames[-1])
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
                    self.face_coords = (x, y, w, h)
                    
                    # Extract face regions from all 8 frames
                    face_frames = []
                    for buffered_frame in frames:
                        # Use the same face coordinates for all frames
                        face_img = buffered_frame[y:y+h, x:x+w]
                        # Resize to 224x224 for consistency
                        face_img = self.resize_transform(face_img)
                        face_frames.append(face_img)
                    # Preprocess and predict using all 8 frames
                    frames = face_frames
                    del face_frames  # Free memory

            result = self.preprocess_frames(frames)
            self.last_result = result
            
            self.has_processed = True  # Mark that frames have been processed
        else:
            result = self.last_result  # Use the last processed result
        return result

    def set_label_on_frame(self, label):
        """Set the label on the current frame"""
        if self.face_coords:
            x, y, w, h = self.face_coords
                # Draw rectangle around face
            self.current_frame = cv2.rectangle(self.current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.current_frame = cv2.putText(self.current_frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    def display_frame(self):
        """Return the current frame for display"""
        # Convert BGR to RGB for display
        try:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

            # Resize frame to fit in the GUI
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 640, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            frame_rgb = cv2.cvtColor(cv2.resize(self.current_frame, (224, 224)), cv2.COLOR_BGR2RGB)

        return frame_rgb
