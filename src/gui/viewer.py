import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import torch
import torch.nn as nn

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from src.models import SocialMAEFT
from src.gui.video.thread import VideoThread
from src.gui.audio.thread import AudioThread

class GUI:
    def __init__(self, root, current_task="emotions"):
        self.root = root
        self.root.title("SocialMAE Audio-Visual Recognition")
        self.root.geometry("900x700")
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.current_video_path = None
        self.input_mode = "webcam"  # "video" or "webcam"
        
        self.main_thread = None
        self.video_thread = None
        self.audio_thread = None

        self.video_buffer_size = 8  # Number of frames for SocialMAE
        self.video_fps = 25  # Assuming 25 FPS for webcam/video
        self.audio_rate = 16000
        self.processing_duration = 0.27  # seconds

        # Emotion labels
        self.task = {
            "lsn": {"model_path": "socialMAE_ndcme.pth", "labels": ["Laughs", "Smiles", "Neutral"]},
            "emotions": {"model_path": "socialMAE_cremad_mm.pth", "labels": ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]}
        }
        self.emotion_labels = self.task[current_task]["labels"]
        self.model_path = self.task[current_task]["model_path"]

        # Initialize SocialMAE model
        self.setup_model()

        # Create GUI elements
        self.setup_gui()

        self.start_capture()

    def setup_model(self):
        """Initialize and load the SocialMAE model"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for processing")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Silicon GPU for processing")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for processing")

            self.model = SocialMAEFT(label_dim=len(self.emotion_labels), n_frame=8)
            best_weight = torch.load(self.model_path,
                                map_location=self.device)
            try:
                model = self.model
                # Load pretrained weights
                missing, unmatched = model.load_state_dict(best_weight, strict=False)
            except (missing, unmatched) as e:
                logger.warning(f"Model weights have missing or unmatched keys: {missing}, {unmatched}")
                if "module" in missing:
                    model = nn.DataParallel(self.model)
                missing, unmatched = model.load_state_dict(best_weight, strict=False)

            model.to(self.device)
            self.model = model.eval()

            logger.info("SocialMAE model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display label
        self.video_label = ttk.Label(main_frame, text="Ready to start...")
        self.video_label.grid(row=0, column=0, columnspan=4, pady=10)

        # Input mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Input Mode", padding="5")
        mode_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        self.mode_var = tk.StringVar(value=self.input_mode)
        # webcam_radio = ttk.Radiobutton(mode_frame, text="Webcam + Mic", variable=self.mode_var, 
        #                               value="webcam", command=self.on_mode_change)
        # webcam_radio.grid(row=0, column=0, padx=10)
        
        # video_radio = ttk.Radiobutton(mode_frame, text="Random Video", variable=self.mode_var, 
        #                              value="video", command=self.on_mode_change)
        # video_radio.grid(row=0, column=1, padx=10)
        
        # Video info label
        self.video_info_label = ttk.Label(mode_frame, text="", font=("Arial", 9))
        self.video_info_label.grid(row=1, column=0, columnspan=2, pady=5)

        # Prediction display
        self.prediction_label = ttk.Label(main_frame, text="Emotion: None", font=("Arial", 14, "bold"))
        self.prediction_label.grid(row=1, column=0, columnspan=4, pady=10)
        
        # Audio status display
        self.audio_status_label = ttk.Label(main_frame, text="Audio: Ready", font=("Arial", 10))
        self.audio_status_label.grid(row=2, column=0, columnspan=4, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_capture)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_capture)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.quit_button = ttk.Button(button_frame, text="Quit", command=self.quit_app)
        self.quit_button.grid(row=0, column=2, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Status: Ready")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def on_mode_change(self):
        """Handle input mode change"""
        self.input_mode = self.mode_var.get()
        self.stop_capture()
        self.start_capture()
        
        
    def start_capture(self):
        """Start webcam or video capture and audio recording"""
        if not self.is_running:
            try:
                # Setup video capture
                self.video_thread = VideoThread(buffer_size=self.video_buffer_size, fps=self.video_fps, processing_duration=self.processing_duration, input_mode=self.input_mode)
                self.video_thread.start()

                if self.input_mode == "webcam":
                    # Setup audio recording for webcam mode
                    self.audio_thread = AudioThread(audio_rate=self.audio_rate, audio_duration=self.processing_duration)
                    self.audio_thread.start_thread()
                    self.audio_thread.start()

                self.is_running = True
                self.main_thread = threading.Thread(target=self.update_frame, daemon=True)
                self.main_thread.start()

                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")
                self.status_label.config(text=f"Status: Running ({self.video_thread.input_mode})")
                self.audio_status_label.config(text="Audio: Recording" if self.input_mode == "webcam" else "Audio: N/A")
                # self.video_info_label.config(text=f"Selected: {self.video_thread.current_video_path}" if self.input_mode == "video" else "")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start capture: {str(e)}")

    def stop_capture(self):
        """Stop capture and audio recording"""
        if self.is_running:
            self.is_running = False
            
            # Stop video thread
            if self.video_thread:
                self.video_thread.stop()

            # Stop audio recording
            self.audio_recording = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.stop_thread()
                self.audio_thread.join(timeout=1.0)

            # Stop main thread
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=1.0)

            self.video_label.config(image="", text="Capture stopped")
            self.prediction_label.config(text="Emotion: None")
            self.audio_status_label.config(text="Audio: Stopped" if self.input_mode == "webcam" else "Audio: N/A")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_label.config(text="Status: Stopped")

    def update_frame(self):
        emotion_text = "Collecting frames..."
        while self.is_running:
            audio_tensor = next(self.audio_thread.get_audio_tensor()) if self.input_mode == "webcam" else None
            # audio_tensor = None
            video_tensor = next(self.video_thread.get_video_tensor())
            if video_tensor is not None:
                self.audio_status_label.config(text=f"Audio: Recording ({self.processing_duration:.2f}s, target: {self.processing_duration:.2f}s)" if audio_tensor is not None else "Audio: N/A")
                emotion, confidence = next(self.predict_emotion_from_frames(video_tensor, audio_tensor))
                emotion_text = f"{emotion} ({confidence:.2f})"
                if audio_tensor is not None:
                    emotion_text += " [A+V]"
                else:
                    emotion_text += " [V]"

            # Update prediction label
            self.prediction_label.config(text=f"Emotion: {emotion_text}")
            self.video_info_label.config(text=f"Selected: {self.video_thread.current_video_path}" if self.input_mode == "video" else "")
            self.video_thread.set_label_on_frame(emotion_text)
            frame_rgb = self.video_thread.display_frame()

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update the label with the new image
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference

            # Control frame rate
            self.root.after(int(1000/self.video_fps))  # ~30 FPS

    
    def predict_emotion_from_frames(self, frame_tensor, audio_tensor=None):
        """Predict emotion using SocialMAE model with 8 frames and audio"""
        if self.model is None or frame_tensor is None:
            return "Model not loaded", 0.0
        
        try:
            with torch.no_grad():
                # Use real audio if available, otherwise use dummy audio
                if audio_tensor is None:
                    audio_tensor = torch.zeros(1, 1024, 128).to(self.device)

                # Choose mode based on available inputs
                # if audio_tensor is not None and self.input_mode == "webcam":
                mode = "multimodal"
                # else:
                #     mode = "videoonly"
                
                # Run inference
                output = self.model(audio_tensor, frame_tensor, mode=mode)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion = self.emotion_labels[predicted.item()]
                conf = confidence.item()
                
                yield emotion, conf
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            yield "Error", 0.0
           
    def quit_app(self):
        self.video_thread.stop_stream()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = GUI(root, current_task="lsn")
    
    # Handle window closing
    def on_closing():
        app.quit_app()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()