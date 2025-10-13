import numpy as np
import torch
import torchaudio

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, audio_rate=16000, audio_chunk=1024, n_mels=128):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.debug("Using GPU for processing")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.debug("Using Apple Silicon GPU for processing")
        else:
            self.device = torch.device("cpu")
            logger.debug("Using CPU for processing")

        self.audio_rate = audio_rate
        self.audio_chunks = audio_chunk  # Model expects 1024 time steps
        self.n_mels = n_mels  # Model expects 128 mel bands

        self.norm_mean=-6.2943
        self.norm_std=12.26344
        self.current_buffer_size = 0  # To track the size of the current audio buffer
    
    def get_audio_features(self, audio_buffer):
        """Extract audio features for the model - records and converts to mel spectrogram using only numpy/pyaudio"""
        try:
            # Get the last portion of audio data
            audio_data = np.array(list(audio_buffer), dtype=np.float32)
            self.current_buffer_size = len(audio_data)
            
            # Take the most recent audio data (e.g., last 1 second)
            samples_needed = int(self.audio_rate * 1.0)  # 1 second of audio
            if len(audio_data) >= samples_needed:
                audio_data = audio_data[-samples_needed:]
            audio_data = torch.from_numpy(audio_data).unsqueeze(0)
            audio_data = audio_data - audio_data.mean()

            fbank = torchaudio.compliance.kaldi.fbank(
                    audio_data, 
                    htk_compat=True, 
                    sample_frequency=self.audio_rate, 
                    use_energy=False, 
                    window_type='hanning', 
                    num_mel_bins=self.n_mels, 
                    dither=0.0,
                    frame_length=25,
                    frame_shift=10
            )
            n_frames = fbank.shape[0]
            p = self.audio_chunks - n_frames
            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:self.audio_chunks, :]
            
            audio_tensor = (fbank - self.norm_mean) / (self.norm_std)

            return audio_tensor.unsqueeze(0).to(self.device)

        except Exception as e:
            logger.error(f"Error processing audio features: {e}")
            return None

    def get_audio_duration(self):
        """Get duration of the audio buffer in seconds"""
        return self.current_buffer_size / self.audio_rate