import torch
import torchaudio
from torchvision import transforms as T
from torch.utils.data import Dataset

import os
import cv2
import random
import numpy as np
import PIL

from glob import glob


class MyDataset(Dataset):
    """
    Custom Dataset for loading and processing video and audio data.
    The dataset handles synchronized video and audio data,
    performing video frame sampling and audio feature extraction as specified.

    Args:
        video_dirpath (str): Path to the directory containing video files.
        audio_dirpath (str): Path to the directory containing audio files.
        partition (str, optional): Specifies the dataset partition to use. Defaults to 'dev'.
        target_fps (int, optional): Target frame rate to sample the video. Defaults to 2.
        target_duration (int, optional): Length in seconds of the video clips to process. Defaults to 4.
    """
    def __init__(self,
                video_dirpath: str,
                audio_dirpath: str,
                partition: str = 'dev',
                target_fps: int =2,
                target_duration:int = 4
            ) -> None:
        super().__init__()
        if not os.path.exists(os.path.abspath(video_dirpath)):
            raise ValueError(f"{os.path.abspath(video_dirpath)} path does not exists.")
        if not os.path.exists(os.path.abspath(audio_dirpath)):
            raise ValueError(f"{os.path.abspath(audio_dirpath)} path does not exists.")
        if not os.path.isdir(os.path.abspath(video_dirpath)):
            raise ValueError(f"{os.path.abspath(video_dirpath)} is not a directory.")
        if not os.path.isdir(os.path.abspath(audio_dirpath)):
            raise ValueError(f"{os.path.abspath(audio_dirpath)} is not a directory.")

        self.dirpath = {'video': os.path.abspath(video_dirpath), 'audio': os.path.abspath(audio_dirpath)}
        self.partition = partition
        
        if not target_fps > 0:
            raise ValueError("Target FPS must be a positive integer.")
        if not target_duration > 0:
            raise ValueError("Target duration must be a positive integer.")
        self.target_fps = target_fps
        self.target_duration = target_duration

        self.ids = [os.path.basename(folder) for folder in glob(os.path.join(self.dirpath['video'], self.partition, '**'))]

        self.vids = []
        self.wavs = []
        self.labels = []
        for id in self.ids:
            urls = [os.path.basename(dir) for dir in glob(os.path.join(self.dirpath['video'], self.partition, id, '**'))]
            for url in urls:
                filenames = [os.path.splitext(os.path.basename(dir))[0] for dir in glob(os.path.join(self.dirpath['video'], self.partition, id, url, '*'))]
                for filename in filenames:
                    self.vids.append(os.path.join(self.dirpath['video'], self.partition, id, url, filename+'.mp4'))
                    self.wavs.append(os.path.join(self.dirpath['audio'], self.partition, id, url, filename+'.wav'))
                    self.labels.append(id)
        if not len(self.wavs) == len(self.vids):
            raise ValueError(f'Video and audio sizes are not matching up.')
        
        self.video_preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224), interpolation=PIL.Image.BICUBIC, antialias=None),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )
        ])
        self.norm_mean = -6.2943
        self.norm_std = 12.26344

    def wav2fbank(self, filename: str) -> torch.Tensor:
        """
        Converts an audio file to Mel-frequency bank (fbank) features using Kaldi's fbank function.
        This function also handles padding or trimming the feature matrix to a fixed size.

        Args:
            filename (str): Path to the audio file.

        Returns:
            torch.Tensor: A tensor containing normalized Mel-frequency bank features.
        """
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple: A tuple containing three elements:
                - torch.Tensor: Processed and normalized video frames as a tensor.
                - torch.Tensor: Processed and normalized Mel-frequency bank features as a tensor.
                - torch.Tensor: One-hot encoded vector representing the label of the data item.
        """
        filename = self.vids[index]
        imgs = torch.Tensor()

        video = cv2.VideoCapture(filename)
        current_fps = video.get(cv2.CAP_PROP_FPS)
        length = video.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_interval = int(current_fps // self.target_fps)

        new_length = self.target_duration * self.target_fps
        n_frames_to_keep = self.target_duration * current_fps
        if length < n_frames_to_keep:
            start_idx = 0
            n_frames_to_keep = length
        else:
            start_idx = random.randint(0, int(length - n_frames_to_keep))

        keep = np.arange(start_idx, start_idx + n_frames_to_keep, frame_interval)
        keep = keep[:new_length]

        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_count in keep:
                frame = self.video_preprocess(frame)
                imgs = torch.cat((imgs, frame.unsqueeze(0)), dim=0)
            frame_count += 1
        video.release()
        imgs = torch.einsum('tchw->chwt', imgs)

        if length < self.target_duration * current_fps:
            imgs = imgs.repeat(1, 1, 1, int(new_length//imgs.shape[-1] + 1))
            imgs = imgs[:, :, :, :new_length]

        audio_file = self.wavs[index]
        fbank = self.wav2fbank(audio_file)

        fbank = (fbank - self.norm_mean) / (self.norm_std)

        label = self.ids.index(self.labels[index])
        label_indices = torch.zeros(len(self.ids))
        label_indices[label] += 1
        label = torch.FloatTensor(label_indices)

        return imgs, fbank, label

    def __len__(self):
        return len(self.labels)