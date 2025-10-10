import os
import cv2
import numpy as np

import torch
import torchaudio

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from argparse import Namespace

class NDCME(Dataset):
    def __init__(self, datapath: str, mode: str, args: Namespace, partition: list= [0.6, 0.2, 0.2]) -> None:
        super().__init__()
        self.datapath = datapath
        self.video_path = os.path.join(self.datapath, 'face', mode)
        self.audio_path = os.path.join(self.datapath, 'audio', mode)
        self.fix_modality_mismatch()

        np.random.seed(0)
        np.random.shuffle(self.filename_list)

        if mode == 'train':
            self.filename_list = self.filename_list[:int(partition[0]*len(self.filename_list))]
        elif mode == 'val':
            self.filename_list = self.filename_list[int(partition[0]*len(self.filename_list)):int((1-partition[2])*len(self.filename_list))]
        elif mode == 'test':
            self.filename_list = self.filename_list[int((1-partition[2])*len(self.filename_list)):]

        
        self.mode = mode

        self.duration = args.duration
        self.fps = args.fps
        self.current_fps = 25
        self.num_frames = int(self.fps*self.duration)
        preprocessing = [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.5452, 0.3590, 0.2160],
                std=[0.2193, 0.1876, 0.1488]
            )
        ]
        self.preprocess = T.Compose(preprocessing)
        self.norm_mean=-7.80455191
        self.norm_std=8.83769718

        self.expr2int = {'Laughs':0, 'Smiles':1, 'None':2}
        expr_count = [
            np.count_nonzero(['Laughs' in vid for vid in self.filename_list]),
            np.count_nonzero(['Smiles' in vid for vid in self.filename_list]),
            np.count_nonzero(['None' in vid for vid in self.filename_list])
        ]
        self.weighted_sampling = [
            1/expr_count[0] if 'Laughs' in vid 
            else 1/expr_count[1] if 'Smiles' in vid
            else 1/expr_count[2] for vid in self.filename_list
        ]
        self.len_limit = np.min(expr_count)

    def fix_modality_mismatch(self):
        files_to_keep = []
        wavs_ext = 'npz'
        video_files = os.listdir(self.video_path)
        audio_files = os.listdir(self.audio_path)
        for vid in video_files:
            if f"{os.path.splitext(vid)[0]}.{wavs_ext}" in audio_files:
                files_to_keep.append(os.path.splitext(vid)[0])

        self.filename_list = files_to_keep

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, index):
        filename = self.filename_list[index]

        video_path = os.path.join(self.video_path, f'{filename}.mp4')
        frames = []
        cap = cv2.VideoCapture(video_path)
        current_fps, current_length = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        retain_frame_count = self.duration * current_fps
        fps_ratio = int(current_fps / self.fps)
        if current_length <= retain_frame_count:
            start_idx = 0
            retain_frame_count = current_length
        else:
            start_idx = np.random.randint(0, int(current_length - retain_frame_count))
        
        keep = np.arange(start_idx, start_idx + retain_frame_count, fps_ratio, dtype=np.uint16)
        keep = keep[:self.num_frames]
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if ret and (frame_count in keep):
                # resize to 224x224
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (224, 224)))

            frame_count += 1
        cap.release()
        
        if len(frames) < self.num_frames:
            # if the video is too short, repeat the the last frame to make it self.num_frames
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))
        else:
            # randomly pick self.num_frame indices, then sort them and form final output
            indices = np.random.choice(len(frames), self.num_frames, replace=False)
            indices = np.sort(indices)
            frames = [frames[i] for i in indices]

        frames = [self.preprocess(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frames)
        frames = torch.einsum('tchw->chwt', frames)

        audio_path = os.path.join(self.audio_path, f'{filename}.npz')
        waveform = np.load(audio_path, allow_pickle=True)['data']
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform - waveform.mean()

        ## In case we need spectrograms
        fbank = torchaudio.compliance.kaldi.fbank(
                waveform, 
                htk_compat=True, 
                sample_frequency=16000, 
                use_energy=False, 
                window_type='hanning', 
                num_mel_bins=128, 
                dither=0.0,
                frame_length=25,
                frame_shift=10
        )
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        
        fbank = (fbank - self.norm_mean) / (self.norm_std)

        label, intensity = self.get_label_from_filename(filename)
        label = self.expr2int[label]
        return frames, fbank, label, intensity
    
    def get_label_from_filename(self, filename):
        return filename.split('_')[4:6]