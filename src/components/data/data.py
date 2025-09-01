import os
import numpy as np
from pathlib import Path
from scipy.signal import  hilbert, decimate
import librosa
from mne.filter import filter_data, notch_filter
import scipy
import config as config
import librosa
from torch.utils.data import Dataset, DataLoader
from config import Config
from src.components.utils import print_herader
import pdb

class EEGMelDataset(Dataset):
    """Custom Dataset for loading EEG and Mel data."""
    def __init__(self, eeg_segments, mel_segments):
        print_herader("Initializing EEGAudioDataset")
        self.eeg_segments = eeg_segments
        self.mel_segments = mel_segments

    def __len__(self):
        return len(self.eeg_segments)

    def __getitem__(self, idx):
        eeg = self.eeg_segments[idx]  # shape [Segments, C, T]
        mel = self.mel_segments[idx]  # shape [Segments, T]
        return eeg, mel

class AudioMelDataset(Dataset):
    """Custom Dataset for loading Audio and Mel data."""
    def __init__(self, audio_segments, mel_segments):
        print_herader("Initializing EEGAudioDataset")
        self.audio_segments = audio_segments
        self.mel_segments = mel_segments

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        audio = self.audio_segments[idx]  # shape [Segments, C, T]
        mel = self.mel_segments[idx]  # shape [Segments, T]
        return audio, mel

class AudioComponent:
    """Class to handle audio data loading and processing."""
    def __init__(self, subject):
        print_herader(f"Initializing AudioComponent for subject {subject}")
        self.audio = None   
        self.file_path = Path(config.DATA_DIR, f'P{subject}_audio.npy')
        self.cfg = Config()
        self.sr = 48000
        self.audio = self.load_audio()

    def load_audio(self):
        print(f"Loading and resampling audio from {self.file_path}")
        audio = np.load(self.file_path)
        audio_22k = librosa.resample(audio, orig_sr=self.sr, target_sr=22050)
        self.audio_22k = audio_22k
        print(f"Audio loaded and resampled to 16kHz. New shape: {self.audio_22k.shape}")

    def _get_audio(self):
        return self.audio_16k
    

class EEGComponent:
    """Class to handle EEG data loading and processing."""
    def __init__(self, subject):
        print_herader(f"Initializing EEGComponent for subject {subject}")
        self.eeg = None
        self.stimulus = None
        self.channels = None
        self.eeg_file_path = Path(config.DATA_DIR, f'P{subject}_sEEG.npy')
        self.stimulus_file_path = Path(config.DATA_DIR, f'P{subject}_stimuli.npy')
        self.channel_info_path = Path(config.DATA_DIR, f'P{subject}_channels.npy')
        self.cfg = Config()
        self.sr = self.cfg.EEG_SR
        self._load_eeg()
        self._clean_eeg()
        self._preprocess_eeg()
       

    def _load_eeg(self):
        print(f"Loading EEG data from {self.eeg_file_path}")
        print(f"stimulus from {self.stimulus_file_path}")
        print(f"Channel info from {self.channel_info_path}")
        self.eeg = np.load(self.eeg_file_path)
        self.stimulus = np.load(self.stimulus_file_path)
        self.channel_info = np.load(self.channel_info_path, allow_pickle=True)
        self.channel_info = self.channel_info.flatten()
       

    def _electrode_shaft_referencing(self):
        """
        Perform electrode shaft referencing by computing the mean signal
        for each shaft and subtracting it from the corresponding channels.
        """
        print("Performing electrode shaft referencing...")
        self.channels = self.channel_info
        data_esr = np.zeros_like(self.eeg)

        shafts = {}
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            shafts.setdefault(shaft_name, []).append(i)

        shaft_averages = {
            shaft: np.mean(self.eeg[:, indices], axis=1, keepdims=True)
            for shaft, indices in shafts.items()
        }
        
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            data_esr[:, i] = self.eeg[:, i] - shaft_averages[shaft_name].squeeze()

        self.eeg = np.array(data_esr, dtype="float64")
        print(self.eeg.shape)
        print("Electrode shaft referencing completed.") 

    def _clean_eeg(self):
        print("Cleaning EEG data...")
        self.channels = self.channel_info
        clean_data = []
        clean_channels = []
        channels = self.channels
        for i in range(channels.shape[0]):
            if '+' in channels[i][0]: #EKG/MRK/etc channels
                continue
            elif channels[i][0][0] == 'E': #Empty channels
                continue
            elif channels[i][0][:2] == 'el': #Empty channels
                continue
            elif channels[i][0][0] in ['F','C','T','O','P']: #Other channels
                continue        
            else:
                clean_channels.append(channels[i])
                clean_data.append(self.eeg[:,i]) 
        
        self.channels = clean_channels
        print(self.eeg.shape)
        print("EEG cleaning completed.")

    def _preprocess_eeg(self):
        print("Preprocessing EEG data...")
        self._clean_eeg
        self._electrode_shaft_referencing() 
        print("Detrending and filtering EEG data...")
        self.eeg = scipy.signal.detrend(self.eeg, axis=0) 
        print("Applying bandpass and notch filters...")
        self.eeg = self.eeg.T
        self.eeg = filter_data(self.eeg, sfreq=self.sr, l_freq=0.5, h_freq=170.0, verbose=False)
        self.eeg = notch_filter(self.eeg, Fs=self.sr, freqs=[50, 100, 150], verbose=False)
        self.eeg = self.eeg.T
        print("EEG shape: ", self.eeg.shape)
        print("EEG preprocessing completed.")

    def _get_eeg(self):
        return self.eeg
        
       
class DataSegmenter:
    """Class to segment EEG and audio data into chunks based on Config."""
    def __init__(self, eeg, audio):
        print_herader("Initializing DataSegmenter")
        self.eeg = eeg
        self.audio = audio
        self.cfg = Config()
        self.eeg_sr = self.cfg.EEG_SR   
        self.chunk_size = self.cfg.CHUNK_SIZE
        self.audio_sr = self.cfg.AUDIO_SR
        self.eeg_len = int(self.eeg_sr * self.chunk_size)
        self.audio_len = int(self.audio_sr * self.chunk_size)

    def segment_data(self):
        print(f"Segmenting data into chunks of size {self.chunk_size} seconds")
        
        mel_spec = librosa.feature.melspectrogram(
            y=self.audio,
            sr=self.audio_sr,
            n_fft=self.audio_len,
            hop_length=self.audio_len,
            win_length=self.audio_len,
            n_mels=self.cfg.N_MELS,
            power=2.0
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.T
        print("Mel spectrogram shape:", mel_spec_db.shape)
        
        n_eeg_windows = self.eeg.shape[0] // self.eeg_len
        eeg_windows = self.eeg[:n_eeg_windows*self.eeg_len].reshape(n_eeg_windows, self.eeg_len, self.eeg.shape[1])
        print("EEG segmented shape:", eeg_windows.shape)

        n_audio_windows = len(self.audio) // self.audio_len
        audio_segments = self.audio[:n_audio_windows * self.audio_len].reshape(n_audio_windows, self.audio_len)

        print("Audio segments shape:", audio_segments.shape)
        
        min_len = min(mel_spec_db.shape[0], eeg_windows.shape[0], audio_segments.shape[0])
        mel_spec_db = mel_spec_db[:,:min_len]
        eeg_windows = eeg_windows[:min_len]
        audio_segments = audio_segments[:min_len]
        print("-"*70)
        print("Mel spectrogram shape:", mel_spec_db.shape)
        print("EEG segmented shape:", eeg_windows.shape)
        print("Audio segments shape:", audio_segments.shape)

        return eeg_windows, mel_spec_db, audio_segments
        