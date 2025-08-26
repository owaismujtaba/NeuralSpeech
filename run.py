

from src.data.data import DataLoader

dataloader = DataLoader(subject='01')
mel_frames, eeg_frames = dataloader.get_data()