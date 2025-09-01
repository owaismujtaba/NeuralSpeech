import pdb
from src.components.data.data import EEGComponent, AudioComponent
from src.components.data.data import DataSegmenter, AudioMelDataset

from torch.utils.data import DataLoader

eeg_loader = EEGComponent(subject='01')
audio_loader = AudioComponent(subject='01')

eeg_data = eeg_loader._get_eeg()
audio_data = audio_loader._get_audio()


print(f"EEG data shape: {eeg_data.shape}")
print(f"Audio data shape: {audio_data.shape}")

segmenter = DataSegmenter(eeg_data, audio_data)
eeg_segments, mel_spec_segments, audio_segments = segmenter.segment_data()

print(f"Segmented EEG data shape: {eeg_segments.shape}")
print(f"Segmented mel data shape: {mel_spec_segments.shape}")
print(f"Segmented Audio data shape: {audio_segments.shape}")

dataset = AudioMelDataset(
    audio_segments=audio_segments, 
    mel_segments= mel_spec_segments
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

for audi, mel in loader:
    print(audi.shape, mel.shape)
    break