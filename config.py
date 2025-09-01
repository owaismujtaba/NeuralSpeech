import os
from pathlib import Path
#import torch

CUR_DIR = os.getcwd()
DATA_DIR = Path(CUR_DIR, 'data', 'raw')


class Config:
    EEG_CHANNELS = 64
    EEG_SR = 1024 # original sr
    AUDIO_SR = 22050 # compatible with hifi-gan
    CHUNK_SIZE = 0.05  # in seconds
    N_MELS = 80 # bins
    