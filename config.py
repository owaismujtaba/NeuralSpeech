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

    BATCH_SIZE = 16
    EPOCHS = 100
    LR_GEN = 2e-4
    LR_DISC = 2e-4
    LAMBDA_FEAT = 10.0 # weight for feature matching loss
    LAMBDA_ADV = 2.0 # weight for adversarial loss

    