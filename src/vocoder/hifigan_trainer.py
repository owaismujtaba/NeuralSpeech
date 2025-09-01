import json
from hifigan.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from hifigan.models import feature_loss, discriminator_loss, generator_loss
from hifigan.env import AttrDict
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HiFiGAN:
    def __init__(self, config_path, do_path, g_path):
        self.do_path = do_path
        self.g_path = g_path
        self.config_path = config_path

        self._set_configuration()
        self._initialize()


    def _set_configuration(self):
        print()
        with open(self.config_file) as f:
            config = AttrDict(json.load(f))

        self.generator = Generator(config).to(DEVICE)
        self.mpd = MultiPeriodDiscriminator().to(DEVICE)
        self.msd = MultiScaleDiscriminator().to(DEVICE)
    
    def _initialize(self):
        state_dict_g = torch.load(self.g_path, map_location=DEVICE)
        state_dict_do = torch.load(self.do_path, map_location=DEVICE)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.mpd.load_state_dict(state_dict_do['mpd'])
        self.msd.load_state_dict(state_dict_do['msd'])
        self.generator.train()
        self.mpd.train()
        self.msd.train()
