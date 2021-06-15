from ._AMRNet_trainer import AMRNetTrainer
from ._patched_UNet_trainer import PatchedUNetTrainer
from ._UNet_trainer import UNetTrainer
from ._pix2pixHD_trainer import Pix2PixHDTrainer

def get_trainer(name):
    TRAINERS = {
        'AMR_Net': AMRNetTrainer,
        'Patched_UNet': PatchedUNetTrainer,
        'UNet': UNetTrainer,
        'Pix2PixHD': Pix2PixHDTrainer,
    }

    for n, t in TRAINERS.items():
        # Compare as lowercase
        if n.lower() == name.lower():
            return t

    raise ValueError(f'trainer {name} is not defined')
