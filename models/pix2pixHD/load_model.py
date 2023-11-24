import torch

from models.pix2pixHD.pix2pixHD_model import Pix2PixHD


def load_model(dataset='ade', version_='30'):
    if dataset == 'ade':
        if version_ not in ['30', '150', '200', 30, 150, 200]:
            return None
    elif dataset == 'facades':
        if version_ not in ['200', '400', '700', 200, 400, 700]:
            return None
    else:
        return None
    model = Pix2PixHD()
    model.load_state_dict(torch.load(f"models/pix2pixHD/weights/model_{dataset}_128_{version_}"))
    return model
