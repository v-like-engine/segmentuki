import torch

from models.SPADE.architecture import Pix2PixHDwithSPADE


def load_model(dataset='ade', version_='30'):
    if dataset == 'ade' and version_ not in ['30', '150', '200', 30, 150, 200]:
        return None
    elif dataset == 'facades' and version_ not in ['200', '400', '700', 200, 400, 700]:
        return None
    elif dataset not in ['ade', 'facades']:
        return None
    return Pix2PixHDwithSPADE()
