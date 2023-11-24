import numpy as np

import sys

sys.path.append("segmentuki")

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2


class ADEDataset(Dataset):

    def __init__(self, root_path, resize=128):
        self.img_path = root_path / "images/training"
        self.mask_path = root_path / "annotations/training"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
        ])
        self.img_list = sorted(self._get_filenames(self.img_path))
        self.mask_list = sorted(self._get_filenames(self.mask_path))

        self._read_imgs()
        self._read_masks()

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        return self.transform(img).float(), self.transform(mask).float()

    def __len__(self):
        return len(self.img_list)

    def _get_filenames(self, path):
        return [f for f in path.iterdir() if f.is_file()]

    def _read_imgs(self):
        self.images = [cv2.imread(str(f)) for f in self.img_list]

    def _read_masks(self):
        self.masks = [cv2.imread(str(f)) for f in self.mask_list]


class FacadesDataset(Dataset):

    def __init__(self, root_path, resize):
        self.img_path = root_path / "trainA"
        self.mask_path = root_path / "trainB"

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
        ])
        self.img_list = sorted(self._get_filenames(self.img_path))
        self.mask_list = sorted(self._get_filenames(self.mask_path))

        self._read_imgs()
        self._read_masks()

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        return self.transform(img).float(), self.transform(mask).float()

    def __len__(self):
        return len(self.img_list)

    def _get_filenames(self, path):
        return [f for f in path.iterdir() if f.is_file()]

    def _read_imgs(self):
        self.images = [cv2.imread(str(f)) for f in self.img_list]

    def _read_masks(self):
        self.masks = [cv2.imread(str(f)) for f in self.mask_list]


def preprocess(mask):
    file_bytes = np.asarray(bytearray(mask.read()), dtype=np.uint8)
    mask = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    mask = (mask * 255.0).astype("uint8")
    size = mask.shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ])
    return transform(mask).float(), torch.tensor(mask.copy()).float(), size


def postprocess(spade, res, filename, size, dataset='ade', beta=-25):
    size = [size[:2][1], size[:2][0]]
    res = cv2.resize(res, dsize=size, interpolation=cv2.INTER_CUBIC)
    res = (res + 1) / 2
    res = (res * 255.0).astype("uint8")
    np.clip(res + beta, 0, 255).astype(np.uint8)
    if not spade:
        res = cv2.fastNlMeansDenoisingColored(res, None, 10, 20, 7, 21)
    return res
