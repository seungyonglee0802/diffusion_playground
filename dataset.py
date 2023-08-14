import torch
import torchvision
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = torchvision.datasets.MNIST(
            root=self.root, train=True, download=True, transform=self.transform
        )

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, None, label  # None for guide

    def __len__(self):
        return len(self.dataset)


class GuidedMNISTDataset(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        guide_slice=(slice(0, 8), slice(0, 8)),
    ):
        self.root = root
        self.transform = transform
        self.guide_channel = 1
        self.guide_slice = guide_slice
        self.dataset = torchvision.datasets.MNIST(
            root=self.root, train=train, download=True, transform=self.transform
        )

    def get_guide(self, img):
        guide = torch.zeros_like(img)
        # guide_mask is a boolean tensor with 2D shape (H, W)
        guide_mask = torch.zeros_like(img[0], dtype=torch.bool)
        guide_mask[self.guide_slice] = True
        guide[:, guide_mask] = img[:, guide_mask]
        return guide

    def __getitem__(self, index):
        img, label = self.dataset[index]
        guide = self.get_guide(img)
        return img, guide, label

    def __len__(self):
        return len(self.dataset)
