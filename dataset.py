import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np

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
        guide_slice=[0, 8, 0, 8], # [start_h, end_h, start_w, end_w]
    ):
        self.root = root
        self.transform = transform
        self.guide_channel = 1
        self.guide_slice = guide_slice
        self.dataset = torchvision.datasets.MNIST(
            root=self.root, train=train, download=True
        )

    def get_guide(self, img):
        guide = torch.zeros_like(img)
        # guide_mask is a boolean tensor with 2D shape (H, W)
        guide_mask = torch.zeros_like(img[0], dtype=torch.bool)
        guide_mask[self.guide_slice[0]:self.guide_slice[1], self.guide_slice[2]:self.guide_slice[3]] = True
        guide[:, guide_mask] = img[:, guide_mask]
        return guide

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.transform(img)
        guide = self.get_guide(img)
        return img, guide, label

    def __len__(self):
        return len(self.dataset)


class ContextGuidedMNISTDataset(GuidedMNISTDataset):
    def __init__(
        self,
        root,
        context_root='./MNIST_context',
        train=True,
        transform=None,
        guide_slice=[0, 8, 0, 8], # [start_h, end_h, start_w, end_w]
    ):
        super().__init__(root, train, transform, guide_slice)
        self.context_dataset = torchvision.datasets.ImageFolder(root=context_root)
        self.context_transform = transforms.Compose([
            transforms.Resize(size=16),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
        ])
        self.dir_map = {
            "right": 2,
            "left": 1,
            "down": 0,
            "up": 3,
        }

    def __getitem__(self, index):
        _img, label = self.dataset[index]
        context, dir = self.context_dataset[index]
        
        if dir == self.dir_map["right"]:
            # rotate 90 degrees clockwise Image object
            img = _img.rotate(-90)
        elif dir == self.dir_map["left"]:
            # rotate 90 degrees counterclockwise
            img = _img.rotate(90)
        elif dir == self.dir_map["down"]:
            # rotate 180 degrees
            img = _img.rotate(180)
        else:
            img = _img
        img = self.transform(img)
        context = self.context_transform(context)
        _img = self.transform(_img)
        guide = self.get_guide(_img)
        return img, guide, context, label
    

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def visualize_data(dataset, index):
    # Fetch the tensors from the dataset using the given index
    img_with_pattern, guide, pattern, label = dataset[index]

    # Convert the tensors back to PIL images for visualization
    to_pil = transforms.ToPILImage()
    img_with_pattern = to_pil(img_with_pattern)
    guide = to_pil(guide)
    pattern = to_pil(pattern)

    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_with_pattern, cmap='gray')
    axes[0].set_title(f'Image with {label}')
    
    axes[1].imshow(guide, cmap='gray')
    axes[1].set_title('Guide')
    
    axes[2].imshow(pattern, cmap='gray')
    axes[2].set_title('Pattern')

    # Hide axis
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = ContextGuidedMNISTDataset(
        root='./MNIST',
        context_root='./MNIST_context',
        train=True,
        transform=transforms.Compose([
            transforms.Resize(size=16),
            transforms.ToTensor(),
        ]),
        guide_slice=[0, 8, 0, 8],
    )
    visualize_data(dataset, 5)