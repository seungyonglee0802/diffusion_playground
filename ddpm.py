import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.nn import init
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import math

import time
import datetime

from IPython.display import HTML
from IPython.display import clear_output

from functools import partial

from unet import UNetModel
from dataset import MNISTDataset, GuidedMNISTDataset


class Model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T, **kwargs):
        """
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        """

        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0
        ).to(device=device)
        self.backbone = UNetModel(**kwargs).to(device)

        self.to(device=self.device)

        self.in_training = True

    def loss_fn(self, x, guide=None):
        """
        This function performed when only training phase.

        x   : perturbated data
        guide : guide image (concatenated to x)
        """
        time_step = torch.randint(0, len(self.alpha_bars), (x.size(0),)).to(
            device=self.device
        )
        output, epsilon = self.forward(x, time_step=time_step, guide=guide)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, time_step, guide=None):
        if self.in_training:
            used_alpha_bars = self.alpha_bars[time_step][:, None, None, None]
            epsilon = torch.randn_like(x)
            x_tilde = (
                torch.sqrt(used_alpha_bars) * x
                + torch.sqrt(1 - used_alpha_bars) * epsilon
            )

        else:  # one step for inference
            time_step = (
                torch.Tensor([time_step for _ in range(x.size(0))]).to(
                    device=self.device).long()
            )
            x_tilde = x

        if guide is not None:
            x_tilde = torch.cat([x_tilde, guide], dim=1)

        output = self.backbone(x_tilde, time_step)

        if self.in_training:
            return output, epsilon
        else:
            return output


class DiffusionProcess:  # ONLY for the Inference
    def __init__(
        self,
        beta_1,
        beta_T,
        T,
        diffusion_model,
        device,
        shape,
        val_dataset=None,
    ):
        """
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_model  : trained diffusion network
        shape         : data shape
        val_dataset   : validation dataset (for guide image)
        """
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(
            1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0
        ).to(device=device)
        self.alpha_prev_bars = torch.cat(
            [torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]]
        )
        self.shape = shape

        self.diffusion_model = diffusion_model
        self.device = device
        self.val_dataset = val_dataset

    def _one_diffusion_step(self, x, guide=None):
        """
        x   : perturbated data
        guide : guide image (concatenated to x)
        """
        for idx in reversed(range(len(self.alpha_bars))):
            noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
            sqrt_tilde_beta = torch.sqrt(
                (1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
            predict_epsilon = self.diffusion_model(x, idx, guide=guide)
            mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

            x = mu_theta_xt + sqrt_tilde_beta * noise

            yield x

    @torch.no_grad()
    def sampling(self, sampling_batch, only_final=False):
        """
        sampling_batch  : a number of generation
        only_final      : If True, return is an only output of final schedule step
        """
        sample = torch.randn([sampling_batch, *self.shape]
                             ).to(device=self.device)
        sampling_list = []

        if self.val_dataset:
            # from validation dataset sample first sample_batch
            val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=sampling_batch, shuffle=False
            )
            guide = next(iter(val_loader))[1].to(
                device=self.device
            )  # 0: image, 1: guide, 2: label

        assert sample.shape == guide.shape

        final = None
        for sample in self._one_diffusion_step(sample, guide):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return guide, final if only_final else torch.stack(sampling_list)


def imshow(initial, context, result, sampling_batch):
    # TODO: visualize context
    assert (
        initial.shape[0] == result.shape[0]
    ), "init and result must have same batch size"
    multiple_col = 2  # initial, result

    plt.figure(figsize=(10, 10 * multiple_col))
    clear_output()

    row_number = int(math.sqrt(sampling_batch))
    col_number = int(math.sqrt(sampling_batch))

    initial = initial[:sampling_batch].detach().cpu().numpy()
    result = result[:sampling_batch].detach().cpu().numpy()
    B, C, H, W = result.shape
    show_sample = np.zeros(
        [row_number * H, col_number * W * multiple_col], dtype=np.float32
    )
    for row in range(row_number):
        for col in range(col_number):
            _initial = initial[row + col * row_number][0]
            _result = result[row + col * row_number][0]
            # display initial normalize b/w 0~225
            show_sample[
                row * H: (row + 1) * H,
                col * W * multiple_col: (col * multiple_col + 1) * W,
            ] = (
                (_initial - _initial.min()) /
                (_initial.max() - _initial.min()) * 255
            )
            # display result normalize b/w 0~225
            show_sample[
                row * H: (row + 1) * H,
                (col * multiple_col + 1) * W: ((col + 1) * multiple_col) * W,
            ] = (
                (_result - _result.min()) /
                (_result.max() - _result.min()) * 255
            )

    show_sample = show_sample.astype(np.uint8)
    plt.axis(False)
    plt.imshow(show_sample, cmap="gray")
    plt.savefig(
        "./MNIST_diffusion/sample.png", format="png", bbox_inches="tight", pad_inches=0
    )


def train(model, optim, dataloader, device, MAX_EPOCH):
    start = time.time()

    for epoch in range(MAX_EPOCH):
        for idx, (x, guide, _) in enumerate(dataloader):
            x = x.to(device)
            guide = guide.to(device) if guide is not None else None
            loss = model.loss_fn(x, guide)
            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch : {epoch}, Loss : {loss.item()}")

    end = time.time()
    print(f"Training Time : {end - start}")

    if os.path.exists("./model") is False:
        os.mkdir("./model")
    torch.save(model.state_dict(), f"./model/model_final.pth")


def inference(model, device, beta_1, beta_T, T, sampling_batch, shape, val_dataset):
    inference_process = DiffusionProcess(
        beta_1, beta_T, T, model, device, shape, val_dataset
    )
    model.load_state_dict(torch.load(f"./model/model_final.pth"))
    model.eval()
    inference_process.diffusion_model.in_training = False

    guide, result = inference_process.sampling(sampling_batch, only_final=True)
    imshow(guide, None, result, sampling_batch)


if __name__ == "__main__":
    # Hyperparameter
    beta_1 = 1e-4
    beta_T = 0.02
    T = 500
    C, H, W = 1, 16, 16
    MAX_EPOCH = 10

    guide_slice = (slice(0, 8), slice(0, 8))

    # DataSet
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((H, W)),
            torchvision.transforms.ToTensor()
        ]
    )
    trn_dataset = GuidedMNISTDataset(
        "./MNIST",
        transform=transform,
        guide_slice=guide_slice,
    )
    val_dataset = GuidedMNISTDataset(
        "./MNIST",
        train=False,
        transform=transform,
        guide_slice=guide_slice,
    )
    guide_channel = trn_dataset.guide_channel or 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = Model(
        device,
        beta_1,
        beta_T,
        T,
        image_size=H,
        in_channels=C + guide_channel,
        model_channels=128,
        out_channels=C,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        num_heads=8,
    )

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    dataloader = torch.utils.data.DataLoader(
        trn_dataset, batch_size=128, drop_last=True, num_workers=0
    )

    train(model, optim, dataloader, device, MAX_EPOCH)
    inference(
        model,
        device,
        beta_1,
        beta_T,
        T,
        16,
        shape=(C, H, W),
        val_dataset=val_dataset if guide_channel > 0 else None,
    )
