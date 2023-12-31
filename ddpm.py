import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import math
from IPython.display import clear_output

from omegaconf import OmegaConf
from hydra.utils import instantiate

import time
import os

from util import make_beta_schedule, extract_into_tensor, noise_like

from resnet import resnet2


class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 image_size=256,
                 channels=3,
                 concat_channels=0,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 timesteps=1000,
                 beta_schedule="linear",
                 cosine_s=8e-3,
                 given_betas=None,
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 log_every_t=100,
                 clip_denoised=True,
                 original_elbo_weight=0.,
                 # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 v_posterior=0.,
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], \
            'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.concat_channels = concat_channels
        self.use_positional_encodings = use_positional_encodings
        partial_model = instantiate(unet_config)
        self.model = partial_model(
            image_size=self.image_size,
            in_channels=self.channels + self.concat_channels,
            out_channels=self.channels,
        )
        # count_params(self.model, verbose=True)
        # self.use_ema = use_ema
        # if self.use_ema:
        #     self.model_ema = LitEma(self.model)
        #     print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        # self.learn_logvar = learn_logvar
        # self.logvar = torch.full(
        #     fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        #     self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        # below: coefficient for x_0 in the posterior mean calculation
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        # below: coefficient for x_t in the posterior mean calculation
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * \
                np.sqrt(torch.Tensor(alphas_cumprod)) / \
                (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.model_ema.restore(self.model.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(
            1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and noise (epsilon_0).
        """
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Get the distribution q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, guide=None, context=None, emb=None, clip_denoised: bool = True):
        """
        Predict the distribution p(x_{t-1} | x_t) by predicted (reconstructed) x_0.
        """
        if guide is not None:
            model_out = self.model(
                torch.cat([x, guide], dim=1), t, context=context, y=emb)
        else:
            model_out = self.model(x, t, context=context, y=emb)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, guide=None, context=None, emb=None, clip_denoised=True, repeat_noise=False):
        """
        Infer x_{t-1} from x_t.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, guide=guide, context=context, emb=emb, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, guide=None, context=None, emb=None, return_intermediates=False):
        """
        Infer x_0 from noise.
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                guide=guide, context=context, emb=emb,
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, guide=None, context=None, emb=None, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  guide=guide, context=context, emb=emb,
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(
                    target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, guide=None, context=None, emb=None, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = torch.cat(
            [x_noisy, guide], dim=1) if guide is not None else x_noisy

        # FIXME: emb is not embedding but class label
        # (UNetModel) def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        model_out = self.model(x_noisy, t, context=context, y=emb)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        # loss_vlb = (self.lvlb_weights[t] * loss).mean()
        # loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple  # + self.original_elbo_weight * loss_vlb

        # loss_dict.update({f'{log_prefix}/loss': loss})

        return loss  # , loss_dict

    def forward(self, x, guide=None, context=None, emb=None, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps,
                          (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, guide, context, emb, *args, **kwargs)

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = rearrange(x, 'b h w c -> b c h w')
    #     x = x.to(memory_format=torch.contiguous_format).float()
    #     return x

    # def shared_step(self, batch):
    #     x = self.get_input(batch, self.first_stage_key)
    #     loss, loss_dict = self(x)
    #     return loss, loss_dict

    # def training_step(self, batch, batch_idx):
    #     loss, loss_dict = self.shared_step(batch)

    #     self.log_dict(loss_dict, prog_bar=True,
    #                   logger=True, on_step=True, on_epoch=True)

    #     self.log("global_step", self.global_step,
    #              prog_bar=True, logger=True, on_step=True, on_epoch=False)

    #     if self.use_scheduler:
    #         lr = self.optimizers().param_groups[0]['lr']
    #         self.log('lr_abs', lr, prog_bar=True, logger=True,
    #                  on_step=True, on_epoch=False)

    #     return loss

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope():
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {
    #             key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
    #     self.log_dict(loss_dict_no_ema, prog_bar=False,
    #                   logger=True, on_step=False, on_epoch=True)
    #     self.log_dict(loss_dict_ema, prog_bar=False,
    #                   logger=True, on_step=False, on_epoch=True)

    # def on_train_batch_end(self, *args, **kwargs):
    #     if self.use_ema:
    #         self.model_ema(self.model)

    # def _get_rows_from_list(self, samples):
    #     n_imgs_per_row = len(samples)
    #     denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
    #     denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    #     denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    #     return denoise_grid

    # @torch.no_grad()
    # def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.first_stage_key)
    #     N = min(x.shape[0], N)
    #     n_row = min(x.shape[0], n_row)
    #     x = x.to(self.device)[:N]
    #     log["inputs"] = x

    #     # get diffusion row
    #     diffusion_row = list()
    #     x_start = x[:n_row]

    #     for t in range(self.num_timesteps):
    #         if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
    #             t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
    #             t = t.to(self.device).long()
    #             noise = torch.randn_like(x_start)
    #             x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #             diffusion_row.append(x_noisy)

    #     log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

    #     if sample:
    #         # get denoise row
    #         with self.ema_scope("Plotting"):
    #             samples, denoise_row = self.sample(
    #                 batch_size=N, return_intermediates=True)

    #         log["samples"] = samples
    #         log["denoise_row"] = self._get_rows_from_list(denoise_row)

    #     if return_keys:
    #         if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
    #             return log
    #         else:
    #             return {key: log[key] for key in return_keys}
    #     return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        # if self.learn_logvar:
        #     params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


def train(ddpm_process, train_loader, max_epochs=100, learning_rate=1e-4, context_encoder=None):
    start = time.time()

    ddpm_process.learning_rate = learning_rate
    opt = ddpm_process.configure_optimizers()

    best_loss = float('inf')  # Set initial best loss to infinity
    best_model_state = None  # Store the best model's state_dict here
    # Store the best context network's state_dict here
    best_context_network_state = None

    for epoch in range(max_epochs):
        for idx, (x, guide, context, label) in enumerate(train_loader):
            x = x.to(ddpm_process.device)
            guide = guide.to(
                ddpm_process.device) if guide is not None else None
            context = context.to(
                ddpm_process.device) if context is not None else None
            context = context_encoder(
                context) if context_encoder is not None and context is not None else None
            label = label.to(ddpm_process.device)
            loss = ddpm_process(x, guide=guide, context=context, emb=label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {epoch} loss: {loss.item()}")

        # Save best model (lowest loss) in memory
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = ddpm_process.state_dict()
            if context_encoder is not None:
                best_context_network_state = context_encoder.state_dict()

    end = time.time()
    print(f"Training time: {end - start}")

    # Save the best model weights to file at the end of training
    if os.path.exists("./model") is False:
        os.mkdir("./model")

    if best_model_state is not None:
        torch.save(best_model_state, f"./model/model_best.pth")

    if context_encoder is not None and best_context_network_state is not None:
        torch.save(best_context_network_state,
                   f"./model/context_encoder_best.pth")

    # Save the final model weights as well, if desired
    torch.save(ddpm_process.state_dict(), f"./model/model_final.pth")
    if context_encoder is not None:
        torch.save(context_encoder.state_dict(),
                   f"./model/context_encoder_final.pth")


def inference(ddpm_process, sampling_batch, return_intermediates=False, val_dataset=None, context_encoder=None):
    # load model
    ddpm_process.load_state_dict(torch.load(
        "./model/model_best.pth"))  # model_final.pth
    if context_encoder is not None:
        context_encoder.load_state_dict(
            torch.load("./model/context_encoder_best.pth"))  # context_encoder_final.pth

    # inference
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=sampling_batch, shuffle=None
        )
        _, guide, context, label = next(iter(val_loader))
        guide = guide.to(ddpm_process.device) if guide is not None else None
        context = context.to(
            ddpm_process.device) if context is not None else None
        encoded_context = context_encoder(
            context) if context_encoder is not None and context is not None else None
        label = label.to(ddpm_process.device)

    ddpm_process.eval()
    with torch.no_grad():
        samples = ddpm_process.sample(
            batch_size=sampling_batch,
            guide=guide,
            context=encoded_context,
            emb=label,
            return_intermediates=return_intermediates
        )

    if return_intermediates:
        samples, denoise_row = samples
        return guide, samples, denoise_row

    return guide, context, samples


def imshow(initial, context, result, sampling_batch):
    assert (
        initial.shape[0] == result.shape[0]
    ), "init and result must have same batch size"

    # number of not None among initial, context, result
    multiple_col = 1
    if context is not None:
        multiple_col += 1
    if result is not None:
        multiple_col += 1

    plt.figure(figsize=(10, 10 * multiple_col))
    clear_output()

    row_number = int(math.sqrt(sampling_batch))
    col_number = int(math.sqrt(sampling_batch))

    initial = initial[:sampling_batch].detach().cpu().numpy()
    context = context[:sampling_batch].detach(
    ).cpu().numpy() if context is not None else None
    result = result[:sampling_batch].detach().cpu(
    ).numpy() if result is not None else None

    # convert the numpy array into PIL Image format
    initial = np.transpose(initial, (0, 2, 3, 1))
    context = np.transpose(context, (0, 2, 3, 1)
                           ) if context is not None else None
    result = np.transpose(result, (0, 2, 3, 1)) if result is not None else None

    # set up the plot
    fig, axes = plt.subplots(row_number, col_number *
                             multiple_col, figsize=(30, 10))
    for i in range(row_number):
        for j in range(col_number):
            axes[i][j *
                    multiple_col].imshow(initial[i * col_number + j], cmap="gray")
            if context is not None:
                axes[i][j * multiple_col +
                        1].imshow(context[i * col_number + j])
            if result is not None:
                axes[i][j * multiple_col +
                        2].imshow(result[i * col_number + j], cmap="gray")

    for ax in axes.flatten():
        ax.axis("off")

    plt.axis(False)
    plt.savefig(
        "./MNIST_diffusion/sample.png", format="png", bbox_inches="tight", pad_inches=0
    )


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddpm_process = DDPM(
        unet_config=OmegaConf.load("config/model/base.yaml").unet,
        image_size=16,
        channels=1,
        concat_channels=1,
    )

    ddpm_process = ddpm_process.to(device)
    ddpm_process.device = device

    dataset = instantiate(
        OmegaConf.load("config/data/context_guide_mnist.yaml").dataset,
        _convert_="all",  # maintain python syntax
    )

    trn_dataset = dataset(train=True)
    val_dataset = dataset(train=False)

    trn_loader = DataLoader(
        trn_dataset, batch_size=128, drop_last=True, num_workers=0
    )

    # load resnet as context encoder
    context_encoder = resnet2().to(device)

    # train
    train(ddpm_process, trn_loader, max_epochs=10,
          context_encoder=context_encoder)

    # inference
    guide, context, result = inference(
        ddpm_process, sampling_batch=16, val_dataset=val_dataset, context_encoder=context_encoder)

    # visualize
    imshow(guide, context, result, 16)
