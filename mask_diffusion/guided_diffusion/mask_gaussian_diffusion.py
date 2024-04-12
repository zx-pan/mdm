import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import math
import torchgeometry as tgm
import os
from PIL import Image

# helpers functions

import torchvision


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            *,
            channels=3,
            timesteps=1000,
            loss_type='l2',
            patch_size=8
    ):
        super().__init__()
        self.channels = channels
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.patch_size = patch_size

    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_img(self, img, mask_ratio):
        """
        perform random masking on image
        :param img: [1, C, H, W]
        :param mask_ratio: 0 < ratio < 1
        :return: masked imgs
        """
        # img to patches
        x = self.patchify(img)
        N, L, D = x.shape  # batch, length, dim

        # masking patches
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask_patch = torch.zeros([N, L - x_masked.shape[1], D], device=x.device)
        x_masked = torch.cat([x_masked, mask_patch], dim=1)
        x_masked = torch.gather(x_masked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle

        # masked patches to img
        img_masked = self.unpatchify(x_masked)

        return img_masked

    def q_sample(self, x_start, t):
        x_t = torch.zeros_like(x_start, device=x_start.device)

        for i in range(t.shape[0]):
            mask_ratio = int(t[i]) / self.num_timesteps
            x_t[i] = self.random_masking_img(x_start[i].unsqueeze(0), mask_ratio)

        return x_t

    def training_losses(self, model, x_start, t):
        terms = {}
        x_t = self.q_sample(x_start=x_start, t=t)
        output = model(x_t, t)

        if self.loss_type == 'l1':
            terms["loss"] = (x_start - output).abs().mean()
        elif self.loss_type == 'l2':
            terms["loss"] = F.mse_loss(x_start, output)
        elif self.loss_type == 'ssim':
            ssim = tgm.losses.SSIM(5, reduction='mean')
            terms["loss"] = ssim(x_start, output)
        else:
            raise NotImplementedError()

        return terms


# unit test
if __name__ == "__main__":
    import torch
    import sys

    sys.path.append("..")
    sys.path.append(".")
    from src.datasets import make_transform, ImageLabelPNGDataset
    from torch.utils.data import Dataset, DataLoader
    from torchvision.utils import save_image

    ds = ImageLabelPNGDataset(
        data_dir='/home/ubuntu/Data/GlaS/test',
        mode='train',
        resolution=256,
        num_images=1,
        transform=make_transform(
            'ddpm',
            (256, 256)
        )
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    dataiter = iter(loader)
    x, y = next(dataiter)
    gd = GaussianDiffusion()
    x_t = gd.q_sample(x, t=torch.tensor([250]))
    save_image(x_t, '/home/ubuntu/Model/mask-diffusion/test.jpg')
