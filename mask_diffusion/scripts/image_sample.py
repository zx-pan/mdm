"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import torch as th
import torch.distributed as dist

# from guided_diffusion.image_datasets import load_data
from guided_diffusion.bratsloader import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.sample_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_original_imgs = []

    # ---------Sampling from noised img-------#
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    # ---------------------------------------#

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # ---------Sampling from noised img-------#
        batch, cond = next(data)
        t = 249
        t = th.tensor([t]).to(dist_util.dev())

        noise = None
        noisy_x = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
        # ---------------------------------------#

        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            noise=noisy_x,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        batch = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8).to(dist_util.dev())
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_imgs = [th.zeros_like(batch) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_imgs, batch)  # gather not supported with NCCL
        all_original_imgs.extend([batch.cpu().numpy() for batch in gathered_imgs])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    img_arr = np.concatenate(all_original_imgs, axis=0)
    img_arr = img_arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"train_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, img_arr, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        sample_dir="samples",
        data_dir="/afs/crc.nd.edu/user/z/zpan3/Datasets/GlaS/train/img"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
