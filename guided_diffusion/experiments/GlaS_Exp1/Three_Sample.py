import os
import numpy as np
# Remember to set out_channels in script

MODEL_NAME = "GlaS_Exp_2"
SAMPLE_STEPS = [f"0{i}000" for i in np.arange(14, 15, 5)]
MODEL_TYPES = ["ema_0.9999_", "model"]

MODEL_FLAGS = "--in_channels 3 --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 64 --learn_sigma True --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 False"
DIFFUSION_FLAGS = "--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"

if __name__ == "__main__":
    # Sample for each specified step and model
    for step in SAMPLE_STEPS:
        for model in MODEL_TYPES:
            path = f"/afs/crc.nd.edu/user/z/zpan3/Models/guided-diffusion/job/Checkpoints/Finetune_Checkpoint_{MODEL_NAME}/{model}{step}.pt"
            OTHER=f"--batch_size 2 --num_samples 4 --timestep_respacing 250 --model_path {path} --data_dir /afs/crc.nd.edu/user/z/zpan3/Datasets/GlaS/train/img --sample_dir samples/{MODEL_NAME}/{model}{step}"

            CALL=f"python /afs/crc.nd.edu/user/z/zpan3/Models/guided-diffusion/scripts/image_sample.py  {MODEL_FLAGS} {DIFFUSION_FLAGS} {OTHER}"

            os.system(CALL)