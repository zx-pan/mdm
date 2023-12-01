import os
MODEL_NAME = "GlaS_Exp_1_crop_256"

if __name__ == "__main__":
    MODEL_FLAGS="--in_channels 3 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
    DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"
    TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --microbatch 2"
    
    CALL=f"mpiexec -n 4 python /afs/crc.nd.edu/user/z/zpan3/Models/guided-diffusion/scripts/image_train.py --data_dir /afs/crc.nd.edu/user/z/zpan3/Datasets/GlaS/mae_glas/traintest --log_interval 10 --save_interval 10000 --out_dir checkpoints/checkpoint_{MODEL_NAME} {MODEL_FLAGS} {DIFFUSION_FLAGS} {TRAIN_FLAGS}"
    
    os.system(CALL)