import os
MODEL_NAME = "MoNuSeg_Exp_1_crop_256"

if __name__ == "__main__":
    MODEL_FLAGS="--in_channels 3 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --learn_sigma False --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True"
    DIFFUSION_FLAGS="--patch_size 8"
    TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 1"
    
    CALL=f"CUDA_VISIBLE_DEVICES=1 python /home/ubuntu/Model/mask-diffusion/scripts/image_train.py --data_dir /home/ubuntu/Data/MoNuSeg/TrainTest_Folder/img --log_interval 10 --save_interval 5000 --out_dir checkpoints/checkpoint_{MODEL_NAME} {MODEL_FLAGS} {DIFFUSION_FLAGS} {TRAIN_FLAGS}"
    
    os.system(CALL)