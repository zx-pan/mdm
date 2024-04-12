MODEL_FLAGS="--patch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma False --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

DATASET=ffhq_34 # Available datasets: ffhq_34, celeba_19

python train_interpreter.py --exp experiments/${DATASET}/mdm.json $MODEL_FLAGS