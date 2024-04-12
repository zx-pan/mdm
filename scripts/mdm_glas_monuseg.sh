# mask ddpm
DIFFUSION_FLAGS="--diffusion_steps 1000 --use_scale_shift_norm True --patch_size 8"

# denoise ddpm
#DIFFUSION_FLAGS="--diffusion_steps 1000 --use_scale_shift_norm True --noise_schedule cosine"

MODEL_FLAGS="--in_channels 3 --attention_resolutions 32,16,8 --class_cond False --image_size 256 --learn_sigma False --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True"
DATASET="monuseg_1" # Available datasets: monuseg_1, glas_1

python train_interpreter_medical.py --exp ./experiments/${DATASET}/mdm.json $MODEL_FLAGS $DIFFUSION_FLAGS