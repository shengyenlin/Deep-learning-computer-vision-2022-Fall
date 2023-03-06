#!/bin/bash
python3 ./DirectVoxGO/run.py \
    --test_json_path $1 \
    --test_out_dir $2 \
    --ft_path ./p1_fine_last.tar \
    --config ./DirectVoxGO/configs/nerf/hotdog.py \
    --eval_ssim \
    --eval_lpips_vgg \
    --render_only \
    --render_test_no_gt \
    --dump_images;