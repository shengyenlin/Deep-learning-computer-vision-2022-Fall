
echo -e Experiment settings: "\n" \
    --num_workers 4 \
    --lr 1e-3 \
    --batch_size 128 \
    --num_epoch 300 \
    --timesteps 200 \
    --beta_schedule "linear" \
    --loss_type "huber" \
"\n" \
Start Training;

CUDA_LAUNCH_BLOCKING=1 python3 \
    p2_train.py \
    --num_workers 4 \
    --lr 1e-3 \
    --batch_size 128 \
    --num_epoch 300 \
    --timesteps 200 \
    --beta_schedule "linear" \
    --loss_type "huber";