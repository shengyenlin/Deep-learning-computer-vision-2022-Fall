
echo -e Experiment settings: "\n" \
python3 p1_train.py \
    --num_workers 4 \
    --nc 3 \
    --nz 300 \
    --ngf 64 \
    --ndf 64 \
    --lr 2e-4 \
    --beta1 0.5 \
    --batch_size 128 \
    --num_epoch 300 \
    --noise StandardNorm \
    --add_ms_loss 1 \
    --num_pics_generated 1000 \
"\n" \
Start Training;

python3 -W ignore \
    p1_train.py \
    --num_workers 4 \
    --nc 3 \
    --nz 300 \
    --ngf 64 \
    --ndf 64 \
    --lr 2e-4 \
    --beta1 0.5 \
    --batch_size 128 \
    --num_epoch 300 \
    --noise StandardNorm \
    --add_ms_loss 1 \
    --num_pics_generated 1000 \