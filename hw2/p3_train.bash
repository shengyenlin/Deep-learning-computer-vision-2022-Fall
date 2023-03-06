echo -e Experiment settings: "\n" \
    --source_domain mnistm \
    --target_domain svhn \
    --optim SGD \
    --lr 1e-2 \
    --momentum 0.9 \
    --batch_size 64 \
    --num_epoch 300 \
    --use_dann 1 \
    --train_on_source 0 \
    --train_on_target 0 \
    --use_scheduler 0 \
"\n" \
Start Training;

CUDA_LAUNCH_BLOCKING=1 python3 \
    p3_train.py \
    --source_domain mnistm \
    --target_domain svhn \
    --optim SGD \
    --lr 1e-2 \
    --momentum 0.9 \
    --batch_size 64 \
    --num_epoch 300 \
    --use_dann 1 \
    --train_on_source 0 \
    --train_on_target 0 \
    --use_scheduler 0;