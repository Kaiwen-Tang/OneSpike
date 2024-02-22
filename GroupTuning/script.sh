MODEL_DIR=VGG16_t16.pt
SAVE_DIR=VGG16_test_16.pt
LOG_DIR=logs/test0221.txt

echo $MODEL_DIR
# echo $SAVE_DIR

python train.py \
    --resume ${MODEL_DIR} \
    --save ${SAVE_DIR} \
    --testBatch 20 \
    --logdir ${LOG_DIR} \
    --epochs 0 \
    --timestep 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # then 3 linear layer
