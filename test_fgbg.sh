#!/bin/bash
SPLIT="val"
BATCH_SIZE=1
NUM_WORKERS=1

python test_fgbg.py --model-dir models/nusc/1s_forecasting \
    --test-epoch 14 \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --fg-bg "fg"
#    --test-split $SPLIT \
#    --compute-chamfer-dist \

