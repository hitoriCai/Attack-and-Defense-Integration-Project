#!/bin/bash

# 设置数据集路径和模型路径
DATASET_PATH="./imagenet"
MODEL_PATH="./450.pt"

# 执行 Python 脚本以评估模型精度
python evaluate.py --data $DATASET_PATH --model-path $MODEL_PATH
