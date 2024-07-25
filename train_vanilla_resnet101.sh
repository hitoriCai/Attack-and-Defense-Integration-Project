datasets=ImageNet
device=0,1

model=resnet101
path=/opt/data/common/ILSVRC2012/
CUDA_VISIBLE_DEVICES=$device  python3 train_hzb.py -a $model \
    --epochs 90 --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
    --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 $path