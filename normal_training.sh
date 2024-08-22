# without --eps $eps, meaning attack epsilon is zero, which means vanilla training
datasets=ImageNet
device=0,1
model=resnet101
path=/opt/data/common/ILSVRC2012/
epochs=90
DST=eps_8_save_$model
eps=8
CUDA_VISIBLE_DEVICES=$device  python3 normal_training.py -a $model \
    --epochs $epochs --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
    --dist-backend 'nccl' --multiprocessing-distributed \
    --world-size 1 --rank 0 $path --save_dir $DST --eps $eps

# with --eps $eps,  meaning adversarial training
#datasets=ImageNet
#device=0,1
#model=resnet18
#path=/opt/data/common/ILSVRC2012/
#epochs=30
#CUDA_VISIBLE_DEVICES=$device  python3 normal_training.py -a $model \
#    --epochs $epochs --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
#    --dist-backend 'nccl' --multiprocessing-distributed \
#    --world-size 1 --rank 0 $path --save_dir eps_4_save_$model --eps 4

# with --eps $eps, meaning adversarial training
# datasets=ImageNet
# device=0,1
# model=resnet18
# path=/opt/data/common/ILSVRC2012/
# epochs=90
# CUDA_VISIBLE_DEVICES=$device  python3 normal_training.py -a $model \
#     --epochs $epochs --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
#     --dist-backend 'nccl' --multiprocessing-distributed \
#     --world-size 1 --rank 0 $path --save_dir eps_8_save_$model --eps 8

# # without --eps $eps, meaning attack epsilon is zero, which means vanilla training
# datasets=ImageNet
# device=0,1
# model=resnet101
# path=/opt/data/common/ILSVRC2012/
# epochs=90
# CUDA_VISIBLE_DEVICES=$device  python3 normal_training.py -a $model \
#     --epochs $epochs --workers 8  --dist-url 'tcp://127.0.0.1:1234' \
#     --dist-backend 'nccl' --multiprocessing-distributed \
#     --world-size 1 --rank 0 $path --save_dir save_$model

