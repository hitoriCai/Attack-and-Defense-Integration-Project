# TWA (DDP version) 60+2
# without --eps $eps, meaning attack epsilon is zero, which means vanilla training
datasets=ImageNet
device=0,1
model=resnet18
wd_psgd=0.00001
lr=0.3
path=/opt/data/common/ILSVRC2012/
DST=/opt/data/private/checkpoint_resnet18_vanilla_save_DOT_NOT_DELETE
params_start=0
params_end=301
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 2 train_dldr.py \
        --epochs 2 --datasets $datasets --opt SGD --schedule step --worker 8 \
        --lr $lr --params_start $params_start  --params_end $params_end  --train_start -1 --wd $wd_psgd \
        --batch-size 256 --arch $model --save-dir $DST --log-dir $DST --eps 0


# with --eps $eps,  meaning adversarial training
datasets=ImageNet
device=0,1
model=resnet18
wd_psgd=0.00001
lr=0.3
path=/opt/data/common/ILSVRC2012/
DST=/opt/data/private/fgsm-at@eps:4_save_resnet18
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 2 train_dldr.py \
        --epochs 2 --datasets $datasets --opt SGD --schedule step --worker 8 \
        --lr $lr --params_start $params_start  --params_end $params_end   --train_start -1 --wd $wd_psgd \
        --batch-size 256 --arch $model --save-dir $DST --log-dir $DST --eps 4