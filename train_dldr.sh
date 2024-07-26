# TWA (DDP version) 60+2
datasets=ImageNet
device=0,1,2,3

model=resnet18
wd_psgd=0.00001
lr=0.3
# DST=save_resnet18
DST = checkpoint_resnet18_vanilla_save_DOT_NOT_DELETE
CUDA_VISIBLE_DEVICES=$device python -m torch.distributed.launch --nproc_per_node 4 train_twa_ddp.py \
        --epochs 2 --datasets $datasets --opt SGD --schedule step --worker 8 \
        --lr $lr --params_start 0 --params_end 301 --train_start -1 --wd $wd_psgd \
        --batch-size 256 --arch $model --save-dir $DST --log-dir $DST