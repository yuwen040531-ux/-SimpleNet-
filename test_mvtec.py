#!/bin/bash

datapath="/root/autodl-tmp/mvtec"
# 如果你只想画某几个类别的图，可以把不想画的从这里删掉，节省时间
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

echo "开始加载 MVTec 最佳权重，批量生成热力图..."

python3 main.py \
  --test \
  --save_segmentation_images \
  --gpu 0 \
  --seed 0 \
  --log_group simplenet_mvtec \
  --log_project MVTecAD_Results_FullPower \
  --results_path /root/autodl-tmp/results_mvtec \
  --run_name run_sota \
  net \
  -b wideresnet50 \
  -le layer2 \
  -le layer3 \
  --pretrain_embed_dimension 1536 \
  --target_embed_dimension 1536 \
  --patchsize 3 \
  --embedding_size 256 \
  --pre_proj 1 \
  dataset \
  --batch_size 8 \
  --resize 329 \
  --imagesize 288 \
  "${dataset_flags[@]}" mvtec $datapath