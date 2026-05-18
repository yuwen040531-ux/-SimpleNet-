#!/bin/bash

# 把路径换成你 AutoDL 上的真实 MVTec 路径
datapath=/root/autodl-tmp/mvtec
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

# 核心修改：gpu 改为 0，因为通常 AutoDL 单卡机器都是 gpu 0
python3 main.py \
--gpu 0 \
--seed 0 \
--log_group SimpleNet_CBAM_PatchGuard \
--log_project METecAD_Results_CBAM_PatchGuard \
--results_path /root/autodl-tmp/results_mvtec \
--run_name run_CBAM_PatchGuard \
--save_segmentation_images \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 \
--resize 329 \
--imagesize 288 \
"${dataset_flags[@]}" mvtec $datapath