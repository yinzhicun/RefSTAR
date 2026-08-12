#!/bin/bash
# Example training launch commands for RefSTAR 512x512.
# Adjust --nproc_per_node and master_port to match your environment.

# Phase 1: base Arc2Face model (4 GPUs)
TF_ENABLE_ONEDNN_OPTS=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
  --nproc_per_node=4 --master_port=29500 \
  train/train.py -opt train/train_phase1_512.yml --launcher pytorch --auto_resume

# Phase 2: RefSTAR dual-reference model (4 GPUs).
# Before running, open train/train_phase2_512.yml and set pretrain_network_g
# and pretrain_network_d to the phase-1 outputs, e.g.:
#   pretrain_network_g: experiments/RefSTAR_phase1_512/models/net_g_xxx.pth
#   pretrain_network_d: experiments/RefSTAR_phase1_512/models/net_d_xxx.pth
# TF_ENABLE_ONEDNN_OPTS=0 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
#   --nproc_per_node=4 --master_port=29500 \
#   train/train.py -opt train/train_phase2_512.yml --launcher pytorch --auto_resume
