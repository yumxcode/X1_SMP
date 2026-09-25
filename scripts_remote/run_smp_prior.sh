#!/bin/bash
# SMP prior (tiny-MDM diffusion) training on the X1 run dataset
set -e
cd "$(dirname "$0")/.."
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_x1_run.yaml \
  --out_dir output/smp_prior_x1
