#!/bin/bash
# SMP policy training (requires prior at output/smp_prior_x1/model.pt)
set -e
cd "$(dirname "$0")/.."
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/smp_x1_env.yaml \
  --agent_config data/agents/smp_x1_agent.yaml \
  --visualize false --out_dir output/
