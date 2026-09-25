#!/bin/bash
# AMP training for X1 29DOF run dataset (Isaac Gym)
set -e
cd "$(dirname "$0")/.."
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/amp_x1_env.yaml \
  --agent_config data/agents/amp_x1_agent.yaml \
  --visualize false --out_dir output/
