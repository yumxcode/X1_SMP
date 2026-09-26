"""Isaac-side policy evaluation: fall time / speed / height per episode.
Answers definitively whether the AMP policy runs in its training engine."""
import os
import sys
import runpy
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-index", "--no-deps", "--find-links",
                       os.path.join(ROOT, "vendor_wheels"),
                       "gymnasium", "farama-notifications", "cloudpickle",
                       "diffusers", "huggingface-hub", "filelock",
                       "importlib-metadata", "zipp", "packaging",
                       "typing-extensions", "regex", "tqdm", "safetensors",
                       "requests", "matplotlib", "contourpy", "cycler",
                       "fonttools", "kiwisolver", "pyparsing",
                       "python-dateutil", "six", "tensorboardX", "protobuf"])

sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import numpy as np  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

NUM_ENVS = 8
EP_SECONDS = 10.0
EPISODES = 3

env = env_builder.build_env("data/envs/amp_x1_env.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            NUM_ENVS, "cuda:0", visualize=False,
                            record_video=False)
agent = agent_builder.build_agent("data/agents/amp_x1_agent.yaml",
                                  env, "cuda:0")
agent.load("data/models/amp/amp_final.pt")
agent.eval()
agent.set_mode(AgentMode.TEST)

char_id = env._get_char_id()
engine = env._engine

for ep in range(EPISODES):
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    n_steps = int(EP_SECONDS * 30)
    root_z = np.zeros((n_steps, NUM_ENVS))
    root_x = np.zeros((n_steps, NUM_ENVS))
    done_any = np.zeros(NUM_ENVS, dtype=bool)
    fall_t = np.full(NUM_ENVS, np.inf)
    for t in range(n_steps):
        action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
        obs, r, done, info = agent._step_env(action)
        agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)
        rp = engine.get_root_pos(char_id).cpu().numpy()
        root_z[t] = rp[:, 2]
        root_x[t] = rp[:, 0]
        fell_now = (rp[:, 2] < 0.30) & (~done_any)
        fall_t[fell_now] = t / 30.0
        done_any |= fell_now
    dt = 1.0 / 30.0
    path = np.abs(np.diff(root_x, axis=0)).sum(axis=0)
    speed = path / EP_SECONDS
    print(f"[eval] ep{ep}: fall_t med {np.median(fall_t):.2f}s "
          f"(min {np.min(fall_t):.2f} max {np.max(fall_t):.2f}) | "
          f"alive>9s: {int(np.sum(fall_t > 9.0))}/{NUM_ENVS} | "
          f"|vx| med {np.median(speed):.2f} m/s | "
          f"rootz med {np.median(root_z):.3f}", flush=True)

print("[eval] done", flush=True)
