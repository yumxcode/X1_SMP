"""Rollout the trained policy in Isaac and dump the full trajectory
(obs/state/action per control step) to output/ for offline MuJoCo diffing."""
import os
import sys
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
import envs.env_builder as env_builder  # noqa: E402  (isaacgym before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

env = env_builder.build_env("data/envs/amp_x1_env.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/amp_x1_agent.yaml",
                                  env, "cuda:0")
agent.load("data/models/amp/amp_final.pt")
agent.eval()
agent.set_mode(AgentMode.TEST)

char_id = env._get_char_id()
e = env._engine
agent._curr_obs, agent._curr_info = agent._reset_envs()

traj = dict(obs=[], root_pos=[], root_quat=[], root_vel=[],
            root_ang_vel=[], dof_pos=[], dof_vel=[], action=[])
N = 120  # 4 s @ 30 Hz
for t in range(N):
    obs = agent._curr_obs[0].cpu().numpy()
    action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
    traj["obs"].append(obs)
    traj["root_pos"].append(e.get_root_pos(char_id)[0].cpu().numpy())
    traj["root_quat"].append(e.get_root_rot(char_id)[0].cpu().numpy())
    traj["root_vel"].append(e.get_root_vel(char_id)[0].cpu().numpy())
    traj["root_ang_vel"].append(e.get_root_ang_vel(char_id)[0].cpu().numpy())
    traj["dof_pos"].append(e.get_dof_pos(char_id)[0].cpu().numpy())
    traj["dof_vel"].append(e.get_dof_vel(char_id)[0].cpu().numpy())
    traj["action"].append(action[0].cpu().numpy())
    nobs, r, done, info = agent._step_env(action)
    agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
torch.save({k: torch.tensor(np.stack(v)) for k, v in traj.items()},
           os.path.join(ROOT, "output", "isaac_traj.pt"))
print("[traj] dumped", N, "steps -> output/isaac_traj.pt", flush=True)
import time; time.sleep(60)  # let the SDK upload before container dies
