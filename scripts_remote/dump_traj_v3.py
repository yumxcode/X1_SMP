"""Dump the SMP v3 policy rollout in Isaac WITH per-step applied torques,
contact forces, and PD targets — for one-to-one MuJoCo replay diffing.

v3 variant of dump_traj_v2.py: v3 env/agent yamls (fixed sole assets +
v3 retarget dataset), checkpoint from X1_DUMP_CKPT.
"""
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


def to_np(x):
    """anything (cuda/cpu tensor, numpy, gymtorch views) -> numpy."""
    import torch
    try:
        return torch.as_tensor(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)
import torch  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

CKPT = os.environ.get("X1_DUMP_CKPT", "data/models/smp/smp_v3b_it1800.pt")

env = env_builder.build_env("data/envs/smp_x1_env_v3.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, "cuda:0")
agent.load(CKPT)
agent.eval()
agent.set_mode(AgentMode.TEST)
print(f"[dump-v3] loaded {CKPT}", flush=True)

char_id = env._get_char_id()
e = env._engine
agent._curr_obs, agent._curr_info = agent._reset_envs()

# engine-side gains/limits actually used
kp, kd = e.get_obj_pd_gains(0, char_id)
kp = np.asarray(kp, dtype=np.float64)
kd = np.asarray(kd, dtype=np.float64)
tlim = np.asarray(to_np(e.get_obj_torque_limits(0, char_id)), dtype=np.float64)
print(f"[dump-v3] kp[:6] {np.round(kp[:6],1)} kd[:6] {np.round(kd[:6],1)} "
      f"tlim[:6] {np.round(tlim[:6],1)}", flush=True)

traj = dict(obs=[], root_pos=[], root_quat=[], root_vel=[],
            root_ang_vel=[], dof_pos=[], dof_vel=[], action=[],
            torque=[], q_tar=[], contacts=[])
N = 120  # 4 s @ 30 Hz
for t in range(N):
    obs = to_np(agent._curr_obs[0])
    action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
    traj["obs"].append(obs)
    traj["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    traj["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    traj["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    traj["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    traj["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    traj["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    traj["action"].append(to_np(action[0]))
    try:
        cf = e._contact_force_raw[char_id].copy()
    except Exception:
        cf = np.zeros((1, 0, 3))
    traj["contacts"].append(cf)
    nobs, r, done, info = agent._step_env(action)
    cmd = np.asarray(to_np(e._get_dof_cmd_buf()[0]), dtype=np.float64)
    q = np.asarray(traj["dof_pos"][-1], dtype=np.float64)
    qd = np.asarray(traj["dof_vel"][-1], dtype=np.float64)
    traj["q_tar"].append(cmd)
    traj["torque"].append(np.clip(kp * (cmd - q) - kd * qd,
                                  -tlim, tlim))
    agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
out = {k: torch.tensor(np.stack(v)) for k, v in traj.items() if v}
out["kp"] = torch.tensor(kp)
out["kd"] = torch.tensor(kd)
out["tlim"] = torch.tensor(tlim)
torch.save(out, os.path.join(ROOT, "output", "isaac_traj_v3.pt"))
print("[dump-v3] saved output/isaac_traj_v3.pt "
      f"({len(traj['obs'])} steps)", flush=True)
