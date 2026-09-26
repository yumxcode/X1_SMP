"""Dump Isaac state + obs after reset and after N steps, for offline
convention diffing against the MuJoCo validator."""
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
import envs.env_builder as env_builder  # isaacgym before torch
import numpy as np
import torch

env = env_builder.build_env("data/envs/amp_x1_env.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            4, "cuda:0", visualize=False, record_video=False)
char_id = env._get_char_id()
e = env._engine

def dump_state(tag):
    rp = e.get_root_pos(char_id)[0].cpu().numpy()
    rq = e.get_root_rot(char_id)[0].cpu().numpy()
    rv = e.get_root_vel(char_id)[0].cpu().numpy()
    rw = e.get_root_ang_vel(char_id)[0].cpu().numpy()
    dp = e.get_dof_pos(char_id)[0].cpu().numpy()
    dv = e.get_dof_vel(char_id)[0].cpu().numpy()
    print(f"[dump] {tag} root_pos {np.array2string(rp, precision=5)}")
    print(f"[dump] {tag} root_quat(wxyz) {np.array2string(rq, precision=5)}")
    print(f"[dump] {tag} root_vel {np.array2string(rv, precision=5)}")
    print(f"[dump] {tag} root_angvel {np.array2string(rw, precision=5)}")
    print(f"[dump] {tag} dof_pos[:8] {np.array2string(dp[:8], precision=4)}")
    print(f"[dump] {tag} dof_vel[:8] {np.array2string(dv[:8], precision=4)}")
    print(f"[dump] {tag} dof_vel_norm {np.linalg.norm(dv):.4f}", flush=True)
    return rp, rq, rv, rw, dp, dv

obs, info = env.reset()
print(f"[dump] obs_dim {obs.shape[-1]} obs[0][:14] {np.array2string(obs[0][:14].cpu().numpy(), precision=5)}", flush=True)
# ang-vel block (dims 10:13) and dof_vel block (129:158 or 158+..): print all
o0 = obs[0].cpu().numpy()
print(f"[dump] obs[0][7:13] (vel+angvel) {np.array2string(o0[7:13], precision=5)}")
dump_state("t0")

# one random-ish action step
a = torch.zeros((4, env.get_action_space().shape[0]), device="cuda:0")
obs2, r, done, info2 = env.step(a)
o1 = obs2[0].cpu().numpy()
print(f"[dump] after step obs[0][7:13] {np.array2string(o1[7:13], precision=5)}")
dump_state("t1")
print(f"[dump] obs t0 dofvel-block [158:163] {np.array2string(o0[158:163], precision=5)}")
print(f"[dump] obs t1 dofvel-block [158:163] {np.array2string(o1[158:163], precision=5)}", flush=True)
print("[dump] done", flush=True)
