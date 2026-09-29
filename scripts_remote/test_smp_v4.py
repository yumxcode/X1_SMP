"""Isaac-side SMP v4 policy evaluation: fall time / speed / height.

v4 variant of test_smp_pdx2.py: v4 env yaml (low-gain x1_v4.xml assets) +
pdx2 semantics patches. No DR at eval (AgentMode.TEST skips all patches).
Checkpoint via X1_EVAL_CKPT (default: newest smpv4* in repo data/models/smp).
"""
import os
import sys
import runpy
import subprocess
import glob

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

NUM_ENVS = 8
EP_SECONDS = 10.0
EPISODES = 3
CKPT = os.environ.get("X1_EVAL_CKPT", "")

# pdx2 semantics patches (same as run_smp_v4.py)
import engines.engine as _eng
import engines.isaac_gym_engine as _ige
_orig_modify = _ige.IsaacGymEngine._modify_control_mode_dof_props
def _keep_damping(self, control_mode, dof_props):
    kd_backup = dof_props["damping"].copy()
    _orig_modify(self, control_mode, dof_props)
    if control_mode == _eng.ControlMode.pd_explicit:
        dof_props["damping"] = kd_backup
        dof_props["stiffness"] = 0.0
_ige.IsaacGymEngine._modify_control_mode_dof_props = _keep_damping
def _calc_pdx2_torque(self):
    dof_pos = self._dof_state[..., :, 0]
    tar_dof = self._get_dof_cmd_buf()
    return self._kp_raw * (tar_dof - dof_pos)
_ige.IsaacGymEngine._calc_pd_explicit_torque = _calc_pdx2_torque
print("[v4/pdx2] eval patches applied", flush=True)

if not CKPT:
    # fixed eval name first (unambiguous), else any smpv4* ckpt
    _fixed = os.path.join(ROOT, "data", "models", "smp", "smpv4_eval.pt")
    if os.path.exists(_fixed):
        CKPT = _fixed
    else:
        cands = sorted(glob.glob(os.path.join(
            ROOT, "data", "models", "smp", "smpv4*.pt")))
        cands += sorted(glob.glob(os.path.join(
            ROOT, "upload", "**", "smpv4*.pt"), recursive=True))
        cands += sorted(glob.glob("/workspace/**/smpv4*.pt", recursive=True))
        if not cands:
            raise RuntimeError("no smpv4* ckpt found; set X1_EVAL_CKPT")
        CKPT = cands[-1]

env = env_builder.build_env("data/envs/smp_x1_env_v4.yaml",
                            "data/engines/isaac_gym_engine_pdx.yaml",
                            NUM_ENVS, "cuda:0", visualize=False,
                            record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, "cuda:0")
agent.load(CKPT)
agent.eval()
agent.set_mode(AgentMode.TEST)
print(f"[smp-eval-v4] loaded {CKPT}", flush=True)

char_id = env._get_char_id()
engine = env._engine

for ep in range(EPISODES):
    agent._curr_obs, agent._curr_info = agent._reset_envs()
    n_steps = int(EP_SECONDS * 30)
    root_z = np.zeros((n_steps, NUM_ENVS))
    root_x = np.zeros((n_steps, NUM_ENVS))
    fall_t = np.full(NUM_ENVS, np.inf)
    done_any = np.zeros(NUM_ENVS, dtype=bool)
    # IDEA-008: collect disc_obs for offline sds (maturity metric without DR)
    disc_obs_list = []
    for t in range(n_steps):
        action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
        obs, r, done, info = agent._step_env(action)
        if isinstance(info, dict) and "disc_obs" in info:
            d = info["disc_obs"]
            disc_obs_list.append(d.detach().cpu() if hasattr(d, "detach") else np.asarray(d))
        agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)
        rp = engine.get_root_pos(char_id).cpu().numpy()
        root_z[t] = rp[:, 2]
        root_x[t] = rp[:, 0]
        fell_now = (rp[:, 2] < 0.30) & (~done_any)
        fall_t[fell_now] = t / 30.0
        done_any |= fell_now
    path = np.abs(np.diff(root_x, axis=0)).sum(axis=0)
    speed = path / EP_SECONDS
    sds_str = ""
    if disc_obs_list:
        try:
            all_disc = torch.cat([torch.as_tensor(d, dtype=torch.float32, device="cuda:0")
                                  for d in disc_obs_list], dim=0)
            b = all_disc.shape[0]
            reshaped = all_disc.reshape(b, env._num_disc_obs_steps, -1)
            norm = agent._prior_model.normalize(reshaped)
            _, sds_info = agent._calc_smp_rewards(norm.reshape(b, -1))
            sds_str = f" | sds_loss_mean {float(sds_info['sds_loss_mean']):.4f}"
            sds_str += f" (n={b} frames, no-DR eval)"
        except Exception as exc:  # sds is auxiliary; never block eval
            sds_str = f" | sds collection FAILED: {exc}"
    print(f"[smp-eval-v4] ep{ep}: fall_t med {np.median(fall_t):.2f}s "
          f"(min {np.min(fall_t):.2f}) | alive>9s: "
          f"{int(np.sum(fall_t > 9.0))}/{NUM_ENVS} | "
          f"|vx| med {np.median(speed):.2f} m/s | "
          f"rootz med {np.median(root_z):.3f}{sds_str}", flush=True)

print("[smp-eval-v4] done", flush=True)
