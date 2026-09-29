"""Dump the SMP v4 (low-gain assets + pdx2 semantics) policy rollout in
Isaac for one-to-one MuJoCo replay diffing.

Purpose (SW-R001 follow-up, E-v4-RESID-02): the v3-era first-step engine
residual (2.4-3.9 rad/s max, reproduced locally as 3.89 on this dump's
predecessor) was measured under v3 HIGH gains where hip tau saturates at
180 N*m. v4 lowered kp/kd so tau stays unsaturated; the v4 engine-pair
residual has never been measured. This dump provides the Isaac side.

Same semantics as dump_traj_pdx2.py: explicit clip(kp*(tar-q)) + implicit
kd; NO DR (AgentMode.TEST); env yaml smp_x1_env_v4.yaml (x1_v4.xml assets).

Checkpoint resolution: X1_DUMP_CKPT env var, else fixed
data/models/smp/smpv4_eval.pt (upload-mount preferred, exact-basename
priority - same convention as test_smp_v4.py).
"""
import os
import sys
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


def _apply_pdx2_patches():
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
    print("[v4-dump] pdx2 patches applied", flush=True)


_apply_pdx2_patches()
import envs.env_builder as env_builder  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

# early verification channel (which code/assets is this run using?)
import hashlib

def _md5(p):
    return hashlib.md5(open(os.path.join(ROOT, p), "rb").read()).hexdigest()[:12]

print(f"[v4-dump] commit-marker files: x1_v4.xml md5 {_md5('data/assets/x1/x1_v4.xml')}"
      f" | x1_sim_v4.xml md5 {_md5('data/assets/x1/x1_sim_v4.xml')}"
      f" | env yaml md5 {_md5('data/envs/smp_x1_env_v4.yaml')}", flush=True)

CKPT = os.environ.get("X1_DUMP_CKPT", "")
if not CKPT:
    fixed = os.path.join(ROOT, "data", "models", "smp", "smpv4_eval.pt")
    if os.path.exists(fixed):
        CKPT = fixed
    else:
        cands = (sorted(glob.glob(os.path.join(
            ROOT, "data", "models", "smp", "smpv4*.pt")))
            + sorted(glob.glob(os.path.join(
                ROOT, "upload", "**", "smpv4*.pt"), recursive=True))
            + sorted(glob.glob("/workspace/**/smpv4*.pt", recursive=True)))
        if not cands:
            raise RuntimeError("no smpv4* ckpt; set X1_DUMP_CKPT")
        CKPT = cands[-1]
print(f"[v4-dump] ckpt: {CKPT} (md5 "
      f"{hashlib.md5(open(CKPT,'rb').read()).hexdigest()[:12]})", flush=True)

env = env_builder.build_env("data/envs/smp_x1_env_v4.yaml",
                            "data/engines/isaac_gym_engine_pdx.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, "cuda:0")
agent.load(CKPT)
agent.eval()
agent.set_mode(AgentMode.TEST)

char_id = env._get_char_id()
e = env._engine
agent._curr_obs, agent._curr_info = agent._reset_envs()


def to_np(x):
    try:
        return torch.as_tensor(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


kp, kd = e.get_obj_pd_gains(0, char_id)
kp = np.asarray(kp, dtype=np.float64)
kd = np.asarray(kd, dtype=np.float64)
tlim = np.asarray(to_np(e.get_obj_torque_limits(0, char_id)), dtype=np.float64)
print(f"[v4-dump] kp[:6] {np.round(kp[:6],1)} kd[:6] {np.round(kd[:6],1)} "
      f"tlim[:6] {np.round(tlim[:6],1)}", flush=True)

traj = dict(obs=[], root_pos=[], root_quat=[], root_vel=[],
            root_ang_vel=[], dof_pos=[], dof_vel=[], action=[], q_tar=[])
N = 120
for t in range(N):
    action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
    traj["obs"].append(to_np(agent._curr_obs[0]))
    traj["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    traj["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    traj["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    traj["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    traj["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    traj["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    traj["action"].append(to_np(action[0]))
    nobs, r, done, info = agent._step_env(action)
    traj["q_tar"].append(np.asarray(to_np(e._get_dof_cmd_buf()[0]),
                                    dtype=np.float64))
    agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
out = {k: torch.tensor(np.stack(v)) for k, v in traj.items() if v}
out["kp"] = torch.tensor(kp)
out["kd"] = torch.tensor(kd)
out["tlim"] = torch.tensor(tlim)
torch.save(out, os.path.join(ROOT, "output", "isaac_traj_v4.pt"))
print("[v4-dump] saved output/isaac_traj_v4.pt "
      f"({N} steps)", flush=True)
