"""Diagnose CPU-pipeline state readback staleness after env reset (r3 scan).

Observed (TASK_20260930_034 analysis): reset is deterministic (probe q0
bitwise identical across configs/devices), but the FIRST recorded rollout
state (root_pos y off by 0.25 m, dof_pos off by 0.36 rad) differs between
CPU and GPU pipeline configs. This probe pinpoints WHEN the readback
diverges: (A) right after reset, (B) immediate second read, (C) after
agent._decide_action, (D) after agent._step_env.
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF = os.path.abspath(__file__)
os.chdir(ROOT)

CONFIGS = [("tgs4_0_gpu", "1", 4, 0, "gpu"),
           ("tgs4_0_cpu", "1", 4, 0, "cpu"),
           ("pgs4_0_cpu", "0", 4, 0, "cpu")]


def run_child(tag, solver_type, pos_iter, vel_iter, pipeline):
    env = dict(os.environ)
    env.update({"X1_SOLVER_CHILD": "1", "X1_SOLVER_TYPE": solver_type,
                "X1_POS_ITER": str(pos_iter), "X1_VEL_ITER": str(vel_iter),
                "X1_PIPELINE": pipeline, "X1_OUT_TAG": tag})
    print(f"[diag] === {tag} ===", flush=True)
    proc = subprocess.run([sys.executable, SELF], env=env,
                          capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    for ln in out.splitlines():
        if "[diag]" in ln or "Traceback" in ln or "Error" in ln:
            print(ln, flush=True)
    if proc.returncode != 0:
        print(f"[diag] child {tag} FAILED rc={proc.returncode}; tail:", flush=True)
        for ln in out.splitlines()[-25:]:
            print(f"  {ln}", flush=True)


if os.environ.get("X1_SOLVER_CHILD", "") != "1":
    for tag, st, pos, vel, pipe in CONFIGS:
        run_child(tag, st, pos, vel, pipe)
    sys.exit(0)

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

TAG = os.environ["X1_OUT_TAG"]
SOLVER_TYPE = int(os.environ["X1_SOLVER_TYPE"])
POS_ITER = int(os.environ["X1_POS_ITER"])
VEL_ITER = int(os.environ["X1_VEL_ITER"])
PIPELINE = os.environ["X1_PIPELINE"]
DEVICE = "cuda:0" if PIPELINE == "gpu" else "cpu"


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


def _apply_solver_patch():
    import engines.isaac_gym_engine as _ige
    _orig_bsp = _ige.IsaacGymEngine._build_sim_params

    def _bsp(self, sim_timestep):
        sp = _orig_bsp(self, sim_timestep)
        sp.physx.solver_type = SOLVER_TYPE
        sp.physx.num_position_iterations = POS_ITER
        sp.physx.num_velocity_iterations = VEL_ITER
        if PIPELINE == "cpu":
            sp.physx.use_gpu = False
            sp.use_gpu_pipeline = False
        return sp
    _ige.IsaacGymEngine._build_sim_params = _bsp


def _apply_fixed_reset_patch():
    import envs.deepmimic_env as _de

    def _fixed_sample(self, n):
        ids = torch.zeros(n, dtype=torch.long, device=self._device)
        times = torch.zeros(n, dtype=torch.float, device=self._device)
        return ids, times
    _de.DeepMimicEnv._sample_motion_times = _fixed_sample


_apply_pdx2_patches()
_apply_solver_patch()
_apply_fixed_reset_patch()
import envs.env_builder as env_builder  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

env = env_builder.build_env("data/envs/smp_x1_env_v4.yaml",
                            "data/engines/isaac_gym_engine_pdx.yaml",
                            1, DEVICE, visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, DEVICE)
agent.load(os.path.join(ROOT, "data", "models", "smp", "smpv4_eval.pt"))
agent.eval()
agent.set_mode(AgentMode.TEST)

char_id = env._get_char_id()
e = env._engine


def snap():
    return dict(root=e.get_root_pos(char_id)[0].detach().cpu().numpy().copy(),
                dof=e.get_dof_pos(char_id)[0].detach().cpu().numpy().copy(),
                dofvel=e.get_dof_vel(char_id)[0].detach().cpu().numpy().copy())


def show(label, s):
    print(f"[diag] {TAG} {label}: root {np.round(s['root'], 4)} "
          f"| dof[:5] {np.round(s['dof'][:5], 4)} "
          f"| |dofvel| max {np.abs(s['dofvel']).max():.4f}", flush=True)


agent._curr_obs, agent._curr_info = agent._reset_envs()
A = snap()
show("A after-reset", A)
B = snap()
show("B second-read", B)
action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
C = snap()
show("C post-decide", C)
obs, r, done, info = agent._step_env(action)
D = snap()
show("D post-step  ", D)

# ref state the reset should have written (from env buffers)
ref_root = env._ref_root_pos[0].detach().cpu().numpy()
ref_dof = env._ref_dof_pos[0].detach().cpu().numpy()
print(f"[diag] {TAG} ref_root  {np.round(ref_root, 4)}", flush=True)
print(f"[diag] {TAG} ref_dof[:5] {np.round(ref_dof[:5], 4)}", flush=True)
print(f"[diag] {TAG} A-root vs ref_root max diff "
      f"{np.max(np.abs(A['root'] - ref_root)):.4f} | A-dof vs ref_dof max diff "
      f"{np.max(np.abs(A['dof'] - ref_dof)):.4f}", flush=True)
