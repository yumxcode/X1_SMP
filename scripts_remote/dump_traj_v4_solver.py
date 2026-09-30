"""E-ISAAC-SOLVER-01 (IDEA-009, SW-PREREG-002): scan the Isaac PhysX solver
precision parameter family, which training/dumps have never varied.

Background (SW-R003): v4 first-step Isaac-vs-MuJoCo residual 17.84 rad/s was
pinned on "PhysX TGS@120Hz joint solver behavior" - but TGS+4/0 iterations
is the only solver config ever exercised (isaac_gym_engine.py hardcoded).
This scan answers: is the residual a solver PRECISION artifact (H2',
fixable in-place) or an engine-family property (H2'', migration needed)?

Configs (per SW-PREREG-002, frozen):
  tgs4_0_gpu  anchor - reproduces TASK_20260929_165 conditions
  tgs4_0_cpu, tgs32_0_cpu, tgs4_8_cpu
  pgs4_0_cpu, pgs32_0_cpu, pgs4_8_cpu   (PGS needs CPU pipeline)

Each config runs in a FRESH subprocess (Isaac Gym: one sim per process),
same ckpt (data/models/smp/smpv4_eval.pt, abs18642), same env yamls, same
deterministic reset. Per config we dump:
  - 40 control steps of the policy rollout (same keys as dump_traj_v4.py)
  - zero-error probe (addendum5): q_tar pinned at q0, 8 control steps ->
    pure gravity+damping+constraint settling, no PD error path.

Driver mode (default): loop configs, spawn child subprocesses, print
per-config status + md5. Child mode: X1_SOLVER_CHILD=1 + config env vars.

Outputs: output/isaac_traj_v4_solver_{tag}.pt (SDK auto-uploads to platform).
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF = os.path.abspath(__file__)   # absolute BEFORE chdir (gm-run may use a
                                   # relative script path; child re-exec needs it)
os.chdir(ROOT)

CONFIGS = [
    ("tgs4_0_gpu", "1", 4, 0, "gpu"),
    ("tgs4_0_cpu", "1", 4, 0, "cpu"),
    ("tgs32_0_cpu", "1", 32, 0, "cpu"),
    ("tgs4_8_cpu", "1", 4, 8, "cpu"),
    ("pgs4_0_cpu", "0", 4, 0, "cpu"),
    ("pgs32_0_cpu", "0", 32, 0, "cpu"),
    ("pgs4_8_cpu", "0", 4, 8, "cpu"),
]
N_ROLLOUT = 40
N_PROBE = 8


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def run_child(tag, solver_type, pos_iter, vel_iter, pipeline):
    env = dict(os.environ)
    env.update({
        "X1_SOLVER_CHILD": "1",
        "X1_SOLVER_TYPE": solver_type,
        "X1_POS_ITER": str(pos_iter),
        "X1_VEL_ITER": str(vel_iter),
        "X1_PIPELINE": pipeline,
        "X1_OUT_TAG": tag,
    })
    print(f"[solver-scan] === {tag}: solver_type={solver_type} "
          f"pos={pos_iter} vel={vel_iter} pipeline={pipeline} ===", flush=True)
    proc = subprocess.run([sys.executable, SELF], env=env,
                          capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    # relay key lines so they reach the platform log
    tail = [ln for ln in out.splitlines()
            if ("[solver-scan]" in ln or "[v4-dump]" in ln
                or "Error" in ln or "error" in ln or "Traceback" in ln)]
    for ln in tail[-40:]:
        print(ln, flush=True)
    if proc.returncode != 0:
        print(f"[solver-scan] child {tag} failed; full output tail:", flush=True)
        for ln in out.splitlines()[-25:]:
            print(f"  {ln}", flush=True)
    out_path = os.path.join(ROOT, "output", f"isaac_traj_v4_solver_{tag}.pt")
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    md5 = _md5(out_path) if os.path.exists(out_path) else "missing"
    print(f"[solver-scan] {tag}: {status} | dump md5 {md5}", flush=True)
    return proc.returncode == 0, md5


if os.environ.get("X1_SOLVER_CHILD", "") != "1":
    # ---------------- driver ----------------
    print("[solver-scan] driver start; configs:", flush=True)
    results = {}
    for tag, st, pos, vel, pipe in CONFIGS:
        ok, md5 = run_child(tag, st, pos, vel, pipe)
        results[tag] = (ok, md5)
    print("[solver-scan] SUMMARY", flush=True)
    for tag, (ok, md5) in results.items():
        print(f"[solver-scan]   {tag}: {'OK' if ok else 'FAIL'} md5 {md5}",
              flush=True)
    sys.exit(0 if all(ok for ok, _ in results.values()) else 1)

# ================= child mode =================
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
    """Deterministic reset across configs AND devices (r2 lesson).

    r1 scan (TASK_20260930_032) obs0 self-check failed: sample_motions
    uses torch.multinomial on the DEVICE tensor (cpu/cuda RNG streams
    differ even under identical seeds) and sample_time uses torch.rand on
    device. Pin both to (motion 0, time 0) so every config starts from the
    exact same ref state; the character always resets to the fixed home
    pose (char_env._reset_char), making the full reset deterministic.
    """
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

print(f"[solver-scan] child {TAG}: solver_type={SOLVER_TYPE} "
      f"pos={POS_ITER} vel={VEL_ITER} pipeline={PIPELINE} device={DEVICE}",
      flush=True)
for p in ("data/assets/x1/x1_v4.xml", "data/assets/x1/x1_sim_v4.xml",
          "data/envs/smp_x1_env_v4.yaml"):
    print(f"[solver-scan] md5 {p} {_md5(os.path.join(ROOT, p))}", flush=True)

CKPT = os.environ.get("X1_DUMP_CKPT", "")
if not CKPT:
    CKPT = os.path.join(ROOT, "data", "models", "smp", "smpv4_eval.pt")
    assert os.path.exists(CKPT), "smpv4_eval.pt missing"
print(f"[solver-scan] ckpt: {CKPT} (md5 {_md5(CKPT)})", flush=True)

env = env_builder.build_env("data/envs/smp_x1_env_v4.yaml",
                            "data/engines/isaac_gym_engine_pdx.yaml",
                            1, DEVICE, visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, DEVICE)
agent.load(CKPT)
agent.eval()
agent.set_mode(AgentMode.TEST)

char_id = env._get_char_id()
e = env._engine
agent._curr_obs, agent._curr_info = agent._reset_envs()


def to_np(x):
    # MUST copy: on CPU pipeline the engine tensors are shared gymtorch
    # views; .cpu() on an already-CPU tensor is a no-op and .numpy()
    # returns a LIVE view - without copy() every "snapshot" aliases the
    # same buffer and the whole trajectory collapses to the final state
    # (r1/r3 CPU-config dumps were corrupted this way; GPU configs were
    # fine because .cpu() copies across devices).
    try:
        return torch.as_tensor(x).detach().cpu().numpy().copy()
    except Exception:
        return np.array(x)


kp, kd = e.get_obj_pd_gains(0, char_id)
kp = np.asarray(kp, dtype=np.float64)
kd = np.asarray(kd, dtype=np.float64)
tlim = np.asarray(to_np(e.get_obj_torque_limits(0, char_id)), dtype=np.float64)

traj = dict(obs=[], root_pos=[], root_quat=[], root_vel=[],
            root_ang_vel=[], dof_pos=[], dof_vel=[], action=[], q_tar=[],
            done=[])
for t in range(N_ROLLOUT):
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
    traj["done"].append(np.asarray(to_np(done), dtype=np.float64).reshape(-1))
    agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

print(f"[solver-scan] reset: motion_id {int(to_np(env._motion_ids[0]))} "
      f"time_offset {float(to_np(env._motion_time_offsets[0])):.4f}", flush=True)

# ---- zero-error probe (addendum5): q_tar pinned at q0 ----
agent._curr_obs, agent._curr_info = agent._reset_envs()
q0 = np.asarray(to_np(e.get_dof_pos(char_id)[0]), dtype=np.float64).copy()
pinned = torch.tensor(q0, dtype=torch.float32, device=DEVICE).reshape(1, -1)
_orig_cmd = e._get_dof_cmd_buf
e._get_dof_cmd_buf = lambda: pinned  # instance-level pin (engine + pdx2 calc)
probe = dict(root_pos=[], root_quat=[], root_vel=[], root_ang_vel=[],
             dof_pos=[], dof_vel=[], q0=q0)
act_dim = int(to_np(action).shape[-1])
zero_act = torch.zeros(1, act_dim, dtype=torch.float32, device=DEVICE)
for t in range(N_PROBE):
    probe["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    probe["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    probe["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    probe["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    probe["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    probe["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    _ = agent._step_env(zero_act)  # action ignored: cmd buf pinned
e._get_dof_cmd_buf = _orig_cmd

os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
out = {k: torch.tensor(np.stack(v)) for k, v in traj.items() if v}
out["kp"] = torch.tensor(kp)
out["kd"] = torch.tensor(kd)
out["tlim"] = torch.tensor(tlim)
out["solver_config"] = dict(tag=TAG, solver_type=SOLVER_TYPE,
                            num_position_iterations=POS_ITER,
                            num_velocity_iterations=VEL_ITER,
                            pipeline=PIPELINE, device=DEVICE)
out["probe"] = {k: torch.tensor(np.stack(v)) for k, v in probe.items()}
out["obs0_hash"] = hashlib.md5(
    np.asarray(traj["obs"][0], dtype=np.float64).tobytes()).hexdigest()[:12]
out_path = os.path.join(ROOT, "output", f"isaac_traj_v4_solver_{TAG}.pt")
torch.save(out, out_path)
print(f"[solver-scan] saved {out_path} ({N_ROLLOUT} steps + probe {N_PROBE}) "
      f"md5 {_md5(out_path)} obs0_hash {out['obs0_hash']}", flush=True)
