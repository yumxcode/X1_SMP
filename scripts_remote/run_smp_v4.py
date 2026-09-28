"""SMP v4: LOW-GAIN asset (x1_v4.xml) + pdx2 semantics + FULL dynamics-side
domain randomization. This is the X1_stable-recipe transfer recommended by the
cross-project audit (FINAL_REPORT.md 4.3):

  1. Gains 375-450/37-45 -> 120-150/3-8 (legs/lumbar), ankle 200/20 -> 50/1.
     The measured 2.4 rad/s TGS-vs-MuJoCo solver residual now maps to <=6 N*m
     of disturbance instead of 108 N*m (kd 45 -> 4-8).
  2. Dynamics DR per episode (X1_stable recipe):
       - motor strength  x U[0.8, 1.2]   (kp via engine._kp_raw AND kd via
         dof_props damping — NOTE: pdx2r's +-15% gain randomization only
         touched set_actor_dof_properties['stiffness'], which pdx2 NEVER
         reads: the explicit kp comes from engine._kp_raw. So its kp
         randomization silently did nothing; only kd was randomized. Fixed
         here by scaling _kp_raw directly.)
       - joint friction   U[0.0, 1.0] N.m per dof (covers the MuJoCo harness
         which runs frictionless: 0 is in-range)
       - motor offset     U[-0.035, +0.035] rad per dof (injected in the
         explicit torque: tau = kp*(tar + off - q))
       - link mass        x U[0.9, 1.1], base_link +- U[-2, 2] kg
  3. Kept from pdx2r: obs noise 0.01, 1-step action latency, 25 N pushes.
     Control freq stays 30 Hz (SMP prior is 30 fps — raising it would need a
     prior retrain; deferred to v4b if v4 falls short).

Warm-start: platform-mounted smppdx_it*.pt (pdx2 family ckpt).
Early verification: prints commit + asset md5 + first-reset DR samples.
"""
import os
import sys
import runpy
import subprocess
import threading
import time
import glob
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# ---------------------------------------------------------------------------
# pdx2 semantics patch (unchanged from run_smp_pdx2r.py):
#   torque = kp * (tar + offset - q)  explicit, clipped +-tlim by engine
#   dof_props["damping"] = kd          PhysX implicit joint damping
#   -> semantically identical to the MuJoCo sim2sim harness
# ---------------------------------------------------------------------------
def _apply_pdx2_patches():
    import engines.engine as _eng
    import engines.isaac_gym_engine as _ige

    _orig_modify = _ige.IsaacGymEngine._modify_control_mode_dof_props

    def _keep_damping(self, control_mode, dof_props):
        kd_backup = dof_props["damping"].copy()
        _orig_modify(self, control_mode, dof_props)
        if control_mode == _eng.ControlMode.pd_explicit:
            dof_props["damping"] = kd_backup   # implicit damping stays
            dof_props["stiffness"] = 0.0       # kp applied explicitly below

    _ige.IsaacGymEngine._modify_control_mode_dof_props = _keep_damping

    def _calc_pdx2_torque(self):
        dof_pos = self._dof_state[..., :, 0]
        tar_dof = self._get_dof_cmd_buf()
        off = getattr(self, "_v4_off", None)
        if off is None:
            off = torch_zeros_like(self._kp_raw)
            self._v4_off = off
        return self._kp_raw * (tar_dof + off - dof_pos)  # NO explicit kd*qd

    def torch_zeros_like(x):
        import torch
        return torch.zeros_like(x)

    _ige.IsaacGymEngine._calc_pd_explicit_torque = _calc_pdx2_torque
    print("[v4/pdx2] patches applied: explicit kp(+offset) + implicit kd",
          flush=True)


# ---- early verification channel: WHICH assets/code is this run using?
import hashlib as _hl
def _md5(p):
    try:
        return _hl.md5(open(p, "rb").read()).hexdigest()[:12]
    except Exception as e:
        return f"ERR {e}"
import subprocess as _sp
try:
    _commit = _sp.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               stderr=_sp.DEVNULL).decode().strip()[:8]
except Exception:
    _commit = "no-git"
print(f"[verify] commit {_commit}"
      f" | x1_v4.xml md5 {_md5('data/assets/x1/x1_v4.xml')}"
      f" | x1_sim_v4.xml md5 {_md5('data/assets/x1/x1_sim_v4.xml')}"
      f" | env yaml md5 {_md5('data/envs/smp_x1_env_v4.yaml')}"
      f" | prefix {os.environ.get('X1_EXPORT_PREFIX', '')}", flush=True)
# mimickit modules must be importable BEFORE the patches below
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)

_apply_pdx2_patches()
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

# ------------------------------- DR configuration ---------------------------
OBS_NOISE = 0.01      # gaussian on observations
LATENCY_STEPS = 1     # 33 ms action hold at 30 Hz
PUSH_PROB = 0.004     # per control step, per env
PUSH_FORCE = 25.0     # N horizontal
PUSH_STEPS = 3        # duration in control steps

STRENGTH_RAND = float(os.environ.get("X1_STRENGTH_RAND", "0.20"))  # +-20%
FRICTION_LO, FRICTION_HI = 0.0, 1.0     # N.m joint friction per dof
OFFSET_RAND = 0.035                    # rad motor offset per dof
MASS_RAND = 0.10                       # links x U[0.9, 1.1]
BASE_MASS_DELTA = 2.0                  # base_link +- kg
DR_LOG_EVERY = 50                      # log DR samples every N resets


def start_checkpoint_exporter(prefix):
    """Top-level output/*.pt with unique names = the live upload channel."""
    seen = set()

    def loop():
        while True:
            try:
                files = sorted(glob.glob(os.path.join(
                    ROOT, "output", "int_models", "model_*.pt")),
                    key=lambda f: int(os.path.basename(f)[6:-3]))
                for f in files:
                    base = os.path.basename(f)
                    if base in seen:
                        continue
                    seen.add(base)
                    it = int(base[6:-3])
                    dst = os.path.join(ROOT, "output", f"{prefix}_it{it}.pt")
                    shutil.copyfile(f, dst)
                    print(f"[exporter] {base} -> {os.path.basename(dst)}",
                          flush=True)
            except Exception as e:
                print(f"[exporter] error: {e}", flush=True)
            time.sleep(60)

    threading.Thread(target=loop, daemon=True).start()


def apply_robustness_patches():
    """Monkey-wrap agent decision + env step for noise/latency/pushes/DR."""
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import torch
    import numpy as np
    from learning.base_agent import AgentMode

    orig_build_env = env_builder.build_env
    orig_build_agent = agent_builder.build_agent
    state = {"env": None, "agent": None, "prev_a": None, "push": {},
             "n_resets": 0}

    def build_env(*a, **kw):
        env = orig_build_env(*a, **kw)
        state["env"] = env
        return env

    def build_agent(*a, **kw):
        agent = orig_build_agent(*a, **kw)

        orig_decide = agent._decide_action
        orig_step = agent._step_env

        def noisy_decide(obs, info):
            return orig_decide(obs, info)

        def pushy_step(action):
            env = state["env"]
            agent = state["agent"]
            if agent._mode == AgentMode.TRAIN:
                import numpy as np
                n = env.get_num_envs()
                for e in range(n):
                    pid = (e, state.get("step", 0))
                    cur = state["push"].get(e)
                    if cur is not None and cur[1] > 0:
                        f = cur[0]
                        env._engine.set_body_forces([e], 0, 0, f)
                        state["push"][e] = (f, cur[1] - 1)
                    elif cur is not None and cur[1] == 0:
                        env._engine.set_body_forces(
                            [e], 0, 0,
                            torch.zeros(3, device=agent._device))
                        state["push"][e] = None
                    if (cur is None or cur[1] <= 0) and \
                            torch.rand(1).item() < PUSH_PROB:
                        ang = np.random.rand() * 2 * 3.14159265
                        f = torch.tensor([PUSH_FORCE * float(np.cos(ang)),
                                          PUSH_FORCE * float(np.sin(ang)),
                                          0.0], device=agent._device)
                        env._engine.set_body_forces([e], 0, 0, f)
                        state["push"][e] = (f, PUSH_STEPS)
                state["step"] = state.get("step", 0) + 1
                # 1-step action latency (33 ms): hold previous action
                prev = state["prev_a"]
                state["prev_a"] = action
                if prev is not None and prev.shape == action.shape:
                    action = prev
            obs, r, done, info = orig_step(action)
            if agent._mode == AgentMode.TRAIN:
                obs = obs + torch.randn_like(obs) * OBS_NOISE
            return obs, r, done, info

        orig_reset = agent._reset_envs

        def dr_reset(env_ids=None):
            if (agent._mode == AgentMode.TRAIN and env_ids is not None
                    and len(env_ids) > 0):
                try:
                    _apply_dr(env_ids)
                except Exception as ex:
                    import traceback
                    print(f"[v4/DR] error: {ex}", flush=True)
                    traceback.print_exc()
            return orig_reset(env_ids)

        def noisy_reset(*args, **kwargs):
            obs, info = dr_reset(*args, **kwargs)
            if agent._mode == AgentMode.TRAIN:
                obs = obs + torch.randn_like(obs) * OBS_NOISE
            return obs, info

        def _apply_dr(env_ids):
            """Per-episode dynamics randomization on the resetting envs."""
            env_ = state["env"]
            e = env_._engine
            gym = e._gym
            char_id = env_._get_char_id()
            dev = e._kp_raw.device

            if "base_kp" not in state:
                state["base_kp"] = e._kp_raw.clone()
                if getattr(e, "_v4_off", None) is None:
                    e._v4_off = torch.zeros_like(e._kp_raw)
                props0 = gym.get_actor_dof_properties(
                    e.get_env(0), char_id)
                state["base_kd"] = np.asarray(
                    props0["damping"], dtype=np.float32).copy()
                rb0 = gym.get_actor_rigid_body_properties(
                    e.get_env(0), char_id)
                rb_names = gym.get_actor_rigid_body_names(e.get_env(0),
                                                          char_id)
                state["rb0"] = np.array([rb.mass for rb in rb0],
                                        dtype=np.float64)
                state["rb_names"] = list(rb_names)
                state["base_link_i"] = (state["rb_names"].index("base_link")
                                        if "base_link" in state["rb_names"]
                                        else None)
                print(f"[v4/DR] base kp[:6] {state['base_kp'][0][:6]}"
                      f" base kd[:6] {state['base_kd'][:6]}"
                      f" bodies {len(rb0)}", flush=True)

            ids = env_ids.detach().cpu().numpy().tolist()
            n_dof = e._kp_raw.shape[1]
            for env_id in ids:
                # 1) motor strength: scale the EXPLICIT kp (engine._kp_raw
                #    is what the pdx2 torque formula reads — effective by
                #    construction) and the IMPLICIT kd (dof_props damping)
                s = float(np.random.uniform(1.0 - STRENGTH_RAND,
                                            1.0 + STRENGTH_RAND))
                e._kp_raw[env_id] = state["base_kp"][env_id] * s
                # 2) motor offset (persistent per episode, used in torque)
                e._v4_off[env_id] = torch.tensor(
                    np.random.uniform(-OFFSET_RAND, OFFSET_RAND, n_dof),
                    dtype=torch.float32, device=dev)
                env_ptr = e.get_env(env_id)
                props = gym.get_actor_dof_properties(env_ptr, char_id)
                props["stiffness"] = 0.0                # pdx2: explicit kp
                props["damping"] = state["base_kd"] * s  # implicit kd
                # 3) joint friction per dof
                props["friction"] = np.random.uniform(
                    FRICTION_LO, FRICTION_HI, n_dof).astype(np.float32)
                gym.set_actor_dof_properties(env_ptr, char_id, props)
                # 4) link masses (keep original inertia tensors; mass-only
                #    perturbation is the DR signal, recompute off)
                f = float(np.random.uniform(1.0 - MASS_RAND,
                                            1.0 + MASS_RAND))
                db = float(np.random.uniform(-BASE_MASS_DELTA,
                                             BASE_MASS_DELTA))
                rb = gym.get_actor_rigid_body_properties(env_ptr, char_id)
                for j, rbp in enumerate(rb):
                    if j == state["base_link_i"]:
                        rbp.mass = float(state["rb0"][j]) + db
                    else:
                        rbp.mass = float(state["rb0"][j]) * f
                gym.set_actor_rigid_body_properties(env_ptr, char_id, rb,
                                                    False)
            state["n_resets"] += len(ids)
            if state["n_resets"] <= 3 * len(ids) or \
                    state["n_resets"] % (DR_LOG_EVERY * max(len(ids), 1)) \
                    < len(ids):
                j = ids[0]
                print(f"[v4/DR] env{j}: kp_scale {s:.3f} "
                      f"friction[:3] {props['friction'][:3]} "
                      f"off[:3] {e._v4_off[j][:3].tolist()} "
                      f"mass_f {f:.3f} base_dkg {db:+.2f}", flush=True)

        agent._decide_action = noisy_decide
        agent._step_env = pushy_step
        agent._reset_envs = noisy_reset
        return agent

    env_builder.build_env = build_env
    agent_builder.build_agent = build_agent


start_checkpoint_exporter(os.environ.get("X1_EXPORT_PREFIX", "smpv4"))
apply_robustness_patches()

sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine_pdx.yaml",
            "--env_config", "data/envs/smp_x1_env_v4.yaml",
            "--agent_config", "data/agents/smp_x1_agent_v3.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", os.environ.get("X1_MAX_SAMPLES", "500000000")]
_mf = os.environ.get("X1_MODEL_FILE", "X1_SMP/upload/**/smppdx_it*.pt")
if _mf:
    import glob as _glob
    pats = [_mf] + [os.path.join(ROOT, _f)
                    for _f in (_mf, _mf.lstrip("./"))]
    pats += [os.path.join("/workspace", "**", os.path.basename(_mf)),
             os.path.join(ROOT, "**", os.path.basename(_mf))]
    cands = []
    for p in pats:
        cands += _glob.glob(p, recursive=True)
    cands = sorted(set(cands), key=os.path.getmtime)
    if not cands:
        cands = sorted(_glob.glob(os.path.join(
            "/workspace", "**", os.path.basename(_mf)), recursive=True),
            key=os.path.getmtime)
    if not cands:
        raise RuntimeError(f"X1_MODEL_FILE matched nothing; tried {pats}")
    sys.argv += ["--model_file", cands[-1]]
    print(f"[resume] loading agent weights: {cands[-1]}", flush=True)
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")

try:
    src = os.path.join(ROOT, "output", "model.pt")
    dst = os.path.join(ROOT, "output", "smp_v4_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final model -> output/smp_v4_final.pt", flush=True)
        time.sleep(90)
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
