"""SMP v3-pdx: EXPLICIT PD drive in Isaac (pd_explicit engine mode).

Isaac computes tau = clip(kp*(tar-q) - kd*qd, +-tlim) per physics substep
under DOF_MODE_EFFORT — semantically identical to the MuJoCo sim2sim
harness. Warm-started from the fixed-asset v3b policy.
(v2 history:) robustness-regularized training for MuJoCo sim2sim transfer.

Sim2sim diagnosis (2026-09-26): the v9 policy runs in Isaac (8/8 x 10s,
~1 m/s) but falls in MuJoCo within 2s despite bit-exact obs replication.
Root cause: engine-level actuation/contact differences (Isaac-side dead
right_ankle_roll artifact + contact resolution) concentrate at the ankles.
Fix: train with observation noise, action noise, one-step action latency,
and stochastic root pushes so the policy cannot rely on fragile
engine-specific equilibria.

Patches are monkey-wrapped around the agent (no repo core edits)."""
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
# pdx2 SEMANTICS FIX (measured 2026-09-28, dump TASK_20260928_018):
# stock pd_explicit computes FULL explicit PD (kp*de - kd*qd) per substep
# with damping zeroed -> numerically UNSTABLE on X1 (first-step dof_vel
# 46 rad/s oscillation; small-inertia joints, kd*dt/m ~ 15). The MuJoCo
# sim2sim harness uses: ctrl = clip(kp*de, +-tlim) EXPLICIT + dof_damping
# = kd INTEGRATED IMPLICITLY. Align Isaac to exactly that:
#   * torque = kp*(tar - q)          (explicit, clipped +-tlim by engine)
#   * dof_props["damping"] = kd      (PhysX implicit joint damping)
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
        return self._kp_raw * (tar_dof - dof_pos)   # NO explicit kd*qd

    _ige.IsaacGymEngine._calc_pd_explicit_torque = _calc_pdx2_torque
    print("[pdx2] patches applied: explicit kp + implicit kd damping",
          flush=True)

_apply_pdx2_patches()


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
print(f"[verify] commit {_commit} | x1.xml md5 {_md5('data/assets/x1/x1.xml')}"
      f" | x1_sim.xml md5 {_md5('data/assets/x1/x1_sim.xml')}"
      f" | env {os.environ.get('X1_EXPORT_PREFIX', '')}", flush=True)
# mimickit modules must be importable BEFORE the robustness patches below
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
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

OBS_NOISE = 0.01      # rad / m-scale gaussian on observations
ACT_NOISE = 0.03      # rad gaussian on actions
LATENCY_STEPS = 1     # 30 ms action hold
PUSH_PROB = 0.004     # per control step, per env
PUSH_FORCE = 25.0     # N horizontal
PUSH_STEPS = 3        # duration in control steps
GAIN_RAND = float(os.environ.get("X1_GAIN_RAND", "0.0"))
# ^ per-episode uniform gain factor U[1-GAIN_RAND, 1+GAIN_RAND] applied to
# BOTH kp and kd (preserves the damping ratio) via gymapi dof_props on each
# resetting env. Rationale: same-generation engine-diff experiments proved a
# persistent first-control-step dof_vel divergence (~3.3 rad/s, contact-free)
# between PhysX implicit PD and any MuJoCo parametric variant (kp/kd
# scaling, armature, servo, solref sweeps all rejected). Training the policy
# to be robust across a ±30% gain band is the surviving engineering route.


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
    """Monkey-wrap agent decision + env step for noise/latency/pushes."""
    # isaacgym (pulled in by env_builder) MUST be imported before torch
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import torch
    from learning.base_agent import AgentMode

    orig_build_env = env_builder.build_env
    orig_build_agent = agent_builder.build_agent
    state = {"env": None, "agent": None, "prev_a": None, "push": {}}

    def build_env(*a, **kw):
        env = orig_build_env(*a, **kw)
        state["env"] = env
        return env

    def build_agent(*a, **kw):
        agent = orig_build_agent(*a, **kw)
        state["agent"] = agent

        orig_decide = agent._decide_action
        orig_step = agent._step_env

        def noisy_decide(obs, info):
            # obs noise lives in the obs stream (patched at _step_env /
            # _reset_envs); decision and logged log_prob stay consistent.
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
            obs, r, done, info = orig_step(action)
            if agent._mode == AgentMode.TRAIN:
                obs = obs + torch.randn_like(obs) * OBS_NOISE
            return obs, r, done, info

        orig_reset = agent._reset_envs

        # per-episode gain randomization (C-route): on each resetting env,
        # rescale the engine's kp/kd by a common factor in
        # [1-GAIN_RAND, 1+GAIN_RAND]. Uses gymapi per-actor dof_props
        # (only ~num_resets calls per control step — cheap at 4096 envs).
        def gainrand_reset(env_ids=None):
            if (agent._mode == AgentMode.TRAIN and GAIN_RAND > 0
                    and env_ids is not None and len(env_ids) > 0):
                try:
                    env_ = state["env"]
                    e = env_._engine
                    gym = e._gym
                    char_id = env_._get_char_id()
                    if "base_kp" not in state:
                        import numpy as _np
                        kp0, kd0 = e.get_obj_pd_gains(0, char_id)
                        state["base_kp"] = _np.asarray(kp0, dtype=_np.float32)
                        state["base_kd"] = _np.asarray(kd0, dtype=_np.float32)
                        print(f"[gainrand] base kp[:6] "
                              f"{state['base_kp'][:6]}", flush=True)
                    import numpy as _np
                    ids = env_ids.detach().cpu().numpy().tolist()
                    for env_id in ids:
                        env_ptr = e.get_env(env_id)
                        props = gym.get_actor_dof_properties(env_ptr, char_id)
                        s = float(_np.random.uniform(1.0 - GAIN_RAND,
                                                     1.0 + GAIN_RAND))
                        props["stiffness"] = (state["base_kp"] * s)
                        props["damping"] = (state["base_kd"] * s)
                        gym.set_actor_dof_properties(env_ptr, char_id, props)
                except Exception as ex:
                    print(f"[gainrand] error: {ex}", flush=True)
            return orig_reset(env_ids)

        def noisy_reset(*args, **kwargs):
            obs, info = gainrand_reset(*args, **kwargs)
            if agent._mode == AgentMode.TRAIN:
                obs = obs + torch.randn_like(obs) * OBS_NOISE
            return obs, info

        agent._decide_action = noisy_decide
        agent._step_env = pushy_step
        agent._reset_envs = noisy_reset
        return agent

    env_builder.build_env = build_env
    agent_builder.build_agent = build_agent


start_checkpoint_exporter(os.environ.get("X1_EXPORT_PREFIX", "smppdx"))
apply_robustness_patches()

sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine_pdx.yaml",
            "--env_config", "data/envs/smp_x1_env_v3.yaml",
            "--agent_config", "data/agents/smp_x1_agent_v3.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", os.environ.get("X1_MAX_SAMPLES", "500000000")]
_mf = os.environ.get("X1_MODEL_FILE", "X1_SMP/upload/**/smp_it2700*.pt")
if _mf:
    # platform-mounted resume checkpoint (mount layout varies — search
    # the repo dir and the whole workspace, newest match wins)
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
        # last resort: any file matching the basename anywhere
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
    dst = os.path.join(ROOT, "output", "smp_pdx_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final model -> output/smp_pdx_final.pt", flush=True)
        time.sleep(90)
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
