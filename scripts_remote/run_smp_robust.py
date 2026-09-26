"""SMP v2: robustness-regularized training for MuJoCo sim2sim transfer.

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
    import torch
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
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
            if agent._mode == AgentMode.TRAIN:
                obs = obs + torch.randn_like(obs) * OBS_NOISE
            a, a_info = orig_decide(obs, info)
            if agent._mode == AgentMode.TRAIN:
                a = a + torch.randn_like(a) * ACT_NOISE
                if state["prev_a"] is not None and LATENCY_STEPS > 0:
                    held = state["prev_a"]
                    a = torch.where(
                        torch.rand(a.shape[0], 1, device=a.device) < 0.3,
                        held, a)
                state["prev_a"] = a.detach().clone()
            return a, a_info

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
                        env._engine.set_body_forces([e], 0, 0,
                                                     torch.zeros(3))
                        state["push"][e] = None
                    if (cur is None or cur[1] <= 0) and \
                            torch.rand(1).item() < PUSH_PROB:
                        ang = torch.rand(1).item() * 2 * 3.14159265
                        f = torch.tensor([PUSH_FORCE * float(torch.cos(
                            torch.tensor(ang))),
                                          PUSH_FORCE * float(torch.sin(
                              torch.tensor(ang))), 0.0])
                        env._engine.set_body_forces([e], 0, 0, f)
                        state["push"][e] = (f, PUSH_STEPS)
            state["step"] = state.get("step", 0) + 1
            return orig_step(action)

        agent._decide_action = noisy_decide
        agent._step_env = pushy_step
        return agent

    env_builder.build_env = build_env
    agent_builder.build_agent = build_agent


start_checkpoint_exporter(os.environ.get("X1_EXPORT_PREFIX", "smp"))
apply_robustness_patches()

sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/smp_x1_env.yaml",
            "--agent_config", "data/agents/smp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", os.environ.get("X1_MAX_SAMPLES", "500000000")]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")

try:
    src = os.path.join(ROOT, "output", "model.pt")
    dst = os.path.join(ROOT, "output", "smp_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final model -> output/smp_final.pt", flush=True)
        time.sleep(90)
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
