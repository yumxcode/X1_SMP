"""SMP v3-pdx training: EXPLICIT PD drive in Isaac (pd_explicit).

Why: same-generation, asset-aligned experiments proved the Isaac Gym
DOF_MODE_POS implicit drive diverges from any MuJoCo parametric replica
(first control-step dof_vel diff 3.3 rad/s on a contact-free reset; airborne
step response 1.2-2.2x faster; kp/kd scaling, servo+implicitfast, margins,
solref, frictionloss and armature all rejected). MimicKit's pd_explicit
mode computes tau = clip(kp*(q_tar-q) - kd*qd, +-tlim) per physics substep
and drives DOF_MODE_EFFORT — semantically IDENTICAL to the MuJoCo sim2sim
harness. Training under these dynamics should transfer by construction.

Action bounds and obs are identical to pos mode (char_env shares the
_build_action_bounds_pos path), so the v3 prior and the v3b it2700 policy
warm-start remain valid.
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
      f" | engine pd_explicit", flush=True)

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

# reuse the robustness stack (obs noise + pushes; gain rand default 0 for
# the pure alignment run — X1_GAIN_RAND env can override)
os.environ.setdefault("X1_GAIN_RAND", "0.0")

_mf = os.environ.get("X1_MODEL_FILE", "")


def start_checkpoint_exporter(prefix):
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


# import the robust patches (gainrand + noise) from the v3 launcher without
# executing its runpy tail: exec the file up to the sys.argv assignment
src = open(os.path.join(ROOT, "scripts_remote", "run_smp_robust_v3.py")).read()
src = src.split('sys.argv = ["run.py"')[0]
exec(compile(src, "run_smp_robust_v3_partial", "exec"))

start_checkpoint_exporter(os.environ.get("X1_EXPORT_PREFIX", "smppdx"))

sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine_pdx.yaml",
            "--env_config", "data/envs/smp_x1_env_v3.yaml",
            "--agent_config", "data/agents/smp_x1_agent_v3.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", os.environ.get("X1_MAX_SAMPLES", "500000000")]
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
    dst = os.path.join(ROOT, "output", "smp_pdx_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final model -> output/smp_pdx_final.pt", flush=True)
        time.sleep(90)
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
