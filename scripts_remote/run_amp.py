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
# offline deps: container network blocks pypi; wheels are vendored in repo
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

EXP = os.environ.get("X1_EXP_NAME", "x1_amp")


def start_checkpoint_exporter(prefix):
    """v9: the ONLY live upload channel is top-level output/*.pt with a NEW
    unique filename (probe-verified 2026-09-26, TASK_20260926_046). Copy every
    new int_models checkpoint there; the platform SDK picks it up in seconds."""
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


start_checkpoint_exporter(os.environ.get("X1_EXPORT_PREFIX", "amp"))

sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/amp_x1_env.yaml",
            "--agent_config", "data/agents/amp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", os.environ.get("X1_MAX_SAMPLES", "500000000"),
            *sys.argv[1:]]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")

# natural end: final model through the same top-level channel
try:
    src = os.path.join(ROOT, "output", "model.pt")
    dst = os.path.join(ROOT, "output", "amp_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final model -> output/amp_final.pt", flush=True)
        time.sleep(90)  # let the SDK upload it before the container dies
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
