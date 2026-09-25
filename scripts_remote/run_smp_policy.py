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


def start_checkpoint_exporter(exp_name):
    """Copy each new int_models checkpoint to the platform's blessed path:
    logs/{exp}/exported_data/{load_run}/model_{iter}.pt (verified upload
    convention of the gradmotion SDK)."""
    run_dir = os.path.join(ROOT, "logs", exp_name, "exported_data",
                           time.strftime("%Y-%m-%d_%H-%M-%S"))
    exported = set()

    def loop():
        while True:
            try:
                for f in sorted(glob.glob(os.path.join(
                        ROOT, "output", "int_models", "model_*.pt"))):
                    base = os.path.basename(f)          # model_000001200.pt
                    if base in exported:
                        continue
                    it = int(base[6:-3])
                    os.makedirs(run_dir, exist_ok=True)
                    dst = os.path.join(run_dir, f"model_{it}.pt")
                    shutil.copyfile(f, dst)
                    exported.add(base)
                    print(f"[exporter] {dst}", flush=True)
            except Exception as e:
                print(f"[exporter] error: {e}", flush=True)
            time.sleep(120)

    threading.Thread(target=loop, daemon=True).start()


start_checkpoint_exporter(os.environ.get("X1_EXP_NAME", "x1_smp"))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/smp_x1_env.yaml",
            "--agent_config", "data/agents/smp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/",
            "--save_int_models", "true",
            "--max_samples", "500000000"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")
