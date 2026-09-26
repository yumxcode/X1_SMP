import os
import sys
import runpy
import subprocess
import threading
import time
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

EXP_ID = time.strftime("%H%M%S")


def start_prior_exporter():
    """tinymdm overwrites output/smp_prior_x1/model.pt every output_iter;
    snapshot it to top-level output/*.pt (the only live upload channel)
    every 5 min + a final copy at exit."""
    src = os.path.join(ROOT, "output", "smp_prior_x1", "model.pt")

    def loop():
        while True:
            try:
                if os.path.exists(src):
                    dst = os.path.join(
                        ROOT, "output", f"prior_snap_{EXP_ID}_{int(time.time())}.pt")
                    shutil.copyfile(src, dst)
                    print(f"[exporter] prior snapshot -> {os.path.basename(dst)}",
                          flush=True)
            except Exception as e:
                print(f"[exporter] error: {e}", flush=True)
            time.sleep(300)

    threading.Thread(target=loop, daemon=True).start()


start_prior_exporter()

sys.path.insert(0, os.path.join(ROOT, "tools", "diffusion_model"))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["train_tinymdm.py",
            "--cfg_path", "tools/diffusion_model/config/tinymdm_x1_run.yaml",
            "--out_dir", "output/smp_prior_x1"]
runpy.run_path(os.path.join(ROOT, "tools", "diffusion_model",
                            "train_tinymdm.py"), run_name="__main__")

# natural end: final prior weights through the same top-level channel
try:
    src = os.path.join(ROOT, "output", "smp_prior_x1", "model.pt")
    dst = os.path.join(ROOT, "output", "prior_final.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("[exporter] final prior -> output/prior_final.pt", flush=True)
        time.sleep(90)  # let the SDK upload it before the container dies
except Exception as e:
    print(f"[exporter] final error: {e}", flush=True)
