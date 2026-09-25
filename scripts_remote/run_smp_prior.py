import os
import sys
import runpy
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
# offline deps: container network blocks pypi; wheels are vendored in repo
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-index", "--find-links",
                       os.path.join(ROOT, "vendor_wheels"),
                       "gymnasium", "diffusers", "huggingface-hub",
                       "filelock", "importlib-metadata", "zipp",
                       "typing-extensions", "cloudpickle", "regex",
                       "tqdm", "safetensors", "requests"])
sys.path.insert(0, os.path.join(ROOT, "tools", "diffusion_model"))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["train_tinymdm.py",
            "--cfg_path", "tools/diffusion_model/config/tinymdm_x1_run.yaml",
            "--out_dir", "output/smp_prior_x1"]
runpy.run_path(os.path.join(ROOT, "tools", "diffusion_model",
                            "train_tinymdm.py"), run_name="__main__")
