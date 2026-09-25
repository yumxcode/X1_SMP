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
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/amp_x1_env.yaml",
            "--agent_config", "data/agents/amp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")
