import os
import sys
import runpy
import subprocess

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
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/smp_x1_env.yaml",
            "--agent_config", "data/agents/smp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")
