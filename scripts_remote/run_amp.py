"""CWD-safe AMP training launcher (platform executes startScript w/ python)."""
import os
import sys
import runpy

import subprocess, sys
# default pypi unreachable from the container; use mirrors with fallback
for idx in ("-i https://pypi.tuna.tsinghua.edu.cn/simple",
            "-i https://mirrors.aliyun.com/pypi/simple/"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               idx, "gymnasium", "diffusers>=0.36.0",
                               "moviepy", "matplotlib", "pyyaml",
                               "tensorboardX"])
        break
    except subprocess.CalledProcessError:
        continue


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
sys.argv = ["run.py", "--mode", "train", "--num_envs", "4096",
            "--engine_config", "data/engines/isaac_gym_engine.yaml",
            "--env_config", "data/envs/amp_x1_env.yaml",
            "--agent_config", "data/agents/amp_x1_agent.yaml",
            "--visualize", "false", "--out_dir", "output/"]
runpy.run_path(os.path.join(ROOT, "mimickit", "run.py"), run_name="__main__")
