"""CWD-safe SMP prior (tiny-MDM) training launcher."""
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
sys.argv = ["train_tinymdm.py",
            "--cfg_path", "tools/diffusion_model/config/tinymdm_x1_run.yaml",
            "--out_dir", "output/smp_prior_x1"]
runpy.run_path(os.path.join(ROOT, "tools", "diffusion_model",
                            "train_tinymdm.py"), run_name="__main__")
