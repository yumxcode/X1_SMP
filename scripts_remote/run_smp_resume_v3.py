"""Resume SMP v3 training from a platform-mounted checkpoint.

The gradmotion task mounts the source checkpoint (checkPointFilePath)
under the repo; this launcher points the v3 robust trainer at it via
X1_MODEL_FILE and continues training. Glob matches any exported smp_it*
checkpoint (newest wins, search lives in run_smp_robust_v3.py).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# the mounted resume checkpoint: gradmotion mounts under X1_SMP/upload/...
os.environ["X1_MODEL_FILE"] = os.environ.get(
    "X1_MODEL_FILE", "X1_SMP/upload/**/smp_it*.pt")
os.environ.setdefault("X1_MAX_SAMPLES", "500000000")
os.environ.setdefault("X1_EXPORT_PREFIX", "smpv3r")

import runpy
runpy.run_path(os.path.join(ROOT, "scripts_remote", "run_smp_robust_v3.py"),
               run_name="__main__")
