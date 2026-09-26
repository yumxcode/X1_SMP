"""Resume SMP training from a platform-mounted checkpoint.

The gradmotion task mounts the source checkpoint (checkPointFilePath)
under the repo; this launcher points the robust trainer at it via
X1_MODEL_FILE and continues training for the remaining samples.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

# the mounted resume checkpoint: gradmotion mounts under X1_SMP/upload/...
os.environ["X1_MODEL_FILE"] = os.environ.get(
    "X1_MODEL_FILE", "X1_SMP/upload/**/smp_it1900*.pt")
os.environ.setdefault("X1_MAX_SAMPLES", "340000000")
os.environ.setdefault("X1_EXPORT_PREFIX", "smpv2r")

import runpy
runpy.run_path(os.path.join(ROOT, "scripts_remote", "run_smp_robust.py"),
               run_name="__main__")
