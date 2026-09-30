"""One-shot diag entry for the x1-il probe (full-vector permutation dump)."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_IL_DIAG"] = "1"
runpy.run_path(os.path.join(here, "probe_x1_il_entry.py"),
               run_name="__main__")
