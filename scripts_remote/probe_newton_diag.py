"""One-shot diag entry: run probe_newton_entry.py with shape-diag enabled
on a single config (r4-diag for the newton 1.2.1 Controls API drift)."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_PROBE_SHAPE_DIAG"] = "1"
os.environ["X1_PROBE_ONLY"] = "A_explicit"
runpy.run_path(os.path.join(here, "probe_newton_entry.py"),
               run_name="__main__")
