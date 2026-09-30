"""Wrist-coded diag entry: pin the 4 ambiguous all-zero wrist slots (r7)."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_IL_WRIST_CODED"] = "1"
runpy.run_path(os.path.join(here, "probe_x1_il_entry.py"),
               run_name="__main__")
