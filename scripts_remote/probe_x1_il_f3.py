"""F3 variant entry (SW-PREREG-005b): friction readback + zero (F1 state)."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_IL_FRICTION_ZERO"] = "1"
os.environ["X1_IL_TAG"] = "F3_friction0_implicit_v4"
runpy.run_path(os.path.join(here, "probe_x1_il_entry.py"),
               run_name="__main__")
