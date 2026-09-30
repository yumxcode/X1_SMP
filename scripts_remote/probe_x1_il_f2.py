"""F2 variant entry (SW-PREREG-005a): limit-margined state injection."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_IL_MARGIN"] = "0.05"
os.environ["X1_IL_TAG"] = "F2_margin_implicit_v4"
runpy.run_path(os.path.join(here, "probe_x1_il_entry.py"),
               run_name="__main__")
