"""F3 variant entry (SW-PREREG-005b): frictionloss=0 XML copy (F1 state).

r1 lesson: set_dof_friction_coefficients needs an indices arg (API risk);
loading a frictionloss=0 copy of x1_v4.xml is deterministic and keeps the
readback prints for verification (expect 0/29 nonzero)."""
import os
import runpy

here = os.path.dirname(os.path.abspath(__file__))
os.environ["X1_IL_XML"] = "data/assets/x1/x1_v4_f0.xml"
os.environ["X1_IL_FRICTION_ZERO"] = "0"  # readback-only; XML already f=0
os.environ["X1_IL_TAG"] = "F3_friction0_implicit_v4"
runpy.run_path(os.path.join(here, "probe_x1_il_entry.py"),
               run_name="__main__")
