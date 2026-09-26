"""Identify per-joint actuation: regress dof_vel[j] on (a[i]-q[j]).
Reveals effective slot mapping, sign, and gain (near-zero = passive)."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
a = traj["action"].numpy()
q = traj["dof_pos"].numpy()
dv = traj["dof_vel"].numpy()
N = 110

print(f"{'joint':>30} {'self_R2':>8} {'best_R2':>8} {'best_slot':>22} "
      f"{'gain':>7}")
for j in range(29):
    y = dv[:N, j]
    best = (0.0, -1, 0.0)
    for i in range(29):
        x = a[:N, i] - q[:N, j]
        if x.std() < 1e-6:
            continue
        k = np.polyfit(x, y, 1)[0]
        r2 = 1 - np.var(y - k * x) / max(np.var(y), 1e-9)
        if abs(r2) > abs(best[0]):
            best = (r2, i, k)
    x = a[:N, j] - q[:N, j]
    k0 = np.polyfit(x, y, 1)[0] if x.std() > 1e-6 else 0
    r0 = 1 - np.var(y - k0 * x) / max(np.var(y), 1e-9) if x.std() > 1e-6 else 0
    flag = ""
    if best[1] != j and abs(best[0]) > abs(r0) + 0.1:
        flag = " <== REMAPPED"
    if abs(best[0]) < 0.05:
        flag = " (passive?)"
    print(f"{X1_DOF_ORDER[j]:>30} {r0:8.2f} {best[0]:8.2f} "
          f"{X1_DOF_ORDER[best[1]] if best[1]>=0 else '-':>22} "
          f"{best[2]:7.1f}{flag}")
