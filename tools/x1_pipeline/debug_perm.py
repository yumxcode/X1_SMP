"""Solve the dof permutation: which engine dof slot feeds which xml joint.
Match Isaac's joint_tan obs blocks against all (slot, joint) combos."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
import util.torch_util as torch_util  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
m = s2s.m

i = 30
dof = traj["dof_pos"][i].numpy()
isa = traj["obs"][i].numpy()
axes = s2s.axes  # xml axis per X1_DOF_ORDER joint

# match[engine_slot, xml_joint] = |tan-norm difference| max
match = np.zeros((29, 29))
for slot in range(29):
    q = torch_util.axis_angle_to_quat(
        torch.tensor(axes), torch.full((29,), float(dof[slot])))
    tn = torch_util.quat_to_tan_norm(q).numpy()  # (29, 6)
    for j in range(29):
        match[slot, j] = np.abs(tn[j] - isa[13 + 6 * j:19 + 6 * j]).max()

perm = np.argmin(match, axis=1)
print("engine_slot -> xml_joint_idx (best match):")
print(perm)
identity = np.array_equal(perm, np.arange(29))
print("is identity?", identity)
print("residual max:", match[np.arange(29), perm].max())
if not identity:
    for s in range(29):
        if perm[s] != s:
            print(f"  slot {s} ({X1_DOF_ORDER[s]:>32}) <- joint {perm[s]} "
                  f"({X1_DOF_ORDER[perm[s]]})")
