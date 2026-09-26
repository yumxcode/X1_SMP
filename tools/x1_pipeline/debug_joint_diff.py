"""Per-joint joint_tan diff + axis comparison between the two models."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
import mujoco  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m

i = 30
d.qpos[:3] = traj["root_pos"][i].numpy()
d.qpos[3:7] = traj["root_quat"][i].numpy()
d.qpos[s2s.qadr] = traj["dof_pos"][i].numpy()
d.qvel[:] = 0
s2s.mj.mj_forward(m, d)
myobs = s2s.obs()
isaac = traj["obs"][i].numpy()

print(f"{'joint':>34} {'tan d':>7} {'norm d':>7}  axis_sim        axis_asset")
ma = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1.xml"))
for k, jn in enumerate(X1_DOF_ORDER):
    a, b = 13 + 6 * k, 19 + 6 * k
    dt_ = np.abs(myobs[a:a+3] - isaac[a:a+3]).max()
    dn_ = np.abs(myobs[b:b+3] - isaac[b:b+3]).max()
    ax_s = m.jnt_axis[m.joint(jn).id]
    ax_a = ma.jnt_axis[ma.joint(jn).id]
    flag = " <== MISMATCH" if max(dt_, dn_) > 0.01 else ""
    print(f"{jn:>34} {dt_:7.3f} {dn_:7.3f}  {ax_s} {ax_a}{flag}")

# key_rel per body
print("\nkey_rel per body (isaac - mine):")
kb = ["left_ankle_roll_link", "right_ankle_roll_link", "lumbar_pitch_link",
      "left_wrist_roll_link", "right_wrist_roll_link"]
mine = d.xpos[s2s.key_body_ids]
for c, nm in enumerate(kb):
    ia = isaac[216 + 3 * c: 219 + 3 * c]
    print(f"  {nm:>26}: diff {np.round(ia - mine[c], 3)}")
