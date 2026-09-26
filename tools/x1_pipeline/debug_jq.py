import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))
from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m
i = 30
dof = traj["dof_pos"][i].numpy()
d.qpos[:3] = traj["root_pos"][i].numpy()
d.qpos[3:7] = traj["root_quat"][i].numpy()
d.qpos[s2s.qadr] = dof
d.qvel[:] = 0
s2s.mj.mj_forward(m, d)

print("dof fed [:6]      :", np.round(dof[:6], 3))
print("qpos[qadr][:6]    :", np.round(d.qpos[s2s.qadr][:6], 3))
print("qadr[:6]          :", s2s.qadr[:6])
print("axes[0..2]        :", s2s.axes[0], s2s.axes[1], s2s.axes[2])
jq = s2s.joint_quats()
print("joint_quats[0..2] :")
print(np.round(jq[:3], 4))
o = s2s.obs()
print("obs[13:19]        :", np.round(o[13:19], 3))
print("obs[19:25]        :", np.round(o[19:25], 3))
print("isaac[13:19]      :", np.round(traj["obs"][i].numpy()[13:19], 3))
print("isaac[19:25]      :", np.round(traj["obs"][i].numpy()[19:25], 3))
