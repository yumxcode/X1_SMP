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
d.qpos[:3] = traj["root_pos"][i].numpy()
d.qpos[3:7] = traj["root_quat"][i].numpy()
d.qpos[s2s.qadr] = traj["dof_pos"][i].numpy()
d.qvel[:] = 0
s2s.mj.mj_forward(m, d)
mine = s2s.obs()
isa = traj["obs"][i].numpy()
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("mine [0:20] :", mine[:20])
print("isaac[0:20] :", isa[:20])
print()
print("isaac [13:19]:", isa[13:19])
print("isaac [19:25]:", isa[19:25])
print("mine joint0  :", mine[13:19])
print()
print("isaac dofvel-block norm [187:216]:", np.linalg.norm(isa[187:216]))
print("isaac [222:231]:", isa[222:231])
minekey = (d.xpos[s2s.key_body_ids] - d.qpos[:3]).reshape(-1)
print("mine key_rel[:9]:", minekey[:9])
print("isaac [216:225]:", isa[216:225])
