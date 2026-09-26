"""One-to-one obs diff: feed each Isaac trajectory state into MuJoCo,
compute obs via the validator, compare per block against Isaac's obs."""
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
print("keys:", list(traj.keys()))

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m

names = [("h", 0, 1), ("rot_tan", 1, 7), ("root_vel", 7, 10),
         ("ang_vel", 10, 13), ("joint_tan", 13, 187), ("dof_vel", 187, 216),
         ("key_rel", 216, 231)]

n = traj["obs"].shape[0]
myobs_all = np.zeros((n, 231))
for i in range(n):
    d.qpos[:3] = traj["root_pos"][i].numpy()
    q_ = traj["root_quat"][i].numpy()  # xyzw as dumped
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]  # -> wxyz for MuJoCo
    d.qpos[s2s.qadr] = traj["dof_pos"][i].numpy()
    d.qvel[:3] = traj["root_vel"][i].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][i].numpy()  # local-frame pass-through
    d.qvel[s2s.vadr] = traj["dof_vel"][i].numpy()
    s2s.mj.mj_forward(m, d)
    myobs_all[i] = s2s.obs()

isaac_obs = traj["obs"].numpy()
print(f"\n{'block':>10} {'max|diff|':>10} {'mean|diff|':>11}  note")
for nm, a, b in names:
    diff = np.abs(myobs_all[:, a:b] - isaac_obs[:, a:b])
    print(f"{nm:>10} {diff.max():10.4f} {diff.mean():11.5f}")

# ang vel block special: try world-frame alternative
print("\nang_vel block with WORLD-frame conversion instead:")
for i in range(n):
    d.qpos[:3] = traj["root_pos"][i].numpy()
    q_ = traj["root_quat"][i].numpy()  # xyzw
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][i].numpy()
    d.qvel[:3] = traj["root_vel"][i].numpy()
    R = d.xmat[s2s.root_bid].reshape(3, 3)
    d.qvel[3:6] = R.T @ traj["root_ang_vel"][i].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][i].numpy()
    s2s.mj.mj_forward(m, d)
    myobs_all[i] = s2s.obs()
diff = np.abs(myobs_all[:, 10:13] - isaac_obs[:, 10:13])
print(f"  max {diff.max():.4f} mean {diff.mean():.5f}")
