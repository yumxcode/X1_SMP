"""Action replay: start from the Isaac traj initial state, apply the SAME
action sequence in MuJoCo, measure state divergence. Isolates physics-
model mismatch from policy-feedback mismatch."""
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


def set_state(i):
    d.qpos[:3] = traj["root_pos"][i].numpy()
    q_ = traj["root_quat"][i].numpy()  # xyzw
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][i].numpy()
    d.qvel[:3] = traj["root_vel"][i].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][i].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][i].numpy()
    s2s.mj.mj_forward(m, d)


set_state(0)
print(f"{'t':>5} {'dz':>7} {'quat_d':>7} {'dof_d':>7} {'vz':>6}")
for t in range(90):
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
    for _ in range(4):
        q = d.qpos[s2s.qadr]
        tau = np.clip(s2s.kp * (q_tar - q), -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        s2s.mj.mj_step(m, d)
    if t % 5 == 0:
        # divergence vs the actual Isaac state at t+1
        k = min(t + 1, 119)
        dz = abs(d.qpos[2] - traj["root_pos"][k, 2].item())
        q_ = traj["root_quat"][k].numpy()
        qd = min(np.abs(d.qpos[3:7] - np.array([q_[3], q_[0], q_[1], q_[2]])).max(),
                 np.abs(d.qpos[3:7] + np.array([q_[3], q_[0], q_[1], q_[2]])).max())
        dd = np.abs(d.qpos[s2s.qadr] - traj["dof_pos"][k].numpy()).max()
        vz = abs(d.qvel[2] - traj["root_vel"][k, 2].item())
        print(f"{t/30:5.2f} {dz:7.3f} {qd:7.3f} {dd:7.3f} {vz:6.2f}")
    if d.qpos[2] < 0.30:
        print(f"FELL at {t/30:.2f}s")
        break
