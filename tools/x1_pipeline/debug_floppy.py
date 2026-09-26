"""Test the 'Isaac floppy joints' hypothesis: zero kp/kd on the 5
weak-response joints in MuJoCo, replay Isaac actions, check tracking."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))

FLOPPY = ["right_ankle_roll_joint"]


def replay(mode):
    s2s = Sim2Sim(pol)
    d, m = s2s.d, s2s.m
    if mode == "isaacPD":
        m.dof_damping[s2s.vadr] = 0
        m.opt.timestep = 1.0 / 1200.0
        fine = 10
        pd = "explicit"
    else:
        fine = 1
        pd = "implicit"
    if mode == "floppy":
        # no torque output on right_ankle_roll (eff=0), keep passive damping
        k = 28
        s2s.eff[k] = 0.0

    d.qpos[:3] = traj["root_pos"][0].numpy()
    q_ = traj["root_quat"][0].numpy()
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][0].numpy()
    d.qpos[s2s.qadr][28] = -d.qpos[s2s.qadr][28]  # mirrored joint init
    d.qvel[:3] = traj["root_vel"][0].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][0].numpy()
    s2s.mj.mj_forward(m, d)

    fell = None
    errs = []
    for t in range(90):
        a = traj["action"][t].numpy()
        a[28] = -a[28]  # mirrored joint: negate command
        q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
        for _ in range(4 * fine):
            q = d.qpos[s2s.qadr]
            qd = d.qvel[s2s.vadr]
            if pd == "explicit":
                tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                              -s2s.eff, s2s.eff)
            else:
                tau = np.clip(s2s.kp * (q_tar - q), -s2s.eff, s2s.eff)
            d.ctrl[:] = tau
            s2s.mj.mj_step(m, d)
        ref = traj["dof_pos"][min(t + 1, 119)].numpy().copy()
        ref[28] = -ref[28]
        errs.append(np.abs(d.qpos[s2s.qadr] - ref))
        if d.qpos[2] < 0.30 and fell is None:
            fell = t / 30
            break
    E = np.stack(errs)
    print(f"{mode:>8}: {'FELL %.2fs' % fell if fell else 'UP 3s  '} "
          f"med {np.median(E):.3f} max {E.max():.3f} "
          f"@t{E.max(axis=0).argmax()/30:.2f}s")


replay("isaacPD")
