"""Divergence-driver hunt: replay Isaac actions under different physics
variants; find which change makes MuJoCo track Isaac."""
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

NCTRL = 60  # 2 s


def replay(variant):
    s2s = Sim2Sim(pol)
    d, m = s2s.d, s2s.m
    if variant in ("nofric", "nofric+isaacPD"):
        m.dof_frictionloss[:] = 0
    if variant in ("noarm", "noarm+isaacPD"):
        m.dof_armature[:] = 0
    if variant in ("isaacPD", "nofric+isaacPD", "noarm+isaacPD"):
        m.dof_damping[s2s.vadr] = 0  # kd explicit instead
    else:
        m.dof_damping[s2s.vadr] = s2s.kd

    fine = 10 if variant in ("isaacPD", "nofric+isaacPD", "noarm+isaacPD",
                             "fine1200") else 1
    m.opt.timestep = 1.0 / 120.0 / fine

    d.qpos[:3] = traj["root_pos"][0].numpy()
    q_ = traj["root_quat"][0].numpy()
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][0].numpy()
    d.qvel[:3] = traj["root_vel"][0].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][0].numpy()
    s2s.mj.mj_forward(m, d)

    fell = None
    dof_errs = []
    for t in range(NCTRL):
        a = traj["action"][t].numpy()
        q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
        for _ in range(4 * fine):
            q = d.qpos[s2s.qadr]
            qd = d.qvel[s2s.vadr]
            if variant in ("isaacPD", "nofric+isaacPD", "noarm+isaacPD"):
                tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                              -s2s.eff, s2s.eff)
            else:
                tau = np.clip(s2s.kp * (q_tar - q), -s2s.eff, s2s.eff)
            d.ctrl[:] = tau
            s2s.mj.mj_step(m, d)
        k = min(t + 1, NCTRL)
        dof_errs.append(np.abs(d.qpos[s2s.qadr] -
                               traj["dof_pos"][k].numpy()).max())
        if d.qpos[2] < 0.30 and fell is None:
            fell = t / 30
            break
    e = np.array(dof_errs)
    print(f"{variant:>16}: {'FELL %.2fs' % fell if fell else 'up 2s  '} "
          f"dof_err max {e.max():.3f} @t{e.argmax()/30:.2f}s "
          f"last {e[-1]:.3f}")


for v in ("baseline", "fine1200", "isaacPD", "nofric", "nofric+isaacPD",
          "noarm+isaacPD"):
    replay(v)
