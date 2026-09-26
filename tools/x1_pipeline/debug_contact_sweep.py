"""Contact-parameter sweep on the action-replay tracking metric."""
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


def replay(fr=None, solref=None, solimp=None, condim=None, iters=None):
    s2s = Sim2Sim(pol)
    d, m = s2s.d, s2s.m
    m.dof_damping[s2s.vadr] = 0
    m.opt.timestep = 1.0 / 1200.0
    if fr is not None:
        m.geom_friction[:, 0] = fr
    if solref is not None:
        m.geom_solref[:] = solref
    if solimp is not None:
        m.geom_solimp[:] = solimp  # (dmin,dmax,width,mid,slip)
    if condim is not None:
        m.geom_condim[:] = condim
    if iters is not None:
        m.opt.iterations = iters

    d.qpos[:3] = traj["root_pos"][0].numpy()
    q_ = traj["root_quat"][0].numpy()
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][0].numpy()
    d.qvel[:3] = traj["root_vel"][0].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][0].numpy()
    s2s.mj.mj_forward(m, d)

    fell = None
    ank = []
    for t in range(90):
        a = traj["action"][t].numpy()
        q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
        for _ in range(40):
            q = d.qpos[s2s.qadr]
            qd = d.qvel[s2s.vadr]
            tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                          -s2s.eff, s2s.eff)
            d.ctrl[:] = tau
            s2s.mj.mj_step(m, d)
        ank.append(abs(d.qpos[s2s.qadr][28] -
                       traj["dof_pos"][min(t + 1, 119), 28].item()))
        if d.qpos[2] < 0.30 and fell is None:
            fell = t / 30
            break
    a_ = np.array(ank)
    label = (f"fr={fr} solref={solref} solimp={solimp} cd={condim} "
             f"it={iters}")
    print(f"{label:>64}: {'FELL %.2f' % fell if fell else 'UP 3s'} "
          f"ankle_err med {np.median(a_):.3f} max {a_.max():.3f}")


replay(solref=(0.004, 50))
replay(solref=(0.002, 100))
replay(solref=(0.002, 100), solimp=(0.99, 0.9999, 0.0001, 0.5, 2))
replay(solref=(0.001, 200), solimp=(0.999, 0.99999, 0.00001, 0.5, 2), iters=100)
replay(solimp=(0.99, 0.9999, 0.0001, 0.5, 2))
