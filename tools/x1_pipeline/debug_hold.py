"""Positive control: can the X1 sim2sim model stand at all?
1) reset-state geometry: foot sole heights, root z
2) hold-home controller (q_tar = home): does it stay up 3s?
3) zero-action controller: does it stay up?"""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Sim2Sim, Policy  # noqa: E402


class HoldPolicy:
    def __init__(self, target):
        self.target = target

    def forward(self, obs):
        return self.target.copy()


pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
s2s.reset(seed=3)

d = s2s.d
s2s.mj.mj_forward(s2s.m, d)
print("reset state: root z =", round(d.qpos[2], 4))
for site in ("x_lfoot", "x_rfoot"):
    print(f"  {site} z = {d.site(site).xpos[2]:.4f}")
lf = d.body("left_ankle_roll_link")
rf = d.body("right_ankle_roll_link")
print(f"  ankle bodies z: L={lf.xpos[2]:.4f} R={rf.xpos[2]:.4f}")

for name, target in (("hold-home", s2s.home), ("zero", np.zeros(29))):
    s2s.reset(seed=3)
    n_fall = None
    for it in range(90):  # 3 s
        q = d.qpos[s2s.qadr]
        qd = d.qvel[s2s.vadr]
        tau = np.clip(s2s.kp * (target - q) - s2s.kd * qd, -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        for _ in range(4):
            s2s.mj.mj_step(s2s.m, d)
        if d.qpos[2] < 0.30:
            n_fall = it / 30
            break
    R = d.xmat[s2s.torso_bid].reshape(3, 3)
    pitch = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
    print(f"[{name}] rootz={d.qpos[2]:.3f} pitch={pitch:.1f} "
          f"{'FELL at %.2fs' % n_fall if n_fall else 'up 3s'}")
