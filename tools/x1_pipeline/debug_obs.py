"""Obs block z-score vs training normalizer stats at reset — find the mismatched block."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)

# blocks: [h(1), rottan(6), rootvel(3), angvel(3), jointtan(116), dofvel(29), keyrel(15)]
names = [("h", 0, 1), ("rot_tan", 1, 7), ("root_vel", 7, 10),
         ("ang_vel", 10, 13), ("joint_tan", 13, 129), ("dof_vel", 129, 158),
         ("key_rel", 158, None)]
obs_dim = pol.l1_w.shape[1]
print("obs_dim =", obs_dim)
names[-1] = ("key_rel", 158, obs_dim)

s2s.reset(seed=7)
o = s2s.obs()
z = (o - pol.obs_mean) / (pol.obs_std + 1e-8)
print(f"{'block':>10} {'|z|max':>7} {'z_med':>7}  first-3-obs / mean / std")
for nm, a, b in names:
    zz = z[a:b]
    print(f"{nm:>10} {np.abs(zz).max():7.2f} {np.median(zz):7.2f}  "
          f"{o[a:a+3]} | {pol.obs_mean[a:a+3]} | {pol.obs_std[a:a+3]}")

# also print raw mean/std magnitudes of joint_tan & key_rel (scale sanity)
print("\nobs_std min/max per block:")
for nm, a, b in names:
    print(f"  {nm:>10}: std [{pol.obs_std[a:b].min():.4f}, {pol.obs_std[a:b].max():.4f}]")
