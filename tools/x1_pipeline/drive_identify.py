"""Identify Isaac's EFFECTIVE drive gains from the fixed-gen dump via
mj_inverse, then test the identified gains in the MuJoCo harness.

Method:
  1. From dump: q0, qd0, q_tar0, and qd1 (after one 33ms control step).
     qacc = (qd1 - qd0) / dt   (crude but unbiased for identification)
  2. mj_inverse(x1_sim, qpos0, qvel0, qacc) -> required generalized force
     tau_inv (this INCLUDES gravity/coriolis/passive as felt by MuJoCo's
     model; assuming both engines share the URDF inertia, the difference
     tau_inv - tau_passive0 is the drive torque Isaac effectively applied).
  3. Per joint regress drive = kp_eff*(q_tar-q) - kd_eff*qd  over the
     first K contact-free steps; group scales by joint class.
  4. Sim2sim with scaled gains — quick 4-episode probe.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import mujoco

import sim2sim_validate as SV

t = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3_fixed.pt",
               map_location="cpu", weights_only=False)
kp_r, kd_r = t["kp"].numpy(), t["kd"].numpy()
q, qd, qt = t["dof_pos"].numpy(), t["dof_vel"].numpy(), t["q_tar"].numpy()

pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3b_it1800.pt"))
sim = SV.Sim2Sim(pol)
m = sim.m
d = sim.d
dt = 1.0 / 30.0

K = 8  # first steps: contact-free (verified ncon=0 at t=0/1 in prior diff)

drive_est = []
for k in range(K):
    qpos = np.concatenate([t["root_pos"][k].numpy(),
                           t["root_quat"][k].numpy()[[3, 0, 1, 2]],
                           q[k]])
    qvel = np.concatenate([t["root_vel"][k].numpy(),
                           t["root_ang_vel"][k].numpy(),
                           qd[k]])
    qacc = np.concatenate([
        (t["root_vel"][k + 1].numpy() - t["root_vel"][k].numpy()) / dt,
        (t["root_ang_vel"][k + 1].numpy() - t["root_ang_vel"][k].numpy()) / dt,
        (qd[k + 1] - qd[k]) / dt])
    sim.mj.mj_resetData(m, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    d.qacc[:] = qacc
    sim.mj.mj_inverse(m, d)
    tau_inv = d.qfrc_inverse.copy()
    # subtract passive (spring/damper/friction) — x1_sim has none on joints,
    # but armature is inside qM; mj_inverse already accounts for it
    drive_est.append(tau_inv[sim.vadr])

drive_est = np.array(drive_est)  # (K, 29)
# target command at step k (q_tar recorded after step k -> use k for step k+1)
errs = []
for k in range(K):
    qtk = qt[k]
    pd_r = kp_r * (qtk - q[k]) - kd_r * qd[k]
    errs.append(drive_est[k] - pd_r)
errs = np.array(errs)

names = [m.actuator(i).name.replace("motor_", "") for i in range(m.nu)]
print("== implied drive torque (mj_inverse) vs reported-PD formula ==")
for k in np.argsort(-np.abs(errs).mean(axis=0))[:10]:
    print(f"  {names[k]:32s} implied[0] {drive_est[0,k]:+8.2f} "
          f"PD[0] {kp_r[k]*(qt[0,k]-q[0,k]) - kd_r[k]*qd[0,k]:+8.2f} "
          f"mean|err| {errs[:,k].mean():7.2f}")

# least squares per joint for kp_eff/kd_eff:
#   drive_est ≈ a*(qt-q) + b*qd  (b = -kd_eff)
A = np.stack([qt[:K] - q[:K], qd[:K]], axis=2)  # (K, 29, 2)
sol = np.linalg.lstsq(A.reshape(-1, 2)[:0] if False else
                      A.reshape(-1, 2),
                      drive_est.reshape(-1), rcond=None)[0]
# NOTE: joint-specific gains differ; do per-joint via normal equations
kp_eff = np.zeros(29)
kd_eff = np.zeros(29)
for j in range(29):
    Aj = np.stack([(qt[:K, j] - q[:K, j]), qd[:K, j]], axis=1)
    try:
        a, b = np.linalg.lstsq(Aj, drive_est[:, j], rcond=None)[0]
        kp_eff[j] = a
        kd_eff[j] = -b
    except Exception:
        kp_eff[j], kd_eff[j] = np.nan, np.nan
valid = np.isfinite(kp_eff) & (kp_eff > 0)
print("\n== per-joint effective gains (contact-free first steps) ==")
for j in range(29):
    if valid[j]:
        print(f"  {names[j]:32s} kp_eff {kp_eff[j]:8.1f} "
              f"(reported {kp_r[j]:6.0f}, scale {kp_eff[j]/kp_r[j]:6.3f})  "
              f"kd_eff {kd_eff[j]:7.2f} (reported {kd_r[j]:5.1f}, "
              f"scale {kd_eff[j]/kd_r[j]:6.3f})")
kp_scale = np.nanmedian(kp_eff[valid] / kp_r[valid])
kd_scale = np.nanmedian(kd_eff[valid] / kd_r[valid])
print(f"\nmedian kp scale {kp_scale:.4f} | kd scale {kd_scale:.4f}")
np.save(REPO / "output/diag_v3/eff_gains.npy",
        np.stack([kp_eff, kd_eff, kp_r, kd_r]))
print("saved output/diag_v3/eff_gains.npy")
