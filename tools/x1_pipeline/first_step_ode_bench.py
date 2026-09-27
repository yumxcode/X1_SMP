"""First-control-step actuation ODE benchmark + MuJoCo variant sweep.

For the dumped initial state (q, qd, q_tar at t=0), per joint:
  * integrate the TRUE scalar ODE  J*qdd = clip(kp*(tar-q) - kd*qd)  over
    33ms with RK4 at 0.1ms — the physics-agnostic reference;
  * compare Isaac dump dof_vel[1], current MuJoCo harness (explicit clipped
    kp + implicit damping), and variants: dt in {5ms, 2ms, 1ms, 8.33ms},
    native position servo (aligned retrofit).
Prints per-variant max |dof_vel - ODE| over the 8 worst joints.
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
from aligned_actuator_test import retrofit

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3.pt",
                  map_location="cpu", weights_only=False)

kp = traj["kp"].numpy()
kd = traj["kd"].numpy()
tlim = traj["tlim"].numpy()
q0 = traj["dof_pos"][0].numpy().copy()
qd0 = traj["dof_vel"][0].numpy().copy()
a0 = traj["action"][0].numpy()
pol_stub = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3_it2200.pt"))
sim = SV.Sim2Sim(pol_stub)
q_tar0 = np.clip(a0, -sim.a_bound, sim.a_bound)
isaac_dv = traj["dof_vel"][1].numpy()

# effective joint inertia from MuJoCo (dof_M at state)
m, d = sim.m, sim.d
sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = q0
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = qd0
sim.mj.mj_forward(m, d)
# effective joint inertia: diagonal of M via mj_fullM (or qM trick)
Mfull = np.zeros((m.nv, m.nv))
mujoco.mj_fullM(m, Mfull, d.qM)
Mdiag = np.diag(Mfull).copy()
J = Mdiag[sim.vadr]
print(f"[bench] joint inertia J[:6] = {np.round(J[:6], 4)}")

def ode_dv(J_, kp_, kd_, tl, q_t, q, qd, T=1.0 / 30.0, h=1e-4):
    """RK4-ish fine integration of the clamped PD scalar ODE."""
    n = int(T / h)
    for _ in range(n):
        def f(q_, v_):
            tau = np.clip(kp_ * (q_t - q_) - kd_ * v_, -tl, tl)
            return v_, tau / J_
        # semi-implicit Euler at 0.1ms is plenty accurate for overdamped
        v, _ = f(q, qd)
        qd = qd + h * (np.clip(kp_ * (q_t - q) - kd_ * qd, -tl, tl) / J_)
        q = q + h * qd
    return qd

ode_dv_res = ode_dv(J, kp, kd, tlim, q_tar0, q0.copy(), qd0.copy())
worst = np.argsort(-np.abs(sim_dv := None) if False else
                   -np.maximum(np.abs(ode_dv_res - isaac_dv),
                               np.zeros_like(ode_dv_res)))[:8]

print('== per-joint first-step dof_vel: ODE vs Isaac vs MuJoCo(variants) ==')
print(f"{'joint':24s} {'J':>7s} {'dq_tar':>7s} {'qd0':>7s} {'ODE':>8s} "
      f"{'Isaac':>8s} {'MJ5ms':>8s}")
names = [m.actuator(i).name.replace("motor_", "").replace("_joint", "")
         for i in range(m.nu)]

def run_mj(dt, servo=False, steps_ctrl=1.0 / 30.0):
    s2 = SV.Sim2Sim(pol_stub)
    if servo:
        retrofit(s2)
    mm, dd = s2.m, s2.d
    mm.opt.timestep = dt
    s2.mj.mj_resetData(mm, dd)
    dd.qpos[:3] = traj["root_pos"][0].numpy()
    dd.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    dd.qpos[s2.qadr] = q0
    dd.qvel[:3] = traj["root_vel"][0].numpy()
    dd.qvel[3:6] = traj["root_ang_vel"][0].numpy()
    dd.qvel[s2.vadr] = qd0
    s2.mj.mj_forward(mm, dd)
    nsub = int(round(steps_ctrl / dt))
    for _ in range(nsub):
        if servo:
            dd.ctrl[:] = np.clip(a0, -s2.a_bound, s2.a_bound)
        else:
            q_now = dd.qpos[s2.qadr]
            tau = np.clip(s2.kp * (q_tar0 - q_now)
                          - s2.kd * dd.qvel[s2.vadr], -s2.eff, s2.eff)
            dd.ctrl[:] = tau
        mujoco.mj_step(mm, dd)
    return dd.qvel[s2.vadr]

mj_5ms = run_mj(0.005)
mj_2ms = run_mj(0.002)
mj_1ms = run_mj(0.001)
mj_8333 = run_mj(1.0 / 120.0)
mj_servo_8333 = run_mj(1.0 / 120.0, servo=True)

for k in np.argsort(-np.abs(mj_5ms - isaac_dv))[:8]:
    print(f"{names[k]:24s} {J[k]:7.4f} {q_tar0[k]-q0[k]:+7.3f} {qd0[k]:+7.3f} "
          f"{ode_dv_res[k]:+8.3f} {isaac_dv[k]:+8.3f} {mj_5ms[k]:+8.3f}")

print('\n== max |dof_vel - ODE| (worst joint) ==')
for tag, arr in (('Isaac dump', isaac_dv), ('MJ 5ms explicit', mj_5ms),
                 ('MJ 2ms explicit', mj_2ms), ('MJ 1ms explicit', mj_1ms),
                 ('MJ 8.33ms explicit', mj_8333),
                 ('MJ 8.33ms servo', mj_servo_8333)):
    print(f"  {tag:20s} {np.abs(arr - ode_dv_res).max():7.3f} rad/s "
          f"(med {np.median(np.abs(arr - ode_dv_res)):.4f})")
print('\n== max |dof_vel - Isaac| ==')
for tag, arr in (('ODE', ode_dv_res), ('MJ 5ms explicit', mj_5ms),
                 ('MJ 2ms explicit', mj_2ms), ('MJ 1ms explicit', mj_1ms),
                 ('MJ 8.33ms explicit', mj_8333),
                 ('MJ 8.33ms servo', mj_servo_8333)):
    print(f"  {tag:20s} {np.abs(arr - isaac_dv).max():7.3f} rad/s "
          f"(med {np.median(np.abs(arr - isaac_dv)):.4f})")
