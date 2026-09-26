"""PD-track the retargeted demo motion with a FREE root in MuJoCo.
Fall => motion exceeds X1 physical envelope (or model mismatch)."""
import sys
from pathlib import Path
import numpy as np
import pickle

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))  # only for ctor
s2s = Sim2Sim(pol)
s2s.m.dof_damping[s2s.vadr] = s2s.kd
d, m = s2s.d, s2s.m

mo = pickle.load(open(REPO / "data/motions/x1/x1_sprint1_subject4_seg3.pkl", "rb"))
frames = np.array(mo["frames"], dtype=np.float32)
fps, ts = mo["fps"], mo.get("time_scale", 1.0)
dt_ctrl = 1.0 / fps  # slowdown is BAKED into frame count; loader plays at fps
print(f"frames={len(frames)} fps={fps} time_scale={ts} "
      f"dt_ctrl={dt_ctrl:.4f}s  s_leg={mo.get('s_leg')} s_arm={mo.get('s_arm')}")

# demo speed stats
pos = frames[:, 0:3]
seg_t = len(frames) * dt_ctrl
v_root = np.linalg.norm(pos[-1] - pos[0]) / seg_t
step_v = np.linalg.norm(np.diff(pos, axis=0), axis=1).mean() / dt_ctrl
print(f"demo root displacement speed: end-to-end {v_root:.2f} m/s, "
      f"mean-step {step_v:.2f} m/s, dur {seg_t:.2f}s")
print(f"demo root z: min {pos[:,2].min():.3f} max {pos[:,2].max():.3f}")

# PD-track replay: state init at frame0, then track joints
d.qpos[:3] = frames[0, 0:3]
# frames are [pos(3), root_expmap(3), dof(29)]
em = frames[0, 3:6]
ang = np.linalg.norm(em)
if ang < 1e-8:
    d.qpos[3:7] = [1, 0, 0, 0]
else:
    ax = em / ang
    d.qpos[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
d.qpos[s2s.qadr] = frames[0, 6:35]
d.qvel[:] = 0
mujoco = s2s.mj
mujoco.mj_forward(m, d)

n = len(frames)
fell_t = None
err_track = []
for it in range(n):
    f = frames[min(it, n - 1)]
    q_tar = f[6:35]
    sub = max(1, int(round(dt_ctrl / m.opt.timestep)))
    for _ in range(sub):
        tau = np.clip(s2s.kp * (q_tar - d.qpos[s2s.qadr]),
                      -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    err_track.append(np.abs(d.qpos[s2s.qadr] - q_tar).mean())
    if d.qpos[2] < 0.30 and fell_t is None:
        fell_t = it * dt_ctrl
        break

R = d.xmat[s2s.torso_bid].reshape(3, 3)
pitch = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
et = np.array(err_track)
print(f"[demo-track] {'FELL at %.2fs' % fell_t if fell_t else 'survived full seg'}"
      f" final rootz={d.qpos[2]:.3f} pitch={pitch:.1f}")
print(f"  track err: mean {et.mean():.3f} p90 {np.quantile(et,0.9):.3f} rad")
print(f"  root drift: demo z {frames[-1,2]:.3f} vs sim {d.qpos[2]:.3f}; "
      f"demo x-end {frames[-1,0]:.2f} vs sim x {d.qpos[0]:.2f}")
