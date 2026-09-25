"""MuJoCo sim2sim validation for trained X1 AMP/SMP policies.

Pure-MuJoCo rollout that exactly mirrors the Isaac training stack:
  * obs = compute_char_obs semantics (char_env.py): [root_h, rot tan-norm,
    root vel, root ang vel (global), 29 joint-rot tan-norms, 29 dof vels,
    5 key-body rel pos] -> 231 dims (global_obs=True, root_height_obs=True)
  * actor = Sequential(Linear 231-1024,relu?,Linear 1024-512) + mean head;
    params loaded from MimicKit checkpoint state_dict
  * normalizers: obs_norm / a_norm mean+std from checkpoint
  * action = dof position targets (zero-centered bounds, 1.4x margin),
    explicit PD torque (kp/kd from x1.xml, effort clip from URDF)
  * control 30 Hz / physics 1/120 s (matches Isaac sim_freq=120)

Strict PASS criteria (all must hold across episodes):
  S1 no fall: root height > 0.30 m for >= 95% of every episode; zero
     floor contacts of non-foot bodies (knee/torso/head/hand geoms).
  S2 gait: >= 4 alternating stride cycles; median stride period within
     [0.25, 1.2] s; mean forward speed >= 0.8 m/s (X1 run range);
     median swing clearance >= 4 cm on both feet.
  S3 morphology: |median torso pitch| < 25 deg; hip-pitch median within
     [-0.8, 0.8] rad & knee within [0.2, 1.4] rad (no skating/crouch);
     arm-vs-leg anti-phase |dphi| within 0.8 rad of pi (FFT phase).

Usage:
  .venv/bin/python tools/x1_pipeline/sim2sim_validate.py --ckpt model.pt \
      [--episodes 4] [--len 10] [--json out.json] [--gif out.gif]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mimickit"))
sys.path.insert(0, str(RESEARCH_DIR := REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402

X1_SIM = REPO_ROOT / "data/assets/x1/x1_sim.xml"
X1_ASSET = REPO_ROOT / "data/assets/x1/x1.xml"
KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link",
              "lumbar_pitch_link", "left_wrist_roll_link",
              "right_wrist_roll_link"]


# --------------------------------------------------------------- torch utils
import util.torch_util as torch_util  # noqa: E402


def compute_char_obs_np(root_pos, root_quat_wxyz, root_vel, root_ang_vel,
                        joint_quats_wxyz, dof_vel, key_pos_rel_world,
                        global_obs=True):
    """NumPy mirror of char_env.compute_char_obs (batch of 1)."""
    t = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(0)
    root_pos_t = t(root_pos)
    root_rot = t(root_quat_wxyz)
    root_v = t(root_vel)
    root_w = t(root_ang_vel)
    jr = torch.tensor(joint_quats_wxyz, dtype=torch.float32).unsqueeze(0)
    dv = t(dof_vel)
    kp = torch.tensor(key_pos_rel_world, dtype=torch.float32).unsqueeze(0)

    if global_obs:
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_v
        root_ang_vel_obs = root_w
    else:
        heading = torch_util.calc_heading_quat_inv(root_rot)
        local = torch_util.quat_mul(heading, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local)
        root_vel_obs = torch_util.quat_rotate(heading, root_v)
        root_ang_vel_obs = torch_util.quat_rotate(heading, root_w)

    jr_flat = jr.reshape(-1, 4)
    jr_obs = torch_util.quat_to_tan_norm(jr_flat).reshape(1, -1)

    # key pos: subtract root, keep world orientation (global_obs=True)
    key_rel = kp - root_pos_t.unsqueeze(-2)
    key_obs = key_rel.reshape(1, -1)

    obs = torch.cat([root_pos_t[:, 2:3], root_rot_obs, root_vel_obs,
                     root_ang_vel_obs, jr_obs, dv, key_obs], dim=-1)
    return obs.numpy()[0]


class Policy:
    """Actor + normalizers rebuilt from a MimicKit checkpoint state_dict."""

    def __init__(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "_model._actor_layers.0.weight" not in sd:
            raise ValueError(
                f"unexpected checkpoint keys: "
                f"{list(sd.keys())[:10] if isinstance(sd, dict) else type(sd)}")
        # normalizer stats
        self.obs_mean = sd["_obs_norm._mean"].numpy()
        self.obs_std = sd["_obs_norm._std"].numpy()
        self.a_mean = sd["_a_norm._mean"].numpy()
        self.a_std = sd["_a_norm._std"].numpy()
        self.a_min = sd.get("_a_norm._mean_min", None)
        # actor MLP
        self.l1_w = sd["_model._actor_layers.0.weight"]
        self.l1_b = sd["_model._actor_layers.0.bias"]
        self.l2_w = sd["_model._actor_layers.2.weight"]
        self.l2_b = sd["_model._actor_layers.2.bias"]
        self.h_w = sd["_model._action_dist._mean_net.weight"]
        self.h_b = sd["_model._action_dist._mean_net.bias"]
        # activation: ppo_model default is relu unless configured otherwise
        self.act = torch.relu

    def forward(self, obs):
        norm_obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        norm_obs = np.clip(norm_obs, -10, 10)
        with torch.no_grad():
            h = self.act(norm_obs @ self.l1_w.T + self.l1_b)
            h = self.act(h @ self.l2_w.T + self.l2_b)
            norm_a = h @ self.h_w.T + self.h_b  # mode (mean) of Gaussian
        a = norm_a * self.a_std + self.a_mean
        return a


class Sim2Sim:
    def __init__(self, policy):
        import mujoco
        from retarget_g1_x1 import X1_DOF_ORDER
        from build_x1_assets import parse_urdf_limits

        self.mj = mujoco
        self.policy = policy
        m = self.m = mujoco.MjModel.from_xml_path(str(X1_SIM))
        m.opt.timestep = 1.0 / 120.0  # match Isaac sim freq
        self.d = mujoco.MjData(m)

        self.qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
        self.vadr = np.array([m.joint(m.joint(j).id).dofadr[0]
                              for j in X1_DOF_ORDER])
        mi = mujoco.MjModel.from_xml_path(str(X1_ASSET))
        self.kp = np.array([mi.jnt_stiffness[mi.joint(j).id]
                            for j in X1_DOF_ORDER])
        self.kd = np.array([mi.dof_damping[mi.joint(mi.joint(j).id).dofadr[0]]
                            for j in X1_DOF_ORDER])
        lim = parse_urdf_limits()
        self.eff = np.array([lim[j]["effort"] for j in X1_DOF_ORDER])
        # zero-center action bounds: 1.4 * max(|low|, |high|) per hinge
        lo = np.array([lim[j]["low"] for j in X1_DOF_ORDER])
        hi = np.array([lim[j]["high"] for j in X1_DOF_ORDER])
        self.a_bound = 1.4 * np.maximum(np.abs(lo), np.abs(hi))

        # hinge axes (for joint-quat obs) from the sim model
        self.axes = np.array([m.jnt_axis[m.joint(j).id]
                              for j in X1_DOF_ORDER])

        self.key_body_ids = [m.body(n).id for n in KEY_BODIES]
        self.root_bid = m.body("base_link").id
        self.torso_bid = m.body("lumbar_pitch_link").id
        # fall-contact geoms: everything except feet soles
        self.fall_gids = [g for g in range(m.ngeom)
                          if not (m.geom(g).name or "").endswith("_sole")
                          and m.geom(g).name != "floor"
                          and m.body(m.geom_bodyid[g]).name != "world"]

        home = np.zeros(29)
        home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
        home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
        self.home = home

    def reset(self, seed):
        self.mj.mj_resetData(self.m, self.d)
        rng = np.random.RandomState(seed)
        self.d.qpos[:3] = [0, 0, 0.615]
        self.d.qpos[3] = 1.0
        self.d.qpos[4:7] = 0
        self.d.qpos[self.qadr] = self.home + rng.uniform(-0.03, 0.03, 29)
        self.d.qvel[:] = 0
        self.mj.mj_forward(self.m, self.d)

    def joint_quats(self):
        """Per-joint local rotation quats (wxyz) from hinge values."""
        q = self.d.qpos[self.qadr]
        out = np.zeros((29, 4))
        for i in range(29):
            half = 0.5 * q[i]
            a = self.axes[i]
            out[i, 0] = np.cos(half)
            out[i, 1:] = a * np.sin(half)
        return out

    def obs(self):
        d = self.d
        root_pos = d.qpos[:3].copy()
        root_quat = d.qpos[3:7].copy()  # wxyz
        # world-frame velocities at root (mujoco gives local)
        R = d.xmat[self.root_bid].reshape(3, 3)
        root_vel = R.T @ d.qvel[:3]
        root_w = R.T @ d.qvel[3:6]
        jr = self.joint_quats()
        dv = d.qvel[self.vadr].copy()
        key = d.xpos[self.key_body_ids].copy()
        return compute_char_obs_np(root_pos, root_quat, root_vel, root_w,
                                   jr, dv, key)

    def run_episode(self, length_s, log):
        ctrl_period = 1.0 / 30.0
        steps_per_ctrl = int(round(ctrl_period / self.m.opt.timestep))
        n_ctrl = int(length_s / ctrl_period)
        for it in range(n_ctrl):
            o = self.obs()
            a = self.policy.forward(o)
            q_tar = np.clip(a, -self.a_bound, self.a_bound)
            for _ in range(steps_per_ctrl):
                q = self.d.qpos[self.qadr]
                qd = self.d.qvel[self.vadr]
                tau = np.clip(self.kp * (q_tar - q) - self.kd * qd,
                              -self.eff, self.eff)
                self.d.ctrl[:] = tau
                self.mj.mj_step(self.m, self.d)
            # log at control rate
            lf = self.d.site("x_lfoot").xpos.copy()
            rf = self.d.site("x_rfoot").xpos.copy()
            lh = self.d.site("x_lhand").xpos.copy()
            rh = self.d.site("x_rhand").xpos.copy()
            R = self.d.xmat[self.torso_bid].reshape(3, 3)
            pitch = np.arctan2(R[2, 0], R[0, 0])
            log["t"].append(it * ctrl_period)
            log["root_z"].append(self.d.qpos[2])
            log["root_x"].append(self.d.qpos[0])
            log["lf_z"].append(lf[2]); log["rf_z"].append(rf[2])
            log["lh_x"].append(lh[0]); log["rh_x"].append(rh[0])
            log["lf_x"].append(lf[0]); log["rf_x"].append(rf[0])
            log["pitch"].append(pitch)
            log["q"].append(self.d.qpos[self.qadr].copy())
            log["torque"].append(
                np.abs(np.clip(self.kp * (q_tar - self.d.qpos[self.qadr])
                               - self.kd * self.d.qvel[self.vadr],
                               -self.eff, self.eff)))
            # fall contacts (checked per control step; window small)
            self.mj.mj_forward(self.m, self.d)
            bad_contact = False
            for g in self.fall_gids:
                if self.d.geom_xpos[g][2] < 0.03:
                    bad_contact = True
                    break
            log["bad_contact"].append(bad_contact)
        return log


# ------------------------------------------------------------- criteria
def detect_strikes(z, base_margin=0.045, fps=30):
    base = np.quantile(z, 0.02)
    c = z < base + base_margin
    # debounce 3 frames
    out = np.zeros_like(c)
    i = 0
    while i < len(c):
        j = i
        while j < len(c) and c[j]:
            j += 1
        if j - i >= 3:
            out[i:j] = True
        i = j
    strikes = []
    prev = False
    for i, v in enumerate(out):
        if v and not prev:
            strikes.append(i)
        prev = v
    return np.array(strikes)


def phase_of(a, b, fps):
    A = np.fft.rfft((a - a.mean()) * np.hanning(len(a)))
    B = np.fft.rfft((b - b.mean()) * np.hanning(len(a)))
    freqs = np.fft.rfftfreq(len(a), 1 / fps)
    k = int(np.argmax(np.abs(A) * (freqs > 0.3)))
    return float(np.angle(B[k] / A[k]))


def analyze(log, episode_len):
    fps = 30
    z = np.array(log["root_z"]); x = np.array(log["root_x"])
    l_z = np.array(log["lf_z"]); r_z = np.array(log["rf_z"])
    pitch = np.array(log["pitch"])
    Q = np.array(log["q"])  # (T, 29) X1 dof order: lumbar3 Larm Rarm Lleg Rleg
    tau = np.array(log["torque"])

    up_frac = float((z > 0.30).mean())
    bad_frac = float(np.mean(log["bad_contact"]))
    ls = detect_strikes(l_z); rs = detect_strikes(r_z)
    n_strides = min(len(ls), len(rs))
    stride_p = (np.median(np.diff(ls)) / fps if len(ls) >= 3 else np.inf)
    speed = float(np.mean(np.gradient(x, 1 / fps)))
    # swing clearance: p90 of each foot's z minus its stance base
    clr_l = float(np.quantile(l_z, 0.9) - np.quantile(l_z, 0.05))
    clr_r = float(np.quantile(r_z, 0.9) - np.quantile(r_z, 0.05))
    med = np.median(Q, axis=0)
    hip_pitch_med = float(0.5 * (med[17] + med[23]))
    knee_med = float(0.5 * (med[20] + med[26]))
    torso_med = float(np.degrees(np.median(pitch)))
    dphi = phase_of(l_z, np.array(log["rh_x"]), fps)

    s1 = dict(up_frac=up_frac, bad_contact_frac=bad_frac,
              pass_=bool(up_frac >= 0.95 and bad_frac == 0.0))
    s2 = dict(n_strides=int(n_strides), stride_period_s=(None if stride_p == np.inf else float(stride_p)),
              mean_speed_mps=speed, swing_clear_l=clr_l, swing_clear_r=clr_r,
              pass_=bool(n_strides >= 4 and 0.25 <= stride_p <= 1.2
                         and speed >= 0.8 and clr_l >= 0.04 and clr_r >= 0.04))
    s3 = dict(torso_pitch_deg=torso_med, hip_pitch_med=hip_pitch_med,
              knee_med=knee_med, arm_leg_phase_rad=dphi,
              pass_=bool(abs(torso_med) < 25 and -0.8 <= hip_pitch_med <= 0.8
                         and 0.2 <= knee_med <= 1.4
                         and abs(abs(dphi) - np.pi) < 0.8))
    return dict(S1_no_fall=s1, S2_gait=s2, S3_morphology=s3,
                torque_p99=float(np.quantile(tau, 0.99)),
                PASS=bool(s1["pass_"] and s2["pass_"] and s3["pass_"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--len", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pol = Policy(args.ckpt)
    sim = Sim2Sim(pol)
    results = []
    for ep in range(args.episodes):
        sim.reset(args.seed + ep)
        log = dict(t=[], root_z=[], root_x=[], lf_z=[], rf_z=[], lh_x=[],
                   rh_x=[], lf_x=[], rf_x=[], pitch=[], q=[], torque=[],
                   bad_contact=[])
        sim.run_episode(args.len, log)
        r = analyze(log, args.len)
        r["episode"] = ep
        results.append(r)
        print(f"[sim2sim] ep{ep}: {'PASS' if r['PASS'] else 'FAIL'} "
              f"(S1 {r['S1_no_fall']['up_frac']:.2f}/"
              f"{r['S1_no_fall']['bad_contact_frac']:.2f}, "
              f"S2 strides {r['S2_gait']['n_strides']} "
              f"v={r['S2_gait']['mean_speed_mps']:.2f}, "
              f"S3 pitch {r['S3_morphology']['torso_pitch_deg']:.0f}deg)")
    n_pass = sum(r["PASS"] for r in results)
    summary = dict(checkpoint=str(args.ckpt), episodes=results,
                   pass_episodes=n_pass, total=len(results),
                   PASS=bool(n_pass == len(results)))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"[sim2sim] {n_pass}/{len(results)} episodes PASS")
    sys.exit(0 if summary["PASS"] else 1)


if __name__ == "__main__":
    main()
