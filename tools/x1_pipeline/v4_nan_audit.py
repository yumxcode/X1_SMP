"""E-v4-NAN-01: how often and when does the MuJoCo sim2sim rollout hit
numerical instability (NaN/Inf in qacc/qvel)?

The E-v4-RESET-01 runs emitted 'Nan, Inf or huge value in QACC at DOF 3/7'
warnings. If divergence happens BEFORE the physical fall (root_z drop), the
fall is a numerical artifact, not dynamics. Log per-step state health.

Also record: time of first huge |qvel| (>50 rad/s), first NaN, first
root_z<0.30, and their ordering.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import sim2sim_validate as SV  # noqa: E402


def run(sim, length_s=5.0):
    steps = int(round((1 / 30.0) / sim.m.opt.timestep))
    n_ctrl = int(length_s * 30)
    first = dict(nan=None, huge_qvel=None, fell=None)
    for it in range(n_ctrl):
        o = sim.obs()
        a = sim.policy.forward(o)
        if not np.all(np.isfinite(a)):
            first["nan"] = min(it / 30.0, first["nan"] or 99)
            break
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for k in range(steps):
            q = sim.d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            sim.d.ctrl[:] = tau
            sim.mj.mj_step(sim.m, sim.d)
            t = (it * steps + k + 1) * sim.m.opt.timestep
            qvel = sim.d.qvel
            if not np.all(np.isfinite(qvel)) or not np.all(np.isfinite(sim.d.qacc)):
                first["nan"] = round(t, 3)
                return first
            if first["huge_qvel"] is None and np.abs(qvel).max() > 50:
                first["huge_qvel"] = round(t, 3)
        if sim.d.qpos[2] < 0.30:
            first["fell"] = round((it + 1) / 30.0, 2)
            return first
    return first


def main():
    ckpt = REPO / "output/remote_ckpt/smpv4f_it2100.pt"
    pol = SV.Policy(str(ckpt))
    SV.Sim2Sim.ENV_YAML_REL = "data/envs/smp_x1_env_v4.yaml"
    sim = SV.Sim2Sim(pol, sim_xml="data/assets/x1/x1_sim_v4.xml",
                     asset_xml="data/assets/x1/x1_v4.xml")
    rows = []
    for seed in range(101, 121):
        sim.reset(seed=seed)
        r = run(sim)
        order = []
        for k in ("huge_qvel", "nan", "fell"):
            if r[k] is not None:
                order.append(f"{k}@{r[k]}")
        rows.append(dict(seed=seed, **r))
        print(f"seed {seed}: {' | '.join(order) if order else 'UP 5s clean'}")
    print("PROBE A: before stats")
    nan_n = 0
    huge_n = 0
    huge_before_fell = 0
    print("PROBE B: after init", nan_n, huge_n, huge_before_fell)
    for r in rows:
        if r.get("nan") is not None:
            nan_n += 1
        if r.get("huge_qvel") is not None:
            huge_n += 1
            fell = r.get("fell")
            if fell is None or r["huge_qvel"] < fell:
                huge_before_fell += 1
    print(f"\nNaN episodes: {nan_n}/20 | huge|qvel| episodes: {huge_n}/20 "
          f"| huge BEFORE fall: {huge_before_fell}/20")
    (REPO / "output/v4_nan_audit.json").write_text(
        json.dumps(dict(experiment="E-v4-NAN-01", rows=rows,
                        nan_n=nan_n, huge_n=huge_n,
                        huge_before_fell=huge_before_fell), indent=1))


if __name__ == "__main__":
    main()
