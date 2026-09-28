"""Build v4 low-gain assets (x1_v4.xml / x1_sim_v4.xml) from the fixed-asset
v3 files (md5-anchored generation: 796e41a6e1e0).

v4 gain design (rationale: X1_stable sim2sim success profile + hardware AMP
policy profile; see FINAL_REPORT.md section 4.3):
  - The 2.4 rad/s PhysX-TGS-vs-MuJoCo solver residual becomes a destabilizing
    torque when multiplied by high kd/kp (kd45 -> 108 N*m residual injection).
    X1_stable absorbed the same residual with ankle kd 0.5 / knee kd 10.
  - Per-joint map (kp/kd, N*m / N*m*s):
      lumbar_yaw/roll   375/37.5 -> 120/4
      lumbar_pitch      450/45.0 -> 150/5
      shoulder_*/elbow*  50/5.0  ->  40/2
      wrist_*            25/2.5  ->  20/1
      hip_pitch         450/45.0 -> 150/4
      hip_roll/yaw      375/37.5 -> 120/3
      knee_pitch        450/45.0 -> 150/8
      ankle_pitch/roll  200/20.0 ->  50/1
  - Damping ratios: hip zeta ~0.13, knee ~0.6, ankle ~0.3 (sane, sprint-capable).
  - armature / inertia / ranges untouched (model-parity audit TASK_20260928_013
    showed 100% match between Isaac and MuJoCo on those; keep it that way).

Usage: python tools/x1_pipeline/build_x1_v4_assets.py
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_MD5 = "eaa2715788be"  # fixed-asset x1.xml (v3, sole-box fixed @ d44ec33)

# keys are matched by SUFFIX (left_/right_ prefixed joints in the XML)
GAIN_MAP = {
    "lumbar_yaw_joint": (120.0, 4.0),
    "lumbar_roll_joint": (120.0, 4.0),
    "lumbar_pitch_joint": (150.0, 5.0),
    "shoulder_pitch_joint": (40.0, 2.0),
    "shoulder_roll_joint": (40.0, 2.0),
    "shoulder_yaw_joint": (40.0, 2.0),
    "elbow_pitch_joint": (40.0, 2.0),
    "elbow_yaw_joint": (40.0, 2.0),
    "wrist_pitch_joint": (20.0, 1.0),
    "wrist_roll_joint": (20.0, 1.0),
    "hip_pitch_joint": (150.0, 4.0),
    "hip_roll_joint": (120.0, 3.0),
    "hip_yaw_joint": (120.0, 3.0),
    "knee_pitch_joint": (150.0, 8.0),
    "ankle_pitch_joint": (50.0, 1.0),
    "ankle_roll_joint": (50.0, 1.0),
}


def _lookup(joint_name):
    for k, v in GAIN_MAP.items():
        if joint_name == k or joint_name.endswith("_" + k):
            return v
    return None


def sub_gains(xml: str):
    n = 0

    def repl(m):
        nonlocal n
        g = _lookup(m.group("name"))
        if g is None:
            return m.group(0)
        kp, kd = g
        n += 1
        return (m.group("head") +
                f'stiffness="{kp}" damping="{kd}"' +
                m.group("tail"))

    out = re.sub(
        r'(?P<head><joint name="(?P<name>[^"]+)"[^>]*?)'
        r'stiffness="[\d.]+" damping="[\d.]+"'
        r'(?P<tail>[^>]*>)',
        repl, xml)
    return out, n


def main():
    src = ROOT / "data/assets/x1/x1.xml"
    md5 = hashlib.md5(src.read_bytes()).hexdigest()[:12]
    if md5 != SRC_MD5:
        print(f"[v4] WARNING: x1.xml md5 {md5} != expected {SRC_MD5}"
              " (still proceeding, gains swap is name-keyed)")
    for src_name, dst_name in [("x1.xml", "x1_v4.xml"),
                               ("x1_sim.xml", "x1_sim_v4.xml")]:
        s = ROOT / "data/assets/x1" / src_name
        d = ROOT / "data/assets/x1" / dst_name
        out, n = sub_gains(s.read_text())
        if n != 29:
            print(f"[v4] ERROR: expected 29 joint gain substitutions in "
                  f"{src_name}, got {n}"); sys.exit(1)
        d.write_text(out)
        print(f"[v4] {d.name}: {n} joints re-gained, md5 "
              f"{hashlib.md5(d.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
