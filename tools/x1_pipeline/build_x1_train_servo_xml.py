"""Build the MuJoCo-native TRAINING variant of x1_sim_v4.xml (SW native route
step 2a) and verify parity against the r4 anchor dump gains (step 2b).

Design (pdx2-mirror, harness bit-identical):
  - actuators: 29 <position> servos, gainprm=kp_i (kv=0 -> torque =
    kp*(q_tar - q), exactly the harness explicit-kp formula), forcerange =
    +-tlim_i (the harness np.clip), ctrlrange wide [-10, 10] (target is NOT
    clipped in the harness).
  - joint dof_damping = kd_i (implicit, same as harness), frictionloss = 0.
  - dt = 1/120.

With this, one control step = set ctrl=q_tar once + 4x mj_step, which makes
the whole batch rollout a pure mujoco.rollout call (threaded) - and the
servo torque is computed by mj_step from the CURRENT q each substep, exactly
like the harness substep loop.

Parity check: kp/kd/tlim read back from the built model vs the r4 anchor
dump (isaac_traj_v4_solver_tgs4_0_gpu.pt, X1_DOF_ORDER).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

SRC = REPO / "data/assets/x1/x1_sim_v4.xml"
DST = REPO / "data/assets/x1/x1_train_servo.xml"


def build():
    ref = torch.load(REPO / "output/remote_ckpt/"
                     "isaac_traj_v4_solver_tgs4_0_gpu.pt",
                     map_location="cpu", weights_only=False)
    kp = ref["kp"].numpy().astype(np.float64)
    kd = ref["kd"].numpy().astype(np.float64)
    tlim = ref["tlim"].numpy().astype(np.float64)

    m_src = mujoco.MjModel.from_xml_path(str(SRC))
    jid = {n: m_src.joint(n).id for n in X1_DOF_ORDER}
    vadr = np.array([m_src.joint(n).dofadr for n in X1_DOF_ORDER])

    acts = []
    for i, name in enumerate(X1_DOF_ORDER):
        # <general> exact mirror of harness formula tau = kp*(tar - q):
        # force = gainprm*ctrl + biasprm[0] + biasprm[1]*length
        #       = kp*ctrl - kp*q  (dyntype none: NO ctrl filtering - the
        # <position> shortcut defaults timeconst=0.02 which low-pass filters
        # the target and broke bit-identity, first attempt dev 19.5)
        acts.append(
            f'<general name="servo_{name}" joint="{name}" '
            f'dyntype="none" gaintype="fixed" biastype="affine" '
            f'gainprm="{kp[i]:.6g}" biasprm="0 -{kp[i]:.6g}" '
            f'ctrlrange="-10 10" '
            f'forcerange="-{tlim[i]:.6g} {tlim[i]:.6g}"/>')
    actuator_block = "\n    ".join(acts)

    src_txt = SRC.read_text()
    # replace the whole <actuator> ... </actuator> block
    import re
    src_txt = re.sub(r"<actuator>.*?</actuator>",
                     f"<actuator>\n    {actuator_block}\n  </actuator>",
                     src_txt, count=1, flags=re.S)
    # set joint damping=kd, frictionloss=0 per joint
    def fix_joint_attrs(match):
        tag = match.group(0)
        jname = re.search(r'name="([^"]+)"', tag).group(1)
        if jname not in jid:
            return tag
        i = X1_DOF_ORDER.index(jname)
        tag = re.sub(r'damping="[^"]*"', f'damping="{kd[i]:.6g}"', tag)
        tag = re.sub(r'frictionloss="[^"]*"', 'frictionloss="0"', tag)
        return tag
    src_txt = re.sub(r'<joint [^>]*name="[^"]*"[^>]*/>', fix_joint_attrs,
                     src_txt)
    src_txt = re.sub(r'timestep="[^"]*"', 'timestep="0.00833333333333333"',
                     src_txt)
    DST.write_text(src_txt)
    print(f"written {DST}")


def verify():
    ref = torch.load(REPO / "output/remote_ckpt/"
                     "isaac_traj_v4_solver_tgs4_0_gpu.pt",
                     map_location="cpu", weights_only=False)
    kp = ref["kp"].numpy().astype(np.float64)
    kd = ref["kd"].numpy().astype(np.float64)
    tlim = ref["tlim"].numpy().astype(np.float64)

    m = mujoco.MjModel.from_xml_path(str(DST))
    print(f"built model: nq {m.nq} nv {m.nv} nu {m.nu} dt {m.opt.timestep}")
    assert m.nu == 29
    jname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    kp_a = np.array([m.actuator_gainprm[i, 0] for i in range(29)])
    lo = np.array([m.actuator_forcerange[i, 0] for i in range(29)])
    hi = np.array([m.actuator_forcerange[i, 1] for i in range(29)])
    # actuator order in the written XML == X1_DOF_ORDER
    names_ok = [jname(i) for i in range(29)] == [f"servo_{n}" for n in X1_DOF_ORDER]
    dyn_none = all(int(m.actuator_dyntype[i]) == 0 for i in range(29))
    print(f"dyntype none (no ctrl filter): {dyn_none}")
    kd_j = np.array([float(m.dof_damping[int(m.joint(n).dofadr[0])]) for n in X1_DOF_ORDER])
    fl_j = np.array([float(m.dof_frictionloss[int(m.joint(n).dofadr[0])]) for n in X1_DOF_ORDER])
    print(f"actuator names == X1 order: {names_ok}")
    print(f"kp parity max dev: {np.abs(kp_a - kp).max():.3e}")
    print(f"forcerange parity (hi vs tlim) max dev: {np.abs(hi - tlim).max():.3e}"
          f" | symmetric: {np.abs(lo + hi).max():.3e}")
    print(f"kd parity max dev: {np.abs(kd_j - kd).max():.3e}")
    print(f"frictionloss all zero: {np.abs(fl_j).max() == 0}")
    # physics parity vs source model
    m0 = mujoco.MjModel.from_xml_path(str(SRC))
    for f in ("body_mass", "body_inertia", "dof_armature"):
        same = np.array_equal(np.asarray(getattr(m, f)),
                              np.asarray(getattr(m0, f)))
        print(f"{f} identical to source: {same}")
    kv = np.array([m.actuator_biasprm[i, 2] for i in range(29)])
    print(f"servo kv all zero: {np.abs(kv).max() == 0} (harness has no kv)")


if __name__ == "__main__":
    build()
    verify()
