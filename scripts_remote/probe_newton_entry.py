"""IDEA-012r / SW-PREREG-003: Newton engine entry probe (route decision data).

Newton (newton.solvers.SolverMuJoCo, MuJoCo algorithm via Warp) vs MuJoCo
CPU harness first-step residual, using the SW-R004 protocol family:
  - deterministic reset (motion 0, time 0; class-level patch, r2 lesson)
  - open-loop q_tar replay from the r4 GPU anchor dump
    (output/remote_ckpt/isaac_traj_v4_solver_tgs4_0_gpu.pt, commit d6ca851c)
  - zero-error subprobe (q_tar pinned at q0)
  - state0 alignment self-check vs the anchor dump (< 1e-5)
  - to_np with forced .copy() (r3 aliasing lesson)

Config matrix (per SW-PREREG-003, frozen):
  A_explicit : pd_explicit, pd_kd_mode=explicit, impratio=10 (negative control)
  B_implicit : pd_explicit, pd_kd_mode=implicit, impratio=10 (MAIN criterion)
  C_pos      : pos mode, impratio=10 (free datapoint)
  D_impratio1: pd_explicit, implicit, impratio=1.0 (harness impratio align)

All at sim_freq=120 (probe engine yaml written per-config in output/).
Driver loops configs as fresh subprocesses (one sim per process).
Outputs: output/newton_probe_{tag}.pt
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF = os.path.abspath(__file__)
os.chdir(ROOT)

CONFIGS = [
    ("A_explicit", "pd_explicit", "explicit", 10),
    ("B_implicit", "pd_explicit", "implicit", 10),
    ("C_pos", "pos", None, 10),
    ("D_impratio1", "pd_explicit", "implicit", 1),
]
N_ROLLOUT = 30
N_PROBE = 8
ANCHOR_DUMP = "output/remote_ckpt/isaac_traj_v4_solver_tgs4_0_gpu.pt"


def run_child(tag, control_mode, pd_kd_mode, impratio):
    env = dict(os.environ)
    env.update({"X1_PROBE_CHILD": "1", "X1_PROBE_TAG": tag,
                "X1_PROBE_CONTROL_MODE": control_mode,
                "X1_PROBE_IMPRATIO": str(impratio)})
    if pd_kd_mode:
        env["X1_PROBE_PD_KD_MODE"] = pd_kd_mode
    print(f"[newton-probe] === {tag}: mode={control_mode} "
          f"kd={pd_kd_mode} impratio={impratio} ===", flush=True)
    proc = subprocess.run([sys.executable, SELF], env=env,
                          capture_output=True, text=True, timeout=1200)
    out = proc.stdout + proc.stderr
    for ln in out.splitlines():
        if ("[newton-probe]" in ln or "Traceback" in ln or "Error" in ln
                or "ImportError" in ln or "ModuleNotFound" in ln):
            print(ln, flush=True)
    out_path = os.path.join(ROOT, "output", f"newton_probe_{tag}.pt")
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    md5 = (hashlib.md5(open(out_path, "rb").read()).hexdigest()[:12]
           if os.path.exists(out_path) else "missing")
    print(f"[newton-probe] {tag}: {status} | dump md5 {md5}", flush=True)
    if proc.returncode != 0:
        print(f"[newton-probe] child {tag} failed; full tail:", flush=True)
        for ln in out.splitlines()[-25:]:
            print(f"  {ln}", flush=True)
    return proc.returncode == 0


if os.environ.get("X1_PROBE_CHILD", "") != "1":
    print("[newton-probe] driver start", flush=True)
    ok_all = True
    for tag, mode, kd, imp in CONFIGS:
        ok_all &= run_child(tag, mode, kd, imp)
    sys.exit(0 if ok_all else 1)

# ================= child =================
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-index", "--no-deps", "--find-links",
                       os.path.join(ROOT, "vendor_wheels"),
                       "gymnasium", "farama-notifications", "cloudpickle",
                       "diffusers", "huggingface-hub", "filelock",
                       "importlib-metadata", "zipp", "packaging",
                       "typing-extensions", "regex", "tqdm", "safetensors",
                       "requests", "matplotlib", "contourpy", "cycler",
                       "fonttools", "kiwisolver", "pyparsing",
                       "python-dateutil", "six", "tensorboardX", "protobuf"])

sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)

TAG = os.environ["X1_PROBE_TAG"]
CONTROL_MODE = os.environ["X1_PROBE_CONTROL_MODE"]
PD_KD_MODE = os.environ.get("X1_PROBE_PD_KD_MODE", "explicit")
IMPRATIO = os.environ["X1_PROBE_IMPRATIO"]
# task r1 (TASK_20260930_040) failed on GPU: CUDA graph capture uses
# conditional graph nodes requiring driver 12.4+ (node has older).
# CPU warp backend is semantics-equivalent for this probe (throughput
# irrelevant); prereg device deviation disclosed in the report.
DEVICE = os.environ.get("X1_PROBE_DEVICE", "cpu")

# environment report (fail-fast per prereg §3)
import newton  # noqa: E402
import warp  # noqa: E402
print(f"[newton-probe] newton version {getattr(newton, '__version__', '?')} "
      f"| warp version {getattr(warp, '__version__', '?')} "
      f"| warp device {warp.get_device()}", flush=True)


def _apply_fixed_reset_patch():
    import torch
    import envs.deepmimic_env as _de

    def _fixed_sample(self, n):
        ids = torch.zeros(n, dtype=torch.long, device=self._device)
        times = torch.zeros(n, dtype=torch.float, device=self._device)
        return ids, times
    _de.DeepMimicEnv._sample_motion_times = _fixed_sample


_apply_fixed_reset_patch()
import envs.env_builder as env_builder  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

# per-config engine yaml (sim_freq 120 unified; defaults differ -> written file)
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
yaml_path = os.path.join(ROOT, "output", f"newton_engine_probe_{TAG}.yaml")
lines = ["engine_name: \"newton\"", "",
         f"control_mode: \"{CONTROL_MODE}\"",
         "control_freq: 30",
         "sim_freq: 120",
         "env_spacing: 5",
         f"pd_kd_mode: \"{PD_KD_MODE}\"",
         f"impratio: {IMPRATIO}"]
open(yaml_path, "w").write("\n".join(lines) + "\n")
print(f"[newton-probe] child {TAG}: engine yaml {yaml_path}", flush=True)
for p in ("data/assets/x1/x1_v4.xml", "data/envs/smp_x1_env_v4.yaml",
          ANCHOR_DUMP):
    fp = os.path.join(ROOT, p)
    print(f"[newton-probe] md5 {p} "
          f"{hashlib.md5(open(fp,'rb').read()).hexdigest()[:12]}", flush=True)

env = env_builder.build_env("data/envs/smp_x1_env_v4.yaml", yaml_path,
                            1, DEVICE, visualize=False, record_video=False)
char_id = env._get_char_id()
e = env._engine


def to_np(x):
    # forced copy (r3 aliasing lesson; CPU-side warp/torch views)
    try:
        return torch.as_tensor(x).detach().cpu().numpy().copy()
    except Exception:
        return np.array(x)


ref = torch.load(os.path.join(ROOT, ANCHOR_DUMP), map_location="cpu",
                 weights_only=False)
q_tar_seq = ref["q_tar"].numpy().astype(np.float64)   # [40, 29]

obs0, _ = env.reset()

# state0 alignment self-check vs anchor dump frame 0 (prereg §3)
s0_root = to_np(e.get_root_pos(char_id)[0]).astype(np.float64)
s0_dof = to_np(e.get_dof_pos(char_id)[0]).astype(np.float64)
root_dev = float(np.max(np.abs(s0_root - ref["root_pos"][0].numpy())))
dof_dev = float(np.max(np.abs(s0_dof - ref["dof_pos"][0].numpy())))
print(f"[newton-probe] {TAG} state0 vs anchor: root max diff {root_dev:.2e} "
      f"| dof max diff {dof_dev:.2e} (must be < 1e-5)", flush=True)
assert root_dev < 1e-5 and dof_dev < 1e-5, "state0 misaligned; abort config"

traj = dict(root_pos=[], root_quat=[], root_vel=[], root_ang_vel=[],
            dof_pos=[], dof_vel=[], done=[])
for t in range(N_ROLLOUT):
    traj["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    traj["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    traj["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    traj["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    traj["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    traj["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    act = torch.tensor(q_tar_seq[t], dtype=torch.float32,
                       device=DEVICE).reshape(1, -1)
    nobs, r, done, info = env.step(act)
    traj["done"].append(np.asarray(to_np(done), dtype=np.float64).reshape(-1))

# zero-error probe: fresh reset, cmd pinned at q0
obs0p, _ = env.reset()
q0 = to_np(e.get_dof_pos(char_id)[0]).astype(np.float64).copy()
probe = dict(root_pos=[], root_quat=[], root_vel=[], root_ang_vel=[],
             dof_pos=[], dof_vel=[], q0=q0)
act0 = torch.tensor(q0, dtype=torch.float32, device=DEVICE).reshape(1, -1)
for t in range(N_PROBE):
    probe["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    probe["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    probe["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    probe["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    probe["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    probe["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    _ = env.step(act0)

out = {k: torch.tensor(np.stack(v)) for k, v in traj.items() if v}
out["probe"] = {k: torch.tensor(np.stack(v)) for k, v in probe.items()}
out["probe_config"] = dict(tag=TAG, control_mode=CONTROL_MODE,
                           pd_kd_mode=PD_KD_MODE, impratio=float(IMPRATIO),
                           sim_freq=120, device=DEVICE,
                           newton_version=getattr(newton, "__version__", "?"),
                           warp_version=getattr(warp, "__version__", "?"))
out["state0_check"] = dict(root_max_diff=root_dev, dof_max_diff=dof_dev,
                           anchor=ANCHOR_DUMP)
out["kp"] = ref["kp"].clone()
out["kd"] = ref["kd"].clone()
out["tlim"] = ref["tlim"].clone()
out["q_tar_seq"] = ref["q_tar"].clone()
out_path = os.path.join(ROOT, "output", f"newton_probe_{TAG}.pt")
torch.save(out, out_path)
print(f"[newton-probe] saved {out_path} ({N_ROLLOUT} steps + probe {N_PROBE}) "
      f"md5 {hashlib.md5(open(out_path,'rb').read()).hexdigest()[:12]}",
      flush=True)
