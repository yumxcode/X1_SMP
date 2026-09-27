"""Model-parity audit: dump Isaac's parsed body masses / inertias /
dof armature / friction, to diff against MuJoCo locally.

Prints (and saves) per-body rigid body properties and per-dof props.
"""
import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
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
import envs.env_builder as env_builder  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

env = env_builder.build_env("data/envs/smp_x1_env_v3.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
char_id = env._get_char_id()
e = env._engine
gym = e._gym
env_ptr = e.get_env(0)

out = {}

# rigid body props (mass, inertia diag, com)
rb_props = gym.get_actor_rigid_body_properties(env_ptr, char_id)
body_names = gym.get_actor_rigid_body_names(env_ptr, char_id)
rows = []
print("[audit] rigid bodies:", flush=True)
for nm, p in zip(body_names, rb_props):
    inertia = p.inertia  # Mat33 of Vec3 rows
    idiag = (inertia.row0.x + inertia.row1.y + inertia.row2.z) / 3.0  # fallback
    try:
        idiag = (float(inertia.row0.x), float(inertia.row1.y),
                 float(inertia.row2.z))
    except Exception:
        idiag = (-1.0, -1.0, -1.0)
    rows.append((nm, float(p.mass), idiag))
    print(f"  {nm:32s} mass {p.mass:8.4f} inertia_diag "
          f"({idiag[0]:7.4f},{idiag[1]:7.4f},{idiag[2]:7.4f})",
          flush=True)
out["bodies"] = rows

dof_props = gym.get_actor_dof_properties(env_ptr, char_id)
dof_names = gym.get_actor_dof_names(env_ptr, char_id)
print("\n[audit] dof props:", flush=True)
drows = []
for nm, i in zip(dof_names, range(len(dof_names))):
    drows.append((nm, float(dof_props["armature"][i]),
                  float(dof_props["friction"][i]),
                  float(dof_props["lower"][i]), float(dof_props["upper"][i])))
    print(f"  {nm:32s} armature {dof_props['armature'][i]:8.4f} "
          f"friction {dof_props['friction'][i]:7.4f} "
          f"range [{dof_props['lower'][i]:+.3f},{dof_props['upper'][i]:+.3f}]",
          flush=True)
out["dofs"] = drows

torch.save({k: v for k, v in out.items()},
           os.path.join(ROOT, "output", "isaac_model_audit.pt"))
print("\n[audit] saved output/isaac_model_audit.pt", flush=True)
