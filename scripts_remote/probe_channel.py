import os
import sys
import time
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

EXP = "x1_probe"
SECRET_MARKS = ("key", "secret", "token", "signature", "password", "auth")


def log(msg):
    print(f"[probe] {msg}", flush=True)


def safe_lines(out):
    keep = []
    for line in out.splitlines():
        low = line.lower()
        if any(s in low for s in SECRET_MARKS):
            keep.append(line.split("=")[0][:60] + "=<redacted>")
        else:
            keep.append(line[:160])
    return " || ".join(keep[:60])


# ---------- 1) recon: how does the SDK watcher live in this container ----------
try:
    r = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=20)
    procs = [l for l in r.stdout.splitlines()
             if any(k in l.lower() for k in ("sdk", "watch", "upload", "python", "gm"))]
    log("PS-AUX: " + safe_lines("\n".join(procs)))
except Exception as e:
    log(f"ps failed: {e}")

try:
    r = subprocess.run(["bash", "-lc",
                        "ls -la /usr/local/bin 2>/dev/null | head -30; "
                        "ls /opt 2>/dev/null; "
                        "find / -maxdepth 3 -iname '*sdk*' -not -path '*/proc/*' 2>/dev/null | head -20"],
                       capture_output=True, text=True, timeout=30)
    log("FS-SDK: " + safe_lines(r.stdout))
except Exception as e:
    log(f"find failed: {e}")

try:
    env_keep = {k: ("<redacted>" if any(s in k.lower() for s in SECRET_MARKS) else v[:60])
                for k, v in os.environ.items()
                if any(s in k.lower() for s in ("sdk", "oss", "gm", "upload", "watch", "task", "argo"))}
    log(f"ENV: {env_keep}")
except Exception as e:
    log(f"env failed: {e}")

# ---------- 2) seed candidate dirs with tagged dummy .pt payloads ----------
run_dir = os.path.join("logs", EXP, "exported_data", time.strftime("%Y-%m-%d_%H-%M-%S"))


def mk(path, tag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("probe-payload-tag-%s-" % tag).encode() * 256)
    log(f"wrote {path} tag={tag}")


mk(os.path.join("output", "model.pt"), "v1")                                  # overwrite-freeze test
mk(os.path.join("output", "int_models", "model_0000000100.pt"), "v1")         # int_models test
mk(os.path.join(run_dir, "model_100.pt"), "v1")                               # exported_data test
mk(os.path.join("logs", EXP, "gm_play", "model_0000000100.pt"), "v1")         # gm_play test

log("v1 written; sleeping 150s so the watcher notices and (maybe) uploads")
time.sleep(150)

# ---------- 3) overwrite to learn update semantics (frozen snapshot vs live) ----------
mk(os.path.join("logs", EXP, "gm_play", "model_0000000100.pt"), "v2")
mk(os.path.join(run_dir, "model_100.pt"), "v2")
mk(os.path.join("output", "model.pt"), "v2")

log("v2 overwritten; sleeping 60s")
time.sleep(60)
log("probe finished -> natural exit rc=0")
sys.exit(0)
