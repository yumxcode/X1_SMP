import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def log(msg):
    print(f"[probe2] {msg}", flush=True)


def mk(path, tag):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(("probe2-payload-tag-%s-" % tag).encode() * 256)
    log(f"wrote {path} tag={tag}")


# t=0: top-level output/ unique names + late int_models write later
mk(os.path.join("output", "model_2000.pt"), "v1")            # top-level unique name
mk(os.path.join("output", "model_final.pt"), "v1")           # top-level 'final' name
mk(os.path.join("output", "ckpt_a.pt"), "v1")                # arbitrary name top-level
mk(os.path.join("output", "sub", "model_3000.pt"), "v1")     # subdir under output/

log("sleeping 90s")
time.sleep(90)

# t=90: late int_models write (fresh name, well after startup scan)
mk(os.path.join("output", "int_models", "model_0000000200.pt"), "v1")

log("sleeping 90s more")
time.sleep(90)
log("probe2 finished -> natural exit rc=0")
sys.exit(0)
