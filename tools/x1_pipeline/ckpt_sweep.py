"""Drive sim2sim validation over ALL checkpoints of a gradmotion task.

Polls `gm task model list`, downloads each PT via policUrlDown, runs
tools/x1_pipeline/sim2sim_validate.py per checkpoint (episodes can be
reduced for triage), and writes a summary table.

Usage:
  .venv/bin/python tools/x1_pipeline/ckpt_sweep.py --task TASK_xxx \
      [--episodes 2] [--len 10] [--out output/sweep_<task>.json] [--triage]
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO / "output/remote_ckpt"


def gm(args):
    import os
    key = os.environ["GM_API_KEY"]
    r = subprocess.run(["gm"] + args, capture_output=True, text=True,
                       env={**os.environ, "GM_API_KEY": key})
    if r.returncode != 0:
        raise RuntimeError(f"gm {args} failed: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--len", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = gm(["task", "model", "list", "--task-id", args.task,
            "--page", "1", "--limit", "50"])
    rows = d["data"] if isinstance(d["data"], list) else d["data"]["rows"]
    rows.sort(key=lambda r: str(r.get("createTime", "")))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, r in enumerate(rows):
        url = (r.get("policUrlDown") or "").strip()
        if not url:
            continue
        name = f"{args.task}_{i}_{r.get('fileName') or 'model'}.pt"
        local = CKPT_DIR / name
        try:
            urllib.request.urlretrieve(url, local)
        except Exception as e:
            print(f"[sweep] download failed {name}: {e}")
            continue
        cmd = [str(REPO / ".venv/bin/python"), "-u",
               str(REPO / "tools/x1_pipeline/sim2sim_validate.py"),
               "--ckpt", str(local), "--episodes", str(args.episodes),
               "--len", str(args.len)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1200)
            lines = [l for l in p.stdout.splitlines() if l.strip()]
            summary = lines[-1] if lines else "no output"
            per_ep = [l for l in lines if "ep" in l]
        except subprocess.TimeoutExpired:
            summary, per_ep = "timeout", []
        rec = dict(file=str(local), fileName=r.get("fileName"),
                   createTime=r.get("createTime"), summary=summary,
                   per_episode=per_ep)
        results.append(rec)
        print(f"[sweep] {name}: {summary}")
        for l in per_ep:
            print("        " + l)

    out = args.out or str(REPO / f"output/sweep_{args.task}.json")
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {out}")


if __name__ == "__main__":
    main()
