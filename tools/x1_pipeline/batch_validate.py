"""Batch validation of retargeted X1 motions -> report (JSON + markdown)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from validate_retarget import validate, render_markdown


def main():
    seg_dir = REPO_ROOT / "data/LAFAN1_g1/g1_segments"
    motion_dir = REPO_ROOT / "data/motions/x1"
    results = []
    for csv in sorted(seg_dir.glob("*.csv")):
        stem = csv.stem
        pkl = motion_dir / f"x1_{stem}.pkl"
        if not pkl.exists():
            print(f"[skip] {pkl} missing")
            continue
        print(f"[validate] {stem}")
        r = validate(csv, pkl, sample_step=4)
        r["_name"] = stem
        results.append(r)
        mark = "PASS" if r["PASS"] else "FAIL"
        fails = [k for k, v in r.items()
                 if isinstance(v, dict) and v.get("pass_") is False]
        print(f"  -> {mark} (fail: {fails})")
    out_json = REPO_ROOT / "output/x1_retarget_validation.json"
    out_md = REPO_ROOT / "output/x1_retarget_validation.md"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    out_md.write_text(render_markdown(results))
    n_pass = sum(r["PASS"] for r in results)
    print(f"\n{n_pass}/{len(results)} clips PASS -> {out_md}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
