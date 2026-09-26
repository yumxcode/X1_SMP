import re
import sys

sys.path.insert(0, "tools/x1_pipeline")
sys.path.insert(0, ".")
from build_x1_assets import parse_urdf_limits  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

lim = parse_urdf_limits()
xml = open("data/assets/x1/x1.xml").read()
xml_ranges = dict(re.findall(
    r'name="([a-z_0-9]+)_joint" type="hinge"[^/]*range="([^"]+)"', xml))
for j in X1_DOF_ORDER:
    u = lim.get(j, {})
    xr = xml_ranges.get(j, None)
    flag = ""
    if u and xr:
        lo, hi = u["low"], u["high"]
        xlo, xhi = map(float, xr.split())
        if abs(lo - xlo) > 1e-3 or abs(hi - xhi) > 1e-3:
            flag = f" <== MISMATCH xml=({xlo},{xhi})"
    elif not u:
        flag = " <== MISSING in urdf parse"
    print(f"{j:>32}: urdf=({u.get('low','?')},{u.get('high','?')}) "
          f"eff={u.get('effort','?')}{flag}")
