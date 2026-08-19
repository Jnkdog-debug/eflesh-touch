#!/usr/bin/env python3
"""给旧格式 skin_frame JSON 补 half_x_mm/half_y_mm（示教实际半幅宽）."""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    if "half_x_mm" in d.get("meta", {}):
        print("已有 half 字段，无需补丁")
        return
    tl, tr, br, bl, c = np.array(d["corners_base_m"])
    d.setdefault("meta", {})
    d["meta"]["half_x_mm"] = float(min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))) * 0.5e3
    d["meta"]["half_y_mm"] = float(min(np.linalg.norm(br - tr), np.linalg.norm(tl - bl))) * 0.5e3
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"补丁完成: half_x={d['meta']['half_x_mm']:.2f}mm half_y={d['meta']['half_y_mm']:.2f}mm")


if __name__ == "__main__":
    main()
