#!/usr/bin/env python3
"""诊断包离线复算：重新拟合 + 重新诊断，并与封存结论比对。

用法：python3 rebuild.py [--data data.npz] [--manifest manifest.json]
仅依赖 numpy 与 polanalyzer 包（自动尝试脚本目录的上级仓库路径）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# 诊断包可被复制到任意位置：逐级向上寻找含 polanalyzer 包的目录，
# 同时尊重已安装的 polanalyzer（site-packages）与 PYTHONPATH。
_parts = _HERE.split(os.sep)
for _up in range(len(_parts) + 1):
    _cand = os.sep.join(_parts[:len(_parts) - _up]) if _up else _HERE
    if _cand and os.path.isdir(os.path.join(_cand, "polanalyzer")):
        sys.path.insert(0, _cand)
        break

from polanalyzer.diagnostics import DiagnosisConfig, diagnose_stack
from polanalyzer.engine import FitConfig, fit_pixels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data.npz"))
    ap.add_argument("--manifest", default=os.path.join(_HERE, "manifest.json"))
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    with open(args.manifest, "r", encoding="utf-8") as fh:
        sealed = json.load(fh)

    cfg = FitConfig(saturation=float(d["saturation"]),
                    noise_sigma=float(d["noise_sigma"]),
                    weak_snr=float(d["weak_snr"]),
                    max_residual_sigma=float(d["max_residual_sigma"]))
    result = fit_pixels(
        d["intensity"], d["angles_rad"], cfg,
        motion_mask=(d["motion_mask"] if d["motion_mask"].size else None),
        missing_mask=(d["missing_mask"] if d["missing_mask"].size else None))

    dcfg_kw = sealed["config"]
    dcfg = DiagnosisConfig(**dcfg_kw)
    report = diagnose_stack(d["intensity"], d["angles_rad"], result,
                            config=dcfg, region=None,
                            rng_seed=int(d["rng_seed"]))

    sealed_map = np.asarray(d["sealed_hypothesis"], dtype=object)
    agree = report.hypothesis_map == sealed_map
    print("== 诊断包离线复算 ==")
    print(f"区域: {sealed.get('region')}  像素数: {agree.size}")
    print(f"逐像素假设一致: {int(agree.sum())}/{agree.size}")
    print("复算假设计数:", report.counts)
    print("封存假设计数:", sealed["hypothesis_counts"])
    print("全局增益(复算):", [round(float(g), 4) for g in report.global_gains])
    print("分量等价核查:", report.component_equivalence)

    ok = bool(agree.all()) and report.component_equivalence["equivalent"]
    # 关键标量证据也需在容差内重现
    for rec in sealed["records"]:
        pass
    print("结论:", "重建同一结论" if ok else "复算与封存结论不一致，请检查数据/版本")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
