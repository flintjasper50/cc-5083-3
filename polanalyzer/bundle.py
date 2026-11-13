"""小型诊断包的提取与脱离原界面的复算。

诊断包是一个目录：

    <name>/
      manifest.json   结论、证据、配置、逐采样点记录、区域定位
      data.npz        区域内原始观测副本、角度、遮罩、基线拟合量（浮点量纲）
      rebuild.py      独立复算脚本，仅依赖 numpy + 本仓库 polanalyzer 包

复算程序不读取任何界面状态：它从 data.npz 重新构造观测栈并跑同一套
engine.fit_pixels + diagnostics.diagnose_stack，再与 manifest.json 中
封存的结论逐像素比对，输出是否重建同一结论。
"""

from __future__ import annotations

import json
import os

import numpy as np

from .diagnostics import (DiagnosisConfig, DiagnosisReport, diagnose_stack)
from .engine import FitConfig, fit_pixels

REBUILD_SCRIPT = '''#!/usr/bin/env python3
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
    print("结论:", "重建同一结论" if ok else "复算与封存结论不一致，请检查数据/版本")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def extract_bundle(report: DiagnosisReport,
                   result,
                   intensity: np.ndarray,
                   out_dir: str,
                   name: str = "diagnostic_bundle",
                   extra_note: str = "",
                   max_records_per_hypothesis: int = 25) -> str:
    """从异常区域提取自包含诊断包。

    report.region 为 None 时以异常像素的包围盒（含 2 像素边距）作为区域，
    避免把整幅逐像素原始数据写入包内。逐采样点记录按假设类别封顶
    （默认每类 25 条，保留证据最重的样本），控制 manifest 体积；
    完整像素级结论保存在 data.npz 的 sealed_hypothesis 图中。
    """
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)

    spatial = report.hypothesis_map.shape[:2]
    if report.region is not None:
        r0, r1, c0, c1 = report.region
    else:
        from .diagnostics import HYP_CLEAN, HYP_UNFITTABLE
        abnormal = ~np.isin(report.hypothesis_map,
                            [HYP_CLEAN, HYP_UNFITTABLE])
        if not abnormal.any():
            r0, r1, c0, c1 = 0, spatial[0], 0, spatial[1]
        else:
            rows, cols = np.where(abnormal)
            pad = 2
            r0 = max(0, rows.min() - pad)
            r1 = min(spatial[0], rows.max() + pad + 1)
            c0 = max(0, cols.min() - pad)
            c1 = min(spatial[1], cols.max() + pad + 1)

    multichannel = intensity.ndim == 4
    if multichannel:
        patch = intensity[r0:r1, c0:c1, :, :]
        vmask = result.valid[r0:r1, c0:c1, :, :]
        mm = (None if result.motion_mask is None
              else result.motion_mask[r0:r1, c0:c1, :, :])
        km = (None if result.missing_mask is None
              else result.missing_mask[r0:r1, c0:c1, :, :])
    else:
        patch = intensity[r0:r1, c0:c1, :]
        vmask = result.valid[r0:r1, c0:c1, :]
        mm = (None if result.motion_mask is None
              else result.motion_mask[r0:r1, c0:c1, :])
        km = (None if result.missing_mask is None
              else result.missing_mask[r0:r1, c0:c1, :])
    # 区域诊断的假设图本身就是窗口形状；整幅诊断则按包围盒切片
    if report.region is not None:
        sealed_map = report.hypothesis_map
    else:
        sealed_map = (report.hypothesis_map[r0:r1, c0:c1, :]
                      if multichannel
                      else report.hypothesis_map[r0:r1, c0:c1])

    np.savez_compressed(
        os.path.join(path, "data.npz"),
        intensity=patch,
        angles_rad=report.angles_rad,
        valid=vmask,
        motion_mask=(np.zeros(0) if mm is None else mm),
        missing_mask=(np.zeros(0) if km is None else km),
        saturation=np.asarray(result.config.saturation),
        noise_sigma=np.asarray(result.config.noise_sigma),
        weak_snr=np.asarray(result.config.weak_snr),
        max_residual_sigma=np.asarray(result.config.max_residual_sigma),
        rng_seed=np.asarray(20261113),
        sealed_hypothesis=np.asarray(sealed_map, dtype=object),
        baseline_A=result.A[r0:r1, c0:c1],
        baseline_B=result.B[r0:r1, c0:c1],
        baseline_C=result.C[r0:r1, c0:c1],
        baseline_state=np.asarray(result.state[r0:r1, c0:c1], dtype=str))

    manifest = report.to_json_dict()
    # 每类假设只保留证据最重（chi2 最高）的有限条记录，防止 manifest 膨胀；
    # 全量逐像素假设图在 data.npz 的 sealed_hypothesis 中，不丢空间结论。
    by_hyp: dict[str, list] = {}
    for rec in report.records:
        by_hyp.setdefault(rec.hypothesis, []).append(rec)
    kept: list = []
    for hyp, recs in by_hyp.items():
        recs_sorted = sorted(
            recs,
            key=lambda x: (x.reduced_chi2 is None,
                           -(x.reduced_chi2 or 0.0)))
        kept.extend(recs_sorted[:max_records_per_hypothesis])
    kept.sort(key=lambda x: (x.row, x.col, x.channel or 0))
    manifest["records"] = [r.to_dict() for r in kept]
    manifest["records_capped_per_hypothesis"] = max_records_per_hypothesis
    manifest["total_records_before_cap"] = len(report.records)
    manifest["bundle_name"] = name
    manifest["region_bbox_rcrc"] = [int(r0), int(r1), int(c0), int(c1)]
    manifest["image_shape"] = list(spatial)
    manifest["multichannel"] = multichannel
    manifest["extra_note"] = extra_note
    with open(os.path.join(path, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    rebuild_path = os.path.join(path, "rebuild.py")
    with open(rebuild_path, "w", encoding="utf-8") as fh:
        fh.write(REBUILD_SCRIPT)
    os.chmod(rebuild_path, 0o755)
    return path


def rebuild_bundle(bundle_dir: str) -> tuple[bool, DiagnosisReport, dict]:
    """在当前进程内复算诊断包并与封存结论比对（供测试与终端核查）。"""
    data = np.load(os.path.join(bundle_dir, "data.npz"), allow_pickle=True)
    with open(os.path.join(bundle_dir, "manifest.json"),
              encoding="utf-8") as fh:
        sealed = json.load(fh)

    cfg = FitConfig(saturation=float(data["saturation"]),
                    noise_sigma=float(data["noise_sigma"]),
                    weak_snr=float(data["weak_snr"]),
                    max_residual_sigma=float(data["max_residual_sigma"]))
    mm = data["motion_mask"]
    km = data["missing_mask"]
    result = fit_pixels(
        data["intensity"], data["angles_rad"], cfg,
        motion_mask=(mm if mm.size else None),
        missing_mask=(km if km.size else None))
    dcfg = DiagnosisConfig(**sealed["config"])
    report = diagnose_stack(data["intensity"], data["angles_rad"], result,
                            config=dcfg, region=None,
                            rng_seed=int(data["rng_seed"]))
    sealed_map = np.asarray(data["sealed_hypothesis"], dtype=object)
    agree = report.hypothesis_map == sealed_map
    beta_match = np.allclose(
        np.nan_to_num(result.A), np.nan_to_num(data["baseline_A"]), atol=1e-6)
    beta_match &= np.allclose(
        np.nan_to_num(result.B), np.nan_to_num(data["baseline_B"]), atol=1e-6)
    beta_match &= np.allclose(
        np.nan_to_num(result.C), np.nan_to_num(data["baseline_C"]), atol=1e-6)
    ok = bool(agree.all()) and beta_match
    return ok, report, sealed
