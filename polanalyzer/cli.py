"""命令行辅助工具。

注意：图形界面（python -m polanalyzer.gui.app）才是偏振估计的主入口，
命令行仅用于批处理拟合、合成数据生成与结果导出，不提供交互式像素检查。

子命令
------
synthetic  生成可控合成观测（含噪声/遮挡/运动）并拟合，输出核对报告
fit        按 JSON 清单载入真实角度图像，配准、共同曝光解释后拟合
report     打印拟合状态统计与可靠区域比例
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .engine import FitConfig, fit_pixels
from .fonts import configure_fonts
from .io_stack import (export_fit_npz, load_stack, register_stack)
from .masks import ALL_STATES, STATE_OK
from .synthetic import demo_truth, make_scene


def _state_summary(result) -> dict:
    states, counts = np.unique(result.state, return_counts=True)
    summary = {s: 0 for s in ALL_STATES}
    summary.update(dict(zip(states.tolist(), counts.tolist())))
    total = int(np.prod(result.state.shape))
    summary["total_pixels"] = total
    summary["reliable_fraction"] = (
        float((result.state == STATE_OK).sum()) / max(total, 1))
    return summary


def _print_summary(summary: dict) -> None:
    print("== 拟合状态统计 ==")
    for st in ALL_STATES:
        print(f"  {st:22s}: {summary[st]}")
    print(f"  可靠区域比例          : {summary['reliable_fraction']:.3f}")


def cmd_synthetic(args) -> int:
    A, B, C = demo_truth()
    scene = make_scene(
        A, B, C, args.angles,
        saturation=args.saturation, noise_sigma=args.noise,
        rng_seed=args.seed,
        occluder_cells=[(slice(12, 22), slice(14, 26), [0, 1])],
        motion_cells=[(slice(48, 58), slice(60, 74), [3])])
    cfg = FitConfig(saturation=scene.saturation,
                    noise_sigma=scene.noise_sigma,
                    weak_snr=args.weak_snr)
    result = fit_pixels(scene.intensity, scene.angles_rad, cfg,
                        motion_mask=scene.motion_mask,
                        missing_mask=scene.missing_mask)
    summary = _state_summary(result)
    _print_summary(summary)

    ok = result.state == STATE_OK
    R_true = np.hypot(scene.truth_B, scene.truth_C)
    print("== 与已知真值的核对（仅可靠像素）==")
    print(f"  A  误差 中位数={np.median(result.A[ok] - scene.truth_A[ok]):.3f}"
          f"  RMS={np.sqrt(np.mean((result.A[ok] - scene.truth_A[ok])**2)):.3f}")
    print(f"  R  误差 中位数={np.median(result.R[ok] - R_true[ok]):.3f}"
          f"  RMS={np.sqrt(np.mean((result.R[ok] - R_true[ok])**2)):.3f}")
    pred_err = result.predicted[ok] - scene.clean[ok]
    print(f"  预测强度误差 RMS={np.sqrt(np.mean(pred_err**2)):.3f}"
          f"（注入噪声 sigma={args.noise}）")

    if args.out:
        unit = f"合成线性灰度 0..{args.saturation}, noise_sigma={args.noise}"
        export_fit_npz(args.out, result, unit)
        np.savez_compressed(
            args.out.replace(".npz", "_truth.npz"),
            A=scene.truth_A, B=scene.truth_B, C=scene.truth_C,
            clean=scene.clean, angles_rad=scene.angles_rad)
        print(f"已导出拟合结果与真值: {args.out}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    return 0


def cmd_fit(args) -> int:
    stack = load_stack(args.manifest, read_noise=args.read_noise)
    intensity = stack.intensity
    motion_mask = None
    missing_mask = None

    if not stack.is_color and args.register:
        intensity, shifts, missing_mask = register_stack(intensity)
        print(f"配准平移 (dy,dx): {shifts.tolist()}")

    if np.ndim(stack.saturation) > 0:
        saturation = float(np.min(stack.saturation))
        print("警告: 各帧饱和值不同，采用最小饱和值以保证排除规则保守")
    else:
        saturation = float(stack.saturation)
    noise = (float(np.mean(stack.noise_sigma))
             if np.ndim(stack.noise_sigma) else float(stack.noise_sigma))

    cfg = FitConfig(saturation=saturation, noise_sigma=noise,
                    weak_snr=args.weak_snr)
    result = fit_pixels(intensity, stack.angles_rad, cfg,
                        motion_mask=motion_mask,
                        missing_mask=missing_mask)
    _print_summary(_state_summary(result))
    if args.out:
        export_fit_npz(args.out, result, stack.unit)
        print(f"已导出: {args.out}（量纲: {stack.unit}）")
    return 0


def cmd_report(args) -> int:
    data = np.load(args.npz, allow_pickle=True)
    states = data["state"]
    unique, counts = np.unique(states, return_counts=True)
    total = states.size
    for st, cnt in zip(unique.tolist(), counts.tolist()):
        print(f"  {st:22s}: {cnt} ({cnt / total:.3f})")
    print(f"  量纲: {str(data['unit'])}")
    print(f"  角度(度): {np.rad2deg(data['angles_rad']).tolist()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polanalyzer",
        description="多角度线性偏振分量分析的命令行辅助工具"
                    "（交互式偏振估计请使用图形界面 pol-gui）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("synthetic", help="生成合成观测并核对拟合")
    p_syn.add_argument("--angles", type=float, nargs="+",
                       default=[0, 30, 60, 90, 120, 150])
    p_syn.add_argument("--noise", type=float, default=8.0)
    p_syn.add_argument("--saturation", type=float, default=4095.0)
    p_syn.add_argument("--weak-snr", type=float, default=3.0)
    p_syn.add_argument("--seed", type=int, default=20261113)
    p_syn.add_argument("--out", type=str, default=None)
    p_syn.add_argument("--report", type=str, default=None)
    p_syn.set_defaults(func=cmd_synthetic)

    p_fit = sub.add_parser("fit", help="按清单拟合真实角度图像")
    p_fit.add_argument("manifest")
    p_fit.add_argument("--read-noise", type=float, default=1.0)
    p_fit.add_argument("--weak-snr", type=float, default=3.0)
    p_fit.add_argument("--register", action="store_true",
                       help="对灰度栈做相位相关整像素配准")
    p_fit.add_argument("--out", type=str, default=None)
    p_fit.set_defaults(func=cmd_fit)

    p_rep = sub.add_parser("report", help="读取 NPZ 并打印状态报告")
    p_rep.add_argument("npz")
    p_rep.set_defaults(func=cmd_report)
    return parser


def main(argv=None) -> int:
    configure_fonts()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
