"""命令行辅助工具。

注意：图形界面（python -m polanalyzer.gui.app）才是偏振估计的主入口，
命令行仅用于批处理拟合、合成数据生成、受控故障对照与异常诊断，
不提供交互式像素检查。

子命令
------
synthetic   生成可控合成观测（含噪声/遮挡/运动）并拟合，输出核对报告
fit         按 JSON 清单载入真实角度图像，配准、共同曝光解释后拟合
report      打印拟合状态统计与可靠区域比例
fault-demo  生成角度错配/曝光变化/联合/运动四类受控对照并运行诊断，
            核对诊断结论与注入故障是否一致，可导出诊断包
diagnose    对清单或 NPZ 运行可复跑诊断，输出证据化结论与小型诊断包
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .bundle import extract_bundle, rebuild_bundle
from .diagnostics import (DIAGNOSIS_LABELS, DiagnosisConfig, diagnose_stack)
from .engine import FitConfig, fit_pixels
from .fonts import configure_fonts
from .io_stack import (export_fit_npz, load_stack, register_stack)
from .masks import ALL_STATES, STATE_OK
from .synthetic import demo_truth, make_scene
from . import faults as fault_lib


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


def _build_demo_scene(args):
    A, B, C = demo_truth(args.height, args.width)
    scene = make_scene(
        A, B, C, args.angles,
        saturation=args.saturation, noise_sigma=args.noise,
        rng_seed=args.seed,
        occluder_cells=[(slice(6, 12), slice(8, 16), [0, 1])],
        motion_cells=[(slice(args.height // 2 - 3, args.height // 2 + 3),
                       slice(args.width // 2 - 4, args.width // 2 + 4),
                       [3])])
    return scene


def _print_diagnosis(report) -> None:
    print("== 异常诊断假设统计 ==")
    for hyp, cnt in sorted(report.counts.items(), key=lambda kv: -kv[1]):
        print(f"  {DIAGNOSIS_LABELS[hyp]:32s}: {cnt}")
    print("  逐帧稳健增益:",
          [round(float(g), 4) for g in report.global_gains])
    print("  增益 MAD     :",
          [round(float(m), 4) for m in report.global_gain_mad])
    print("  正常观测分量等价核查:", report.component_equivalence)
    abnormal = [r for r in report.records
                if r.hypothesis != "consistent_with_model"
                and r.hypothesis != "not_estimable"][:3]
    for rec in abnormal:
        print(f"  示例像素 ({rec.row},{rec.col}) -> "
              f"{DIAGNOSIS_LABELS[rec.hypothesis]}")
        for ev in rec.evidence[:2]:
            print(f"      证据: {ev}")
        print(f"      建议: {rec.suggestion}")


def cmd_fault_demo(args) -> int:
    """四类受控故障对照：只有角度、只有曝光、联合、局部运动。"""
    scene = _build_demo_scene(args)
    cfg = FitConfig(saturation=scene.saturation,
                    noise_sigma=scene.noise_sigma,
                    weak_snr=args.weak_snr)
    dcfg = DiagnosisConfig()
    base_result = fit_pixels(
        scene.intensity, scene.angles_rad, cfg,
        motion_mask=scene.motion_mask, missing_mask=scene.missing_mask)

    perm = tuple(args.permutation)
    gain_tuple = tuple(args.gains)
    scenarios = {
        "clean": (scene.intensity, scene.motion_mask, scene.missing_mask,
                  "consistent_with_model"),
        "angle_only": (
            fault_lib.inject_angle_permutation(
                scene.intensity, scene.angles_rad, permutation=perm).intensity,
            None, None, "angle_order_mismatch"),
        "exposure_only": (
            fault_lib.inject_exposure_gains(
                scene.intensity, scene.angles_rad, gains=gain_tuple).intensity,
            None, None, "exposure_inconsistency"),
        "angle_and_exposure": (
            fault_lib.combine(
                fault_lib.inject_angle_permutation(
                    scene.intensity, scene.angles_rad, permutation=perm),
                fault_lib.inject_exposure_gains(
                    scene.intensity, scene.angles_rad, gains=gain_tuple)
            ).intensity, None, None, "angle_and_exposure"),
        "motion_only": (
            fault_lib.inject_local_motion(
                scene.intensity, scene.angles_rad,
                slice(args.height // 2 - 3, args.height // 2 + 3),
                slice(args.width // 2 - 4, args.width // 2 + 4),
                3, shift=args.motion_shift).intensity,
            None, None, "local_motion"),
    }

    summary = {}
    all_ok = True
    bundles = []
    total_pixels = args.height * args.width
    for name, (stack, mm, km, expected) in scenarios.items():
        result = fit_pixels(stack, scene.angles_rad, cfg,
                            motion_mask=mm, missing_mask=km)
        report = diagnose_stack(stack, scene.angles_rad, result, config=dcfg)
        dominant = max(report.counts, key=report.counts.get)
        if name == "clean":
            # 干净对照：绝大多数像素必须与模型一致，且不允许任何故障假设成势
            ok = (report.counts.get("consistent_with_model", 0)
                  >= 0.8 * sum(report.counts.values())
                  and all(report.counts.get(h, 0) < 0.05 * total_pixels
                          for h in ("angle_order_mismatch",
                                    "exposure_inconsistency",
                                    "angle_and_exposure", "local_motion")))
        elif name == "motion_only":
            # 运动是局部故障：要求在注入区域上检出运动块，而非主导全图
            r0 = args.height // 2 - 3
            r1 = args.height // 2 + 3
            c0 = args.width // 2 - 4
            c1 = args.width // 2 + 4
            patch = report.hypothesis_map[r0:r1, c0:c1]
            hit = int((patch == "local_motion").sum())
            ok = hit >= dcfg.motion_min_pixels
            if ok:
                dominant = f"local_motion (注入区命中 {hit} 像素)"
        elif name == "angle_and_exposure":
            # 联合对照：两类假设都必须成势（任一单独解释都不足以主导）
            n_tot = sum(report.counts.values())
            joint_like = report.counts.get("angle_and_exposure", 0)
            angle_like = report.counts.get("angle_order_mismatch", 0)
            ok = ((joint_like + angle_like) >= 0.30 * n_tot
                  and report.counts.get("exposure_inconsistency", 0)
                  >= 0.20 * n_tot)
        else:
            # 单因帧级故障（角度/曝光）应成为主导假设
            ok = dominant == expected
        all_ok &= ok
        summary[name] = {"expected": expected, "dominant": dominant,
                         "matched": bool(ok), "counts": report.counts}
        print(f"[{name}] 期望={expected} 主导={dominant} "
              f"{'通过' if ok else '不通过'}")
        _print_diagnosis(report)
        if args.bundle_dir:
            path = extract_bundle(
                report, result, stack, args.bundle_dir, name=f"fault_{name}",
                extra_note=f"受控故障对照: {name}，期望假设 {expected}")
            rebuilt, _, _ = rebuild_bundle(path)
            print(f"  诊断包: {path}（离线复算 {'一致' if rebuilt else '不一致'}）")
            bundles.append(path)
        print()

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"scenarios": summary,
                       "config": {"fit": vars(cfg) if hasattr(cfg, '__dict__')
                                  else {},
                                  "diagnosis": dcfg.__dict__}},
                      fh, ensure_ascii=False, indent=2, default=str)
        print(f"对照报告已写入 {args.report}")
    print("总判定:", "全部对照与注入故障一致" if all_ok
          else "存在对照未通过，请检查阈值或故障强度")
    return 0 if all_ok else 1


def cmd_diagnose(args) -> int:
    """对清单图像运行诊断并提取小型诊断包（不修改任何输入）。"""
    stack = load_stack(args.manifest, read_noise=args.read_noise)
    intensity = stack.intensity
    motion_mask = missing_mask = None
    if not stack.is_color and args.register:
        intensity, _, missing_mask = register_stack(intensity)
    if np.ndim(stack.saturation) > 0:
        saturation = float(np.min(stack.saturation))
    else:
        saturation = float(stack.saturation)
    noise = (float(np.mean(stack.noise_sigma))
             if np.ndim(stack.noise_sigma) else float(stack.noise_sigma))

    cfg = FitConfig(saturation=saturation, noise_sigma=noise,
                    weak_snr=args.weak_snr)
    result = fit_pixels(intensity, stack.angles_rad, cfg,
                        motion_mask=motion_mask, missing_mask=missing_mask)
    dcfg = DiagnosisConfig()
    region = None
    if args.region:
        r0, r1, c0, c1 = [int(x) for x in args.region.split(",")]
        region = (r0, r1, c0, c1)
    report = diagnose_stack(intensity, stack.angles_rad, result,
                            config=dcfg, region=region)
    _print_diagnosis(report)

    if args.bundle_dir:
        path = extract_bundle(
            report, result, intensity, args.bundle_dir,
            name=args.bundle_name,
            extra_note=f"来源清单 {os.path.basename(args.manifest)}")
        rebuilt, _, sealed = rebuild_bundle(path)
        print(f"诊断包: {path}")
        print(f"离线复算: {'重建同一结论' if rebuilt else '复算不一致'}")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(sealed, fh, ensure_ascii=False, indent=2)
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

    p_fd = sub.add_parser(
        "fault-demo",
        help="生成角度/曝光/联合/运动受控对照并核对诊断结论")
    p_fd.add_argument("--angles", type=float, nargs="+",
                      default=[0, 30, 60, 90, 120, 150])
    p_fd.add_argument("--height", type=int, default=64)
    p_fd.add_argument("--width", type=int, default=80)
    p_fd.add_argument("--noise", type=float, default=8.0)
    p_fd.add_argument("--saturation", type=float, default=4095.0)
    p_fd.add_argument("--weak-snr", type=float, default=3.0)
    p_fd.add_argument("--seed", type=int, default=20261113)
    p_fd.add_argument("--permutation", type=int, nargs="+",
                      default=[1, 0, 2, 3, 4, 5],
                      help="角度错配排列，长度与角度数一致")
    p_fd.add_argument("--gains", type=float, nargs="+",
                      default=[1.0, 1.0, 1.0, 1.6, 1.0, 1.0],
                      help="逐帧曝光增益，含一个显著偏离 1 的帧")
    p_fd.add_argument("--motion-shift", type=float, default=150.0)
    p_fd.add_argument("--bundle-dir", type=str, default=None,
                      help="若给出，提取四类诊断包到该目录并离线复算")
    p_fd.add_argument("--report", type=str, default=None)
    p_fd.set_defaults(func=cmd_fault_demo)

    p_diag = sub.add_parser("diagnose",
                            help="对清单图像运行异常诊断并提取诊断包")
    p_diag.add_argument("manifest")
    p_diag.add_argument("--read-noise", type=float, default=1.0)
    p_diag.add_argument("--weak-snr", type=float, default=3.0)
    p_diag.add_argument("--register", action="store_true")
    p_diag.add_argument("--region", type=str, default=None,
                        help="r0,r1,c0,c1 诊断窗口，缺省整幅")
    p_diag.add_argument("--bundle-dir", type=str, default=None)
    p_diag.add_argument("--bundle-name", type=str,
                        default="diagnostic_bundle")
    p_diag.add_argument("--report", type=str, default=None)
    p_diag.set_defaults(func=cmd_diagnose)
    return parser


def main(argv=None) -> int:
    configure_fonts()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
