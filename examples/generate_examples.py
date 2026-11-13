"""生成写入仓库的示例数据：受控故障帧图像、清单与诊断包。

运行：

    python3 examples/generate_examples.py

产物（全部在 examples/ 下，可用 git 跟踪，体积受控）：

    clean/                      正常对照：6 帧 16-bit PNG + manifest.json
    angle_only/                 仅分析角顺序错配（帧文件名角度与真实来源错位）
    exposure_only/              仅整帧曝光增益不一致（第 4 帧增益 1.6）
    angle_and_exposure/         两类故障共同存在
    bundles/                    各场景提取的小型诊断包（manifest/data/rebuild）
    clean_reference.npz         干净场景的无噪声真值与含噪观测，供离线核对

设计意图：
- 示例是"真实文件"（PNG + JSON 清单），可直接走 pol-cli diagnose 的完整
  载入链路，而不是只在进程内传数组；
- 故障参数与 tests/test_diagnostics.py、GUI 受控场景完全一致，保证文档、
  测试与界面三处结论可互相复现；
- 角度错配只体现在清单声明与帧内容的对应关系上，强度数值不被修改；
  曝光变化只体现为整帧比例。任何残差都不能归咎于"噪声变大"。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polanalyzer import faults
from polanalyzer.bundle import extract_bundle, rebuild_bundle
from polanalyzer.diagnostics import DiagnosisConfig, diagnose_stack
from polanalyzer.engine import FitConfig, fit_pixels
from polanalyzer.synthetic import demo_truth, make_scene

ANGLES_DEG = [0, 30, 60, 90, 120, 150]
PERM = (1, 0, 2, 3, 4, 5)
GAINS = (1.0, 1.0, 1.0, 1.6, 1.0, 1.0)
HEIGHT, WIDTH = 48, 56
NOISE_SIGMA = 8.0
SATURATION = 4095.0
SEED = 20261113


def _write_frames(intensity: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for j, deg in enumerate(ANGLES_DEG):
        frame = intensity[..., j]
        # 合成域在 0..4095，存入 16-bit PNG 保留线性灰度；NaN（遮挡/缺失）
        # 转 0 并在清单外由诊断包 data.npz 保留完整浮点观测。
        frame = np.where(np.isfinite(frame), frame, 0.0)
        frame = np.clip(np.rint(frame), 0, SATURATION).astype(np.uint16)
        Image.fromarray(frame).save(
            os.path.join(out_dir, f"frame_{deg:03d}deg.png"))


def _write_manifest(out_dir: str, note: str) -> str:
    # 清单始终声明 gain=1.0：示例要模拟的是"未声明、未被共同曝光解释捕获"
    # 的曝光偏差。若把真实增益写进清单，load_stack 会直接归一，故障在进入
    # 诊断前就消失，也就无从演示诊断能力。真实增益只记录在 note 中作为真值。
    frames = []
    for deg in ANGLES_DEG:
        frames.append({
            "path": f"frame_{deg:03d}deg.png",
            "angle_deg": deg,
            "exposure_s": 1.0,
            "gain": 1.0,
            "saturation": SATURATION,
        })
    manifest = {"frames": frames, "note": note}
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return path


def main() -> int:
    rng_seed = SEED
    A, B, C = demo_truth(HEIGHT, WIDTH)
    scene = make_scene(
        A, B, C, ANGLES_DEG,
        saturation=SATURATION, noise_sigma=NOISE_SIGMA, rng_seed=42,
        occluder_cells=[(slice(6, 10), slice(8, 12), [0, 1])],
        motion_cells=[(slice(24, 30), slice(30, 38), [3])])
    cfg = FitConfig(saturation=scene.saturation, noise_sigma=scene.noise_sigma)
    dcfg = DiagnosisConfig()

    # 干净对照：真值与含噪观测都存盘，供离线核对，不用拟合结果自证
    np.savez_compressed(
        os.path.join(_HERE, "clean_reference.npz"),
        truth_A=scene.truth_A, truth_B=scene.truth_B, truth_C=scene.truth_C,
        clean_intensity=scene.clean, noisy_intensity=np.nan_to_num(
            scene.intensity, nan=0.0),
        angles_rad=scene.angles_rad, saturation=SATURATION,
        noise_sigma=NOISE_SIGMA)

    scenarios = {
        "clean": (scene.intensity,
                  "正常对照：仅含已知噪声、遮挡与已标记运动区"),
        "angle_only": (
            faults.inject_angle_permutation(
                scene.intensity, scene.angles_rad, permutation=PERM).intensity,
            f"仅分析角顺序错配：帧内容按排列 {PERM} 重排，"
            "清单仍按 0..150 度声明角度"),
        "exposure_only": (
            faults.inject_exposure_gains(
                scene.intensity, scene.angles_rad, gains=GAINS).intensity,
            f"仅曝光不一致：逐帧增益 {GAINS}（90° 帧增益 1.6，未在清单声明）"),
        "angle_and_exposure": (
            faults.combine(
                faults.inject_angle_permutation(
                    scene.intensity, scene.angles_rad, permutation=PERM),
                faults.inject_exposure_gains(
                    scene.intensity, scene.angles_rad, gains=GAINS)
            ).intensity,
            "角度错配与曝光变化共同存在：排列 "
            f"{PERM}，增益 {GAINS}（增益未在清单声明）"),
    }

    bundle_root = os.path.join(_HERE, "bundles")
    os.makedirs(bundle_root, exist_ok=True)
    print("== 示例数据生成 ==")
    for name, (stack, note) in scenarios.items():
        out_dir = os.path.join(_HERE, name)
        _write_frames(stack, out_dir)
        _write_manifest(out_dir, note)
        result = fit_pixels(stack, scene.angles_rad, cfg)
        report = diagnose_stack(stack, scene.angles_rad, result, config=dcfg)
        bundle_path = extract_bundle(
            report, result, stack, bundle_root, name=f"example_{name}",
            extra_note=note)
        ok, _, _ = rebuild_bundle(bundle_path)
        dominant = max(report.counts, key=report.counts.get)
        print(f"[{name}] 主导假设={dominant}  诊断包复算={'一致' if ok else '不一致'}")
        if not ok:
            return 1
    print("示例数据已写入", _HERE)
    print("离线复算示例：python3",
          os.path.join("examples", "bundles", "example_angle_only", "rebuild.py"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
