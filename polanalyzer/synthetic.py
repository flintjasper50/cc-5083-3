"""可控角度响应的合成观测生成器。

用途：在已知真值参数 (A, B, C) 的条件下生成 K 个分析角的观测，
加入已知噪声、饱和钳位、局部遮挡（缺失）与运动区域，
供测试核对估计参数与预测强度，而不是用真实拟合的结果自证。

所有强度与噪声同量纲（传感器线性灰度，例如 12 bit 的 0..4095）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import predict


@dataclass
class SyntheticScene:
    """合成场景及其真值与污染标记。"""

    intensity: np.ndarray          # (H, W, K) 或 (H, W, C, K)，含噪声后的观测
    clean: np.ndarray              # 无噪声的理想强度
    truth_A: np.ndarray
    truth_B: np.ndarray
    truth_C: np.ndarray
    angles_rad: np.ndarray
    saturation: float
    noise_sigma: float
    missing_mask: np.ndarray       # True = 该角度该像素缺失/遮挡
    motion_mask: np.ndarray        # True = 该角度该像素注入运动错位
    rng_seed: int


def make_scene(truth_A: np.ndarray,
               truth_B: np.ndarray,
               truth_C: np.ndarray,
               angles_deg: np.ndarray,
               saturation: float = 4095.0,
               noise_sigma: float = 8.0,
               occluder_cells: list[tuple] | None = None,
               motion_cells: list[tuple] | None = None,
               motion_shift: float = 60.0,
               rng_seed: int = 20261113) -> SyntheticScene:
    """生成合成场景。

    truth_A/B/C : (H, W) 或 (H, W, C) 的真值分量图。
    angles_deg : (K,) 分析角，度数。
    occluder_cells : [(行切片, 列切片, 角度索引列表), ...]，
                     这些位置/角度被置为缺失（模拟局部遮挡）。
    motion_cells : 同上结构，这些样本叠加偏移以模拟运动，
                   并在 motion_mask 中标记（拟合时必须排除）。
    """
    angles_rad = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    A = np.asarray(truth_A, dtype=np.float64)
    B = np.asarray(truth_B, dtype=np.float64)
    C = np.asarray(truth_C, dtype=np.float64)

    clean = predict(angles_rad, A, B, C)  # (..., K)
    rng = np.random.default_rng(rng_seed)
    noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)

    missing = np.zeros(clean.shape, dtype=bool)
    motion = np.zeros(clean.shape, dtype=bool)

    for row_slice, col_slice, angle_idx in (occluder_cells or []):
        idx = list(angle_idx)
        if clean.ndim == 4:
            missing[row_slice, col_slice, :, idx] = True
        else:
            missing[row_slice, col_slice, idx] = True
        noisy[missing] = np.nan

    for row_slice, col_slice, angle_idx in (motion_cells or []):
        idx = list(angle_idx)
        if clean.ndim == 4:
            motion[row_slice, col_slice, :, idx] = True
            noisy[row_slice, col_slice, :, idx] += motion_shift
        else:
            motion[row_slice, col_slice, idx] = True
            noisy[row_slice, col_slice, idx] += motion_shift

    # 饱和钳位：记录钳位后的观测，饱和由 masks 依据 saturation 判定
    observed = np.where(np.isfinite(noisy),
                        np.clip(noisy, 0.0, saturation), np.nan)

    return SyntheticScene(
        intensity=observed, clean=clean,
        truth_A=A, truth_B=B, truth_C=C,
        angles_rad=angles_rad, saturation=saturation,
        noise_sigma=noise_sigma, missing_mask=missing,
        motion_mask=motion, rng_seed=rng_seed)


def demo_truth(h: int = 96, w: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造一张带结构的灰度真值图（恒定 + 偏振方向随位置变化）。

    设计上包含三类区域：
    - 强偏振区（R 显著）；
    - 弱偏振区（R 接近 0，应被判为 weak_response）；
    - 高反射区（A 较大且部分角度饱和）。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    fx = (xx - w / 2) / w
    fy = (yy - h / 2) / h

    A = 900.0 + 500.0 * np.exp(-(fx ** 2 + fy ** 2) * 4.0)
    # 左下角高反射斑，抬升 A 并使部分角度达到饱和钳位
    A += 3600.0 * np.exp(-(((xx - 0.18 * w) / (0.12 * w)) ** 2
                           + ((yy - 0.82 * h) / (0.12 * h)) ** 2))

    # 偏振方向角随空间旋转，幅度在弱响应带内压到接近零
    phi_true = (np.pi / 3) * (fx + 0.5)
    R = 220.0 * (0.5 + 0.5 * np.cos(2 * np.pi * fy))
    weak_band = (np.abs(fy) < 0.08)
    R = np.where(weak_band, 6.0, R)  # 弱响应：R 与噪声同量级

    B = R * np.cos(2 * phi_true)
    C = R * np.sin(2 * phi_true)
    return A, B, C
