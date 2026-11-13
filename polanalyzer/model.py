"""多角度线性偏振分析的成像模型。

声明的成像模型（每像素、每通道独立）：

    I(theta) = A + B * cos(2*theta) + C * sin(2*theta)
             = A + R * cos(2*(theta - phi))

其中 theta 为检偏器（分析角）的透射轴方向。

参数含义与边界：
- A：角度恒定分量。它包含未偏振（或圆偏振等在旋转线偏振片下不可分辨的）
  贡献，**不等于**"完全无反光的真实物体颜色"。镜面反射中的未偏振部分同样
  落入 A，本模型无法将其与漫反射区分。
- R = sqrt(B**2 + C**2)：随角度变化分量的幅度（线偏振度的原始量）。
- phi = 0.5 * atan2(C, B)：偏振角（AoLP），仅在 R 显著高于噪声时可解释。
- DoLP = R / A：线偏振度（A 必须为正且 R 显著）。

本模块只描述模型，不做任何 I/O。
"""

from __future__ import annotations

import numpy as np

MODEL_FORMULA = "I(theta) = A + B*cos(2*theta) + C*sin(2*theta)"


def design_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """返回形状 (K, 3) 的设计矩阵，列对应 [1, cos2theta, sin2theta]。"""
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    twice = 2.0 * angles_rad
    return np.column_stack([np.ones_like(angles_rad), np.cos(twice), np.sin(twice)])


def predict(angles_rad: np.ndarray, A: np.ndarray, B: np.ndarray,
            C: np.ndarray) -> np.ndarray:
    """按模型预测强度。A/B/C 可广播，返回形状 (..., K)。"""
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    twice = 2.0 * angles_rad
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return (A[..., None] + B[..., None] * np.cos(twice)
            + C[..., None] * np.sin(twice))


def derive_components(B: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由 (B, C) 计算偏振幅度 R 与偏振角 phi（弧度，范围 [-pi/2, pi/2)）。"""
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    R = np.hypot(B, C)
    with np.errstate(invalid="ignore"):
        phi = 0.5 * np.arctan2(C, B)
    phi = np.where(R > 0, phi, np.nan)
    return R, phi
