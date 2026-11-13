"""拟合结果的可视化构造器（与界面框架无关）。

所有图像均由 FitResult 中的实际数组计算，不做示意性伪造：
- 分量图：A、R、phi（HSV 编码）、DoLP；
- 可靠性图：状态分类着色 + 残差 RMS / 归一化卡方；
- 单像素观测曲线：有效样本、被排除样本（标注原因）、拟合曲线与不确定带。
"""

from __future__ import annotations

import numpy as np

from .masks import (STATE_ALL_INVALID, STATE_FEW_ANGLES, STATE_OK,
                    STATE_PHYSICAL, STATE_WEAK, exclusion_reasons)
from .model import predict

STATE_COLORS = {
    STATE_OK: (0.20, 0.65, 0.30),
    STATE_WEAK: (0.95, 0.75, 0.15),
    STATE_FEW_ANGLES: (0.95, 0.55, 0.10),
    STATE_PHYSICAL: (0.85, 0.25, 0.25),
    STATE_ALL_INVALID: (0.55, 0.55, 0.55),
}

STATE_LABELS = {
    STATE_OK: "可靠 (ok)",
    STATE_WEAK: "弱响应：偏振角/偏振度不可解释",
    STATE_FEW_ANGLES: "有效角度不足 3 个",
    STATE_PHYSICAL: "拟合违反物理约束 (A<=0 或 R>A)",
    STATE_ALL_INVALID: "无有效样本",
}


def component_arrays(result) -> dict[str, np.ndarray]:
    """返回供界面显示的分量数组（保持原始量纲，不做归一化承诺）。"""
    return {
        "A_恒定分量": result.A,
        "R_调变幅度": result.R,
        "phi_偏振角_rad": result.phi_rad,
        "DoLP_线偏振度": result.dolp,
        "残差RMS": result.residual_rms,
        "归一化卡方": result.reduced_chi2,
        "R不确定度": result.sigma_R,
        "phi不确定度_rad": result.sigma_phi_rad,
    }


def angle_hsv_image(phi_rad: np.ndarray, R: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
    """把偏振角编码为 HSV 风格 RGB 图。

    色相 = phi（角度本身），明度按 R 缩放；
    非可靠像素消色，避免在弱响应区给出虚假的颜色方向感。
    """
    import matplotlib
    cmap = matplotlib.colormaps["hsv"]
    reliable = state == STATE_OK
    hue = np.where(reliable, (phi_rad % (np.pi / 2)) / (np.pi / 2), 0.0)
    rgb = cmap(hue)[..., :3]
    if reliable.any():
        scale = np.clip(R / (np.nanmedian(R[reliable]) * 1.5), 0, 1)
    else:
        scale = np.zeros_like(R)
    rgb = rgb * scale[..., None]
    rgb[~reliable] = 0.15
    return np.clip(rgb, 0, 1)


def state_overlay(state: np.ndarray) -> np.ndarray:
    """状态分类着色图 (H, W, 3)。"""
    out = np.zeros((*state.shape, 3))
    for st, color in STATE_COLORS.items():
        out[state == st] = color
    return out


def pixel_curve_data(result, row: int, col: int,
                     channel: int | None = None) -> dict:
    """提取单像素观测曲线所需的全部实际数据。

    返回字典包含：角度、观测值、预测值、有效/排除标记与原因、
    拟合参数与不确定度、状态。被排除样本单独列出，供界面明示。
    """
    angles_deg = np.rad2deg(result.angles_rad)

    def pick(arr):
        if channel is not None and arr.ndim == 3:
            return arr[row, col, channel]
        return arr[row, col]

    observed = pick(result.intensity)
    predicted = pick(result.predicted)
    valid = pick(result.valid)

    sub = observed
    motion_1d = (pick(result.motion_mask)
                 if result.motion_mask is not None else None)
    missing_1d = (pick(result.missing_mask)
                  if result.missing_mask is not None else None)
    reasons = exclusion_reasons(
        sub[None, :], result.config.saturation,
        None if motion_1d is None else motion_1d[None, :],
        None if missing_1d is None else missing_1d[None, :])[0]

    dense = np.linspace(result.angles_rad.min(), result.angles_rad.max(), 200)
    fit_curve = predict(dense, pick(result.A), pick(result.B), pick(result.C))
    sigma_R = pick(result.sigma_R)
    # R 的不确定带：保持 phi 不变，沿调变方向整体缩放 R
    R_val = pick(result.R)
    phi_val = pick(result.phi_rad)
    if np.isfinite(R_val) and R_val > 0:
        r_hi = max(R_val + np.nan_to_num(sigma_R), 0.0)
        r_lo = max(R_val - np.nan_to_num(sigma_R), 0.0)
        b_hi, c_hi = r_hi * np.cos(2 * phi_val), r_hi * np.sin(2 * phi_val)
        b_lo, c_lo = r_lo * np.cos(2 * phi_val), r_lo * np.sin(2 * phi_val)
        upper = predict(dense, pick(result.A), b_hi, c_hi)
        lower = predict(dense, pick(result.A), b_lo, c_lo)
    else:
        upper = lower = fit_curve

    return {
        "row": row, "col": col, "channel": channel,
        "angles_deg": angles_deg,
        "observed": np.asarray(observed, dtype=np.float64),
        "predicted": np.asarray(predicted, dtype=np.float64),
        "valid": np.asarray(valid, dtype=bool),
        "exclusion_reasons": np.asarray(reasons, dtype=object),
        "fit_angles_deg": np.rad2deg(dense),
        "fit_curve": fit_curve,
        "band_upper": np.maximum(upper, lower),
        "band_lower": np.minimum(upper, lower),
        "A": pick(result.A), "B": pick(result.B), "C": pick(result.C),
        "R": pick(result.R), "phi_deg": np.rad2deg(pick(result.phi_rad)),
        "dolp": pick(result.dolp),
        "sigma_A": pick(result.sigma_A),
        "sigma_R": sigma_R,
        "sigma_phi_deg": np.rad2deg(pick(result.sigma_phi_rad)),
        "sigma_dolp": pick(result.sigma_dolp),
        "residual_rms": pick(result.residual_rms),
        "reduced_chi2": pick(result.reduced_chi2),
        "n_valid": int(pick(result.n_valid)),
        "state": str(pick(result.state)),
    }
