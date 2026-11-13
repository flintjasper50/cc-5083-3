"""逐像素加权最小二乘拟合引擎与可靠性判定。

拟合模型：I(theta) = A + B cos2theta + C sin2theta。

对每个像素（及每个颜色通道）独立处理：
1. 仅使用 masks 模块判定为有效的角度样本，被排除的样本不进入方程；
2. 有效方程数 < 3 时模型不可辨识，输出 insufficient_angles；
3. 加权最小二乘，权重为 1/sigma^2（sigma 为每样本噪声标准差，
   缺省取全局估计值）；
4. 由参数协方差传播得到 A、R、phi、DoLP 的不确定度；
5. 物理检查：恒定分量 A 必须为正，偏振幅度 R 不能超过 A
   （I 的极小值 A-R 不可为负）。违反时标记 unphysical_fit；
6. 弱响应区：R 与其不确定度之比低于阈值时，phi/DoLP 不可解释，
   标记 weak_response——给出可解释的"不确定"状态而非强行赋角度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .masks import (STATE_ALL_INVALID, STATE_FEW_ANGLES, STATE_OK,
                    STATE_PHYSICAL, STATE_WEAK, valid_sample_mask)
from .model import derive_components, design_matrix

MIN_EQUATIONS = 3


def solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """单像素加权最小二乘求解，引擎与诊断模块共用同一份数学。

    返回 beta、协方差、条件数、秩、残差与加权残差平方和。
    条件数基于加权设计矩阵 XtWX 的特征值，诊断记录据此判断求解是否病态。
    """
    XtW = X.T * w
    gram = XtW @ X
    try:
        cov = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        return {"ok": False}
    eig = np.linalg.eigvalsh(gram)
    eig_pos = eig[eig > 0.0]
    cond = (float(eig_pos.max() / eig_pos.min())
            if eig_pos.size == gram.shape[0] else np.inf)
    beta = cov @ (XtW @ y)
    resid = y - X @ beta
    return {
        "ok": True, "beta": beta, "cov": cov,
        "condition": cond, "rank": int(np.linalg.matrix_rank(X)),
        "residual": resid, "w_rss": float(np.sum(w * resid ** 2))}


@dataclass
class FitConfig:
    """拟合配置。

    saturation : 传感器饱和值，达到该值的样本被排除。
    noise_sigma : 强度域噪声标准差（与强度同量纲），用于权重与显著性。
                  可为标量或与强度同形状的数组。
    weak_snr : 弱响应判定阈值，R / sigma_R 低于该值即 weak_response。
    max_residual_sigma : 残差超过该倍数噪声时计入残差异常计数。
    """

    saturation: float
    noise_sigma: float | np.ndarray = 1.0
    weak_snr: float = 3.0
    max_residual_sigma: float = 4.0


@dataclass
class FitResult:
    """拟合输出，所有图像数组形状为 (H, W) 或 (H, W, C)。"""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    R: np.ndarray
    phi_rad: np.ndarray
    dolp: np.ndarray
    sigma_A: np.ndarray
    sigma_R: np.ndarray
    sigma_phi_rad: np.ndarray
    sigma_dolp: np.ndarray
    residual_rms: np.ndarray
    weighted_rss: np.ndarray
    reduced_chi2: np.ndarray
    n_valid: np.ndarray
    max_abs_residual: np.ndarray
    n_residual_outliers: np.ndarray
    condition_number: np.ndarray     # 加权法矩阵 XtWX 的条件数，病态求解的证据
    design_rank: np.ndarray          # 有效样本上设计矩阵的秩
    state: np.ndarray           # 字符串数组，取值见 masks.ALL_STATES
    valid: np.ndarray           # 布尔，形状 (..., K) 的有效样本遮罩
    motion_mask: np.ndarray | None
    missing_mask: np.ndarray | None
    angles_rad: np.ndarray
    intensity: np.ndarray
    predicted: np.ndarray
    config: FitConfig

    @property
    def reliable(self) -> np.ndarray:
        """仅 ok 状态视为参数（尤其 phi/DoLP）可靠可解释。"""
        return self.state == STATE_OK


def _flatten_spatial(intensity: np.ndarray) -> tuple[np.ndarray, tuple, bool]:
    """把 (H,W,K) 或 (H,W,C,K) 规整为 (N, K)，返回逆展开信息。"""
    if intensity.ndim == 2:
        return intensity, (intensity.shape[0],), False
    if intensity.ndim == 3:
        h, w, k = intensity.shape
        return intensity.reshape(-1, k), (h, w), False
    if intensity.ndim == 4:
        h, w, c, k = intensity.shape
        moved = np.moveaxis(intensity, 2, 1)  # (H, C, W, K)
        flat = moved.reshape(-1, k)
        return flat, (h, w, c), True
    raise ValueError(f"不支持的强度维度: {intensity.shape}")

def fit_pixels(intensity: np.ndarray,
               angles_rad: np.ndarray,
               config: FitConfig,
               motion_mask: np.ndarray | None = None,
               missing_mask: np.ndarray | None = None) -> FitResult:
    """对整幅观测栈逐像素拟合。

    参数
    ----
    intensity : (H, W, K) 或 (H, W, C, K)，K 个分析角的配准后强度。
    angles_rad : (K,) 分析角（弧度）。所有像素共用同一角度序列。
    config : FitConfig。
    motion_mask / missing_mask : 与 intensity 同形状的布尔遮罩。

    调用方需保证各帧已配准、曝光已统一解释（见 io_stack 模块）。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    k = angles_rad.size
    if intensity.shape[-1] != k:
        raise ValueError("角度数与强度最后一维不一致")

    flat, spatial, multichannel = _flatten_spatial(intensity)
    n = flat.shape[0]
    mm = (np.broadcast_to(motion_mask, intensity.shape)
          if motion_mask is not None else None)
    ksm = (np.broadcast_to(missing_mask, intensity.shape)
           if missing_mask is not None else None)
    valid_flat, _, _ = _flatten_spatial(
        valid_sample_mask(intensity, config.saturation, mm, ksm))

    sigma = config.noise_sigma
    if np.isscalar(sigma):
        sigma_flat = np.full(flat.shape, float(sigma))
    else:
        sigma_flat, _, _ = _flatten_spatial(np.broadcast_to(sigma, intensity.shape))

    X_full = design_matrix(angles_rad)

    A = np.full(n, np.nan)
    B = np.full(n, np.nan)
    C = np.full(n, np.nan)
    sA = np.full(n, np.nan)
    sR = np.full(n, np.nan)
    sPhi = np.full(n, np.nan)
    sDo = np.full(n, np.nan)
    rms = np.full(n, np.nan)
    wrss = np.full(n, np.nan)
    chi2 = np.full(n, np.nan)
    nval = np.zeros(n, dtype=np.int32)
    maxres = np.full(n, np.nan)
    nout = np.zeros(n, dtype=np.int32)
    conds = np.full(n, np.nan)
    ranks = np.zeros(n, dtype=np.int32)
    states = np.full(n, STATE_ALL_INVALID, dtype=object)
    pred_flat = np.full((n, k), np.nan)

    for i in range(n):
        v = valid_flat[i]
        n_valid = int(v.sum())
        nval[i] = n_valid
        if n_valid == 0:
            continue
        if n_valid < MIN_EQUATIONS:
            states[i] = STATE_FEW_ANGLES
            continue

        X = X_full[v]
        y = flat[i, v]
        w = 1.0 / np.maximum(sigma_flat[i, v] ** 2, 1e-12)
        sol = solve_wls(X, y, w)
        if not sol["ok"]:
            states[i] = STATE_FEW_ANGLES
            nval[i] = 0
            continue
        cov = sol["cov"]
        beta = sol["beta"]
        resid = sol["residual"]
        conds[i] = sol["condition"]
        ranks[i] = sol["rank"]
        dof = max(n_valid - 3, 1)
        w_rss = sol["w_rss"]
        wrss[i] = w_rss
        chi2[i] = w_rss / dof
        rms[i] = float(np.sqrt(np.mean(resid ** 2)))
        maxres[i] = float(np.max(np.abs(resid)))
        nout[i] = int(np.sum(np.abs(resid)
                             > config.max_residual_sigma * sigma_flat[i, v]))

        a, b, c = beta
        A[i], B[i], C[i] = a, b, c
        var = np.diag(cov)
        sA[i] = float(np.sqrt(max(var[0], 0.0)))
        R = float(np.hypot(b, c))
        if R > 1e-12:
            var_R = (b ** 2 * var[1] + c ** 2 * var[2]
                     + 2.0 * b * c * cov[1, 2]) / R ** 2
            var_phi = (c ** 2 * var[1] + b ** 2 * var[2]
                       - 2.0 * b * c * cov[1, 2]) / (4.0 * R ** 4)
        else:
            var_R = 0.5 * (var[1] + var[2])
            var_phi = np.inf
        sR[i] = float(np.sqrt(max(var_R, 0.0)))
        sPhi[i] = float(np.sqrt(max(var_phi, 0.0))) if np.isfinite(var_phi) else np.nan

        pred_flat[i] = X_full @ beta

        if a <= 0.0 or R > a + 1e-9:
            states[i] = STATE_PHYSICAL
            continue
        if R < config.weak_snr * sR[i]:
            states[i] = STATE_WEAK
        else:
            states[i] = STATE_OK

    R, phi = derive_components(B, C)
    with np.errstate(invalid="ignore", divide="ignore"):
        dolp = np.where(A > 0, R / A, np.nan)
        var_dolp = np.where(
            A > 0,
            (sR ** 2 + dolp ** 2 * sA ** 2) / np.maximum(A ** 2, 1e-12),
            np.nan)
        s_dolp = np.sqrt(np.maximum(var_dolp, 0.0))

    def unfold(arr: np.ndarray) -> np.ndarray:
        if multichannel:
            h, w, c = spatial
            reshaped = arr.reshape(h, c, w, *arr.shape[1:])
            return np.moveaxis(reshaped, 1, 2)
        return arr.reshape(spatial + arr.shape[1:])

    valid_out = valid_flat.reshape(intensity.shape)

    return FitResult(
        A=unfold(A), B=unfold(B), C=unfold(C),
        R=unfold(R), phi_rad=unfold(phi),
        dolp=unfold(dolp),
        sigma_A=unfold(sA), sigma_R=unfold(sR),
        sigma_phi_rad=unfold(sPhi), sigma_dolp=unfold(s_dolp),
        residual_rms=unfold(rms), weighted_rss=unfold(wrss),
        reduced_chi2=unfold(chi2),
        n_valid=unfold(nval.astype(np.float64)),
        max_abs_residual=unfold(maxres),
        n_residual_outliers=unfold(nout.astype(np.float64)),
        condition_number=unfold(conds),
        design_rank=unfold(ranks.astype(np.float64)),
        state=unfold(states),
        valid=valid_out,
        motion_mask=(mm if mm is None else np.broadcast_to(mm, intensity.shape)),
        missing_mask=(ksm if ksm is None else np.broadcast_to(ksm, intensity.shape)),
        angles_rad=angles_rad, intensity=intensity,
        predicted=unfold(pred_flat) if multichannel
        else pred_flat.reshape(*spatial, k),
        config=config)
