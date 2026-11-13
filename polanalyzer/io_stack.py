"""观测栈的载入、共同曝光解释与量纲保留的导出。

约定
----
- 角度图像通过 JSON 清单声明：每张图给出 path、angle_deg、exposure_s、
  gain 以及可选的饱和值；
- "共同曝光解释"：若各帧曝光时间/增益不同，按线性传感器模型归一到
  单位曝光的强度域 I_norm = I / (exposure_s * gain)，并把噪声与饱和值
  一并换算。归一化只是线性比例，不改变 A、R 之间的物理关系；
- 配准：提供基于相位相关的整像素平移估计与平移校正。调用方也可在外部
  完成配准后直接传入数组；
- 导出使用 NPZ（NumPy 原生，保留浮点量纲与不确定度），不通过 8 bit
  截图丢失量纲。另附带刻度的 PNG 仅作可视化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class FrameSpec:
    path: str
    angle_deg: float
    exposure_s: float
    gain: float
    saturation: float | None = None


@dataclass
class ObservationStack:
    """载入并归一化后的观测栈。"""

    intensity: np.ndarray            # (H, W, K) 灰度或 (H, W, C, K) 彩色
    angles_rad: np.ndarray           # (K,)
    saturation: np.ndarray           # 标量或 (K,)，归一化后的饱和值
    noise_sigma: np.ndarray          # (K,) 或标量，归一化后的噪声估计
    frame_specs: list[FrameSpec]
    is_color: bool
    unit: str                        # 量纲说明字符串


def load_manifest(manifest_path: str) -> list[FrameSpec]:
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames = []
    for item in data["frames"]:
        frames.append(FrameSpec(
            path=item["path"],
            angle_deg=float(item["angle_deg"]),
            exposure_s=float(item.get("exposure_s", 1.0)),
            gain=float(item.get("gain", 1.0)),
            saturation=item.get("saturation")))
    return frames


def _read_image(path: str) -> tuple[np.ndarray, float]:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        sat = 255.0
    elif arr.dtype == np.uint16:
        sat = float(np.iinfo(np.uint16).max)
    else:
        sat = float(np.nanmax(arr))
    return arr.astype(np.float64), sat


def load_stack(manifest_path: str,
               read_noise: float = 1.0,
               normalize_exposure: bool = True) -> ObservationStack:
    """按清单载入角度图像并做共同曝光解释。"""
    specs = load_manifest(manifest_path)
    specs.sort(key=lambda f: f.angle_deg)

    arrays, sats, sigmas = [], [], []
    for spec in specs:
        arr, native_sat = _read_image(spec.path)
        sat = spec.saturation if spec.saturation is not None else native_sat
        scale = (spec.exposure_s * spec.gain) if normalize_exposure else 1.0
        arrays.append(arr / scale)
        sats.append(sat / scale)
        sigmas.append(read_noise / scale)

    first = arrays[0]
    is_color = first.ndim == 3
    if is_color:
        stack = np.stack([a for a in arrays], axis=-1)  # (H,W,C,K)
    else:
        stack = np.stack(arrays, axis=-1)              # (H,W,K)

    angles = np.deg2rad(np.array([f.angle_deg for f in specs], dtype=np.float64))
    unit = ("intensity / (exposure_s * gain)" if normalize_exposure
            else "raw sensor intensity")
    return ObservationStack(
        intensity=stack, angles_rad=angles,
        saturation=np.asarray(sats, dtype=np.float64),
        noise_sigma=np.asarray(sigmas, dtype=np.float64),
        frame_specs=specs, is_color=is_color, unit=unit)


def estimate_translation(reference: np.ndarray,
                         target: np.ndarray) -> tuple[int, int]:
    """相位相关估计 target 相对 reference 的整像素平移 (dy, dx)。"""
    ref = reference.astype(np.float64)
    tgt = target.astype(np.float64)
    ref -= ref.mean()
    tgt -= tgt.mean()
    f_ref = np.fft.fft2(ref)
    f_tgt = np.fft.fft2(tgt)
    cross = f_ref * np.conj(f_tgt)
    mag = np.abs(cross)
    mag[mag == 0] = 1e-12
    corr = np.fft.ifft2(cross / mag).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    h, w = ref.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)


def shift_image(image: np.ndarray, dy: int, dx: int,
                fill: float = np.nan) -> np.ndarray:
    """整像素平移校正，越界区域填 fill（后续按缺失样本排除）。"""
    out = np.full_like(image, fill, dtype=np.float64)
    h, w = image.shape[:2]

    src_y0 = max(0, dy)
    src_y1 = min(h, h + dy)
    dst_y0 = max(0, -dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, dx)
    src_x1 = min(w, w + dx)
    dst_x0 = max(0, -dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def register_stack(stack: np.ndarray, reference_index: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对灰度栈 (H,W,K) 做配准。

    返回 (配准后栈, 平移表 (K,2), 越界缺失遮罩 (H,W,K))。
    彩色栈请按通道调用或先转亮度。
    """
    if stack.ndim != 3:
        raise ValueError("register_stack 仅接受 (H, W, K) 灰度栈")
    h, w, k = stack.shape
    ref = stack[..., reference_index]
    aligned = np.empty_like(stack, dtype=np.float64)
    shifts = np.zeros((k, 2), dtype=np.int32)
    missing = np.zeros(stack.shape, dtype=bool)

    for j in range(k):
        if j == reference_index:
            aligned[..., j] = stack[..., j]
            continue
        dy, dx = estimate_translation(ref, stack[..., j])
        # estimate_translation 返回 target 内容相对参考帧的位移；
        # 校正回参考帧坐标系需要反向平移。
        shifts[j] = (dy, dx)
        moved = shift_image(stack[..., j], -dy, -dx)
        aligned[..., j] = moved
        missing[..., j] = ~np.isfinite(moved)
    return aligned, shifts, missing


def export_fit_npz(path: str, result, scene_unit: str) -> None:
    """把拟合结果与量纲信息导出为 NPZ。

    包含全部分量、不确定度、状态、有效样本遮罩、角度与预测强度，
    数组保持拟合时的浮点量纲（不做 8 bit 压缩）。
    """
    np.savez_compressed(
        path,
        A=result.A, B=result.B, C=result.C,
        R=result.R, phi_rad=result.phi_rad, dolp=result.dolp,
        sigma_A=result.sigma_A, sigma_R=result.sigma_R,
        sigma_phi_rad=result.sigma_phi_rad, sigma_dolp=result.sigma_dolp,
        residual_rms=result.residual_rms,
        reduced_chi2=result.reduced_chi2,
        max_abs_residual=result.max_abs_residual,
        n_residual_outliers=result.n_residual_outliers,
        n_valid=result.n_valid, state=np.asarray(result.state, dtype=str),
        valid=result.valid, angles_rad=result.angles_rad,
        observed=result.intensity, predicted=result.predicted,
        saturation=np.asarray(result.config.saturation),
        noise_sigma=np.asarray(result.config.noise_sigma),
        unit=np.asarray(scene_unit))
