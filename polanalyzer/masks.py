"""观测样本的有效性规则。

同一像素在某个分析角下的观测，出现以下任一情况即不得作为有效方程：
- 饱和：强度达到传感器饱和值（钳位），真实强度不可知；
- 运动：配准残差表明该帧该位置存在运动/错位，不满足静态场景假设；
- 缺失：该角度没有数据（NaN / 遮罩标记）；
- 非法：负强度或非有限值。

排除在角度维度上逐像素生效：同一像素的其他角度仍可参与拟合。
"""

from __future__ import annotations

import numpy as np

# 状态码（逐像素的拟合结论，详见 engine 模块）
STATE_OK = "ok"
STATE_WEAK = "weak_response"
STATE_FEW_ANGLES = "insufficient_angles"
STATE_PHYSICAL = "unphysical_fit"
STATE_ALL_INVALID = "no_valid_samples"

ALL_STATES = (STATE_OK, STATE_WEAK, STATE_FEW_ANGLES,
              STATE_PHYSICAL, STATE_ALL_INVALID)


def valid_sample_mask(intensity: np.ndarray,
                      saturation: float,
                      motion_mask: np.ndarray | None = None,
                      missing_mask: np.ndarray | None = None) -> np.ndarray:
    """计算逐像素、逐角度的有效样本布尔遮罩。

    参数
    ----
    intensity : 形状 (..., K) 的观测强度，K 为角度数。
    saturation : 传感器饱和值；强度 >= saturation 视为饱和。
    motion_mask : 形状 (..., K) 的布尔数组，True 表示该样本存在运动。
    missing_mask : 形状 (..., K) 的布尔数组，True 表示该角度缺失。

    返回
    ----
    boolean 数组，形状与 intensity 相同，True 表示该样本可进入方程。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    valid = np.ones(intensity.shape, dtype=bool)

    valid &= np.isfinite(intensity)
    valid &= intensity >= 0.0
    valid &= intensity < float(saturation)

    if motion_mask is not None:
        valid &= ~np.asarray(motion_mask, dtype=bool)
    if missing_mask is not None:
        valid &= ~np.asarray(missing_mask, dtype=bool)
    return valid


def exclusion_reasons(intensity: np.ndarray,
                      saturation: float,
                      motion_mask: np.ndarray | None = None,
                      missing_mask: np.ndarray | None = None) -> np.ndarray:
    """返回逐样本的排除原因字符串数组（空字符串表示有效）。

    优先级：missing > motion > nonfinite/negative > saturated。
    供界面在单像素观测曲线上标注每个被排除的样本。
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    reasons = np.full(intensity.shape, "", dtype=object)

    bad_value = ~np.isfinite(intensity) | (intensity < 0.0)
    saturated = np.isfinite(intensity) & (intensity >= float(saturation))

    reasons[bad_value] = "missing_or_invalid"
    reasons[saturated] = "saturated"
    if motion_mask is not None:
        mm = np.asarray(motion_mask, dtype=bool)
        reasons[mm & (reasons == "")] = "motion"
    if missing_mask is not None:
        mm = np.asarray(missing_mask, dtype=bool)
        reasons[mm] = "missing"
    return reasons
