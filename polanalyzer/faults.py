"""受控错误发生器：在已知真值的合成栈上注入单一或组合故障。

三类故障与 diagnostics 的假设一一对应，便于构造"只有角度错误"、
"只有曝光变化"、"两者共同存在"以及"局部运动"对照：

- inject_angle_permutation：只重排角度索引（模拟角度声明/顺序错配），
  强度数值与噪声完全不变，因此任何残差都不能归咎于曝光或噪声；
- inject_exposure_gains：只对整帧乘常数增益（模拟曝光时间/增益不一致），
  角度声明保持不变；
- inject_local_motion：只在局部切片、单帧叠加平移量（模拟场景运动），
  并返回运动遮罩；其余像素与帧不变。

所有函数都在副本上操作，不修改输入栈，也不触碰任何拟合结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaultScenario:
    """注入后的观测与故障真值（供测试与诊断包核对，不参与拟合）。"""

    intensity: np.ndarray
    angles_rad: np.ndarray
    permutation: tuple | None         # 观测 j 实际来自原角度 permutation[j]
    gains: np.ndarray | None          # 逐帧乘性增益
    motion_mask: np.ndarray | None    # True = 该样本被注入局部运动
    fault_kinds: tuple[str, ...]
    region: tuple | None              # (r0,r1,c0,c1) 运动区域
    motion_angle_idx: int | None

    @property
    def has_angle_fault(self) -> bool:
        return "angle" in self.fault_kinds

    @property
    def has_exposure_fault(self) -> bool:
        return "exposure" in self.fault_kinds

    @property
    def has_motion_fault(self) -> bool:
        return "motion" in self.fault_kinds


def inject_angle_permutation(intensity: np.ndarray,
                             angles_rad: np.ndarray,
                             permutation: tuple | list | None = None,
                             rng_seed: int = 20261113) -> FaultScenario:
    """仅重排最后一维（角度维）的观测顺序，角度数组本身保持声明值。

    permutation[j] = p 表示新栈第 j 帧取自原栈第 p 帧。
    默认生成一个非恒等的确定性排列。
    """
    intensity = np.asarray(intensity)
    k = angles_rad.size
    if permutation is None:
        rng = np.random.default_rng(rng_seed)
        while True:
            perm = tuple(rng.permutation(k).tolist())
            if perm != tuple(range(k)):
                break
        permutation = perm
    permutation = tuple(int(x) for x in permutation)
    if sorted(permutation) != list(range(k)):
        raise ValueError("permutation 必须是 0..K-1 的一个排列")
    if permutation == tuple(range(k)):
        raise ValueError("恒等排列不构成角度错配故障")
    shuffled = intensity[..., np.array(permutation)].copy()
    return FaultScenario(
        intensity=shuffled, angles_rad=np.asarray(angles_rad).copy(),
        permutation=permutation, gains=None, motion_mask=None,
        fault_kinds=("angle",), region=None, motion_angle_idx=None)


def inject_exposure_gains(intensity: np.ndarray,
                          angles_rad: np.ndarray,
                          gains: tuple | list | np.ndarray | None = None,
                          rng_seed: int = 20261113) -> FaultScenario:
    """仅对整帧施加乘性增益，模拟曝光时间/增益不一致。"""
    k = angles_rad.size
    if gains is None:
        rng = np.random.default_rng(rng_seed)
        gains = np.ones(k)
        j = int(rng.integers(0, k))
        gains[j] = 1.6  # 单帧 60% 曝光偏差，默认确定性场景
    gains = np.asarray(gains, dtype=np.float64)
    if gains.shape != (k,):
        raise ValueError("增益长度必须等于角度数")
    if np.allclose(gains, 1.0):
        raise ValueError("全 1 增益不构成曝光故障")
    scaled = intensity * gains.reshape((1,) * (intensity.ndim - 1) + (k,))
    return FaultScenario(
        intensity=scaled, angles_rad=np.asarray(angles_rad).copy(),
        permutation=None, gains=gains, motion_mask=None,
        fault_kinds=("exposure",), region=None, motion_angle_idx=None)


def inject_local_motion(intensity: np.ndarray,
                        angles_rad: np.ndarray,
                        row_slice: slice,
                        col_slice: slice,
                        angle_idx: int,
                        shift: float = 120.0) -> FaultScenario:
    """仅在局部区域、单帧叠加常数平移量并标记运动遮罩。"""
    out = np.array(intensity, dtype=np.float64, copy=True)
    motion = np.zeros(intensity.shape, dtype=bool)
    indexer = (row_slice, col_slice, angle_idx)
    if out.ndim == 4:
        indexer = (row_slice, col_slice, slice(None), angle_idx)
    motion[indexer] = True
    out[indexer] += shift
    region = (row_slice.start, row_slice.stop,
              col_slice.start, col_slice.stop)
    return FaultScenario(
        intensity=out, angles_rad=np.asarray(angles_rad).copy(),
        permutation=None, gains=None, motion_mask=motion,
        fault_kinds=("motion",), region=region,
        motion_angle_idx=int(angle_idx))


def combine(*scenarios: FaultScenario) -> FaultScenario:
    """按顺序组合多个受控故障（角度、曝光、运动可叠加）。"""
    if not scenarios:
        raise ValueError("至少需要一个故障场景")
    base = scenarios[0]
    intensity = base.intensity.copy()
    angles = base.angles_rad.copy()
    perm = base.permutation
    gains = None if base.gains is None else base.gains.copy()
    motion = (None if base.motion_mask is None
              else base.motion_mask.copy())
    kinds = list(base.fault_kinds)
    region = base.region
    motion_idx = base.motion_angle_idx

    for sc in scenarios[1:]:
        if sc.permutation is not None:
            if perm is not None:
                raise ValueError("角度排列故障只能注入一次")
            intensity = intensity[..., np.array(sc.permutation)]
            perm = sc.permutation
        if sc.gains is not None:
            gains = (sc.gains.copy() if gains is None else gains * sc.gains)
            intensity = intensity * sc.gains.reshape(
                (1,) * (intensity.ndim - 1) + (-1,))
        if sc.motion_mask is not None:
            if motion is None:
                motion = np.zeros_like(intensity, dtype=bool)
            motion |= sc.motion_mask
            region = sc.region
            motion_idx = sc.motion_angle_idx
        kinds.extend(k for k in sc.fault_kinds if k not in kinds)

    return FaultScenario(
        intensity=intensity, angles_rad=angles, permutation=perm,
        gains=gains, motion_mask=motion, fault_kinds=tuple(kinds),
        region=region, motion_angle_idx=motion_idx)
