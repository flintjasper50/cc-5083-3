"""可复跑的偏振估计异常诊断。

诊断只在**观测副本与重拟合**上进行，绝不重排用户的角度、不夹紧负值、
不修改原拟合结果。对每个采样点记录：

- 该点实际采用的角度序列与逐样本观测、有效标记、排除原因；
- 进入方程的有效观测数、设计矩阵秩与条件数（求解条件）；
- 基线最小二乘的参数、残差、逐角度重投影值与标准化残差；
- 三类受控反事实检验得到的证据：
  角度重配检验、逐帧曝光增益检验、局部运动空间一致性检验。

三类目标故障的可辨识结构（也是受控错误发生器所模拟的）：

- 角度顺序错配：观测仍满足某个角度排列下的二倍角余弦模型，
  重配角度后残差大幅下降；单帧常数增益解释不了这种调变错位；
- 曝光不一致：某帧整体乘了常数增益，表现为该帧残差沿帧方向同号，
  逐帧（空间稳健）增益估计显著偏离 1，增益校正后残差下降，
  而角度重配无系统性收益；
- 局部场景运动：只有小区域、单帧的观测偏离模型，全局增益与角度
  重配都无法修复，异常在空间上聚成连通小块且落在被标记/可疑运动帧。

证据不足时不下单因结论，保留候选解释（ambiguous），不把残差归咎噪声。
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .engine import solve_wls
from .masks import exclusion_reasons
from .model import design_matrix, predict

# 假设标签
HYP_CLEAN = "consistent_with_model"
HYP_ANGLE = "angle_order_mismatch"
HYP_EXPOSURE = "exposure_inconsistency"
HYP_JOINT = "angle_and_exposure"
HYP_MOTION = "local_motion"
HYP_AMBIGUOUS = "ambiguous"
HYP_UNFITTABLE = "not_estimable"

DIAGNOSIS_LABELS = {
    HYP_CLEAN: "与模型一致（残差可由噪声解释）",
    HYP_ANGLE: "分析角顺序错配",
    HYP_EXPOSURE: "照片曝光不一致（逐帧增益）",
    HYP_JOINT: "角度错配与曝光变化共同存在",
    HYP_MOTION: "局部场景运动",
    HYP_AMBIGUOUS: "证据不足，保留候选解释",
    HYP_UNFITTABLE: "有效观测不足，无法诊断",
}

# 建议只描述可执行的核查动作，不自动改写数据
SUGGESTIONS = {
    HYP_ANGLE: "核对清单/元数据中的 angle_deg 与检偏器实际位置；"
               "可在诊断包内按重配排列复算，确认后再由用户修正角度声明",
    HYP_EXPOSURE: "核对该帧 exposure_s/gain 与 EXIF；可在诊断包内应用"
                  "估计增益复算，确认后由用户修正曝光声明或重新曝光归一",
    HYP_JOINT: "先按建议增益校正曝光，再按重配排列核对角度，二者需分别留痕",
    HYP_MOTION: "对该帧该区域重新配准或重拍；不要通过夹紧负值/删点掩盖，"
                "可在诊断包内排除该样本后复算分量",
    HYP_AMBIGUOUS: "现有证据无法在候选解释间区分，需补充观测或人工核查",
}


@dataclass
class DiagnosisConfig:
    """诊断阈值（全部进入诊断包，保证结论可复跑）。"""

    residual_sigma_hot: float = 4.0       # |标准化残差| 超过该值视为热点
    chi2_elevated: float = 4.0            # reduced chi2 超过该值视为残差异常
    angle_rel_gain: float = 0.30          # 重配/增益修正相对收益阈值
    angle_min_chi2: float = 2.5           # 基线 chi2 低于此值不允许报角度故障
    global_prevalence: float = 0.20       # 帧级故障需占可诊断像素的最小比例
    exposure_gain_tol: float = 0.10       # |g-1| 超过 10% 视为显著增益
    exposure_mad_z: float = 5.0           # |g-1|/MAD 超过该 z 值才算低散布增益
    motion_fraction: float = 0.60         # 区域内异常像素占比阈值
    motion_min_pixels: int = 4            # 连通运动块的最小像素数
    max_permutations: int = 120           # K! 过大时抽样的排列上限
    weak_evidence_margin: float = 0.15    # 两类修正收益差小于此值判为歧义


@dataclass
class PixelRecord:
    """单个采样点（像素/通道）的完整诊断记录，全部为 JSON 可序列化类型。"""

    row: int
    col: int
    channel: int | None
    state: str
    angles_deg: list[float]
    observed: list[float]
    predicted: list[float]
    residual: list[float]
    standardized_residual: list[float]
    valid: list[bool]
    exclusion_reasons: list[str]
    n_valid: int
    design_rank: int | None
    condition_number: float | None
    A: float | None
    B: float | None
    C: float | None
    reduced_chi2: float | None
    residual_rms: float | None
    # 证据
    angle_perm_best: list[int] | None
    angle_chi2_base: float | None
    angle_chi2_best: float | None
    angle_relative_improvement: float | None
    gain_estimates: list[float]
    gain_max_abs_dev: float | None
    gain_chi2_corrected: float | None
    gain_relative_improvement: float | None
    joint_chi2: float | None
    joint_relative_improvement: float | None
    hot_angles_deg: list[float]
    # 结论
    hypothesis: str
    candidates: list[str]
    evidence: list[str]
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiagnosisReport:
    """整幅（或区域）诊断报告。records 仅保留异常像素与少量正常对照。"""

    hypothesis_map: np.ndarray            # 字符串图，与分量图同空间形状
    records: list[PixelRecord]
    angles_rad: np.ndarray
    config: DiagnosisConfig
    global_gains: np.ndarray              # (K,) 逐帧稳健增益估计
    global_gain_mad: np.ndarray           # (K,) 增益估计的稳健散布
    baseline_chi2_median: float
    noise_sigma: float
    region: tuple | None                  # (r0, r1, c0, c1) 或 None（整幅）
    component_equivalence: dict           # 正常观测诊断前后一致性核对
    counts: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "hypothesis_counts": self.counts,
            "global_gains": [float(x) for x in self.global_gains],
            "global_gain_mad": [float(x) for x in self.global_gain_mad],
            "baseline_chi2_median": float(self.baseline_chi2_median),
            "noise_sigma": float(self.noise_sigma),
            "region": self.region,
            "angles_deg": [float(x) for x in np.rad2deg(self.angles_rad)],
            "config": asdict(self.config),
            "component_equivalence": self.component_equivalence,
            "records": [r.to_dict() for r in self.records],
        }


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def _iter_spatial(shape: tuple, multichannel: bool):
    """生成 (row, col, channel) 索引与扁平索引。"""
    if multichannel:
        h, w, c = shape
        for r in range(h):
            for cc in range(w):
                for ch in range(c):
                    yield r, cc, ch
    else:
        h, w = shape
        for r in range(h):
            for cc in range(w):
                yield r, cc, None


def _spatial_shape(intensity: np.ndarray) -> tuple[tuple, bool]:
    if intensity.ndim == 3:
        return intensity.shape[:2], False
    if intensity.ndim == 4:
        return intensity.shape[:3], True
    raise ValueError("诊断仅接受 (H,W,K) 或 (H,W,C,K) 观测栈")


def _take(arr: np.ndarray, r: int, c: int, ch: int | None):
    if ch is None:
        return arr[r, c]
    return arr[r, c, ch]


def _permutations(k: int, rng: np.random.Generator, cap: int):
    """除恒等排列外的角度排列；k! 过大时固定种子抽样，保证可复跑。"""
    identity = tuple(range(k))
    all_perms = [p for p in itertools.permutations(range(k)) if p != identity]
    if len(all_perms) <= cap:
        return [np.array(p) for p in all_perms]
    idx = rng.choice(len(all_perms), size=cap, replace=False)
    return [np.array(all_perms[i]) for i in sorted(idx.tolist())]


def _wls_chi2(X: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict:
    w = 1.0 / np.maximum(sigma ** 2, 1e-12)
    sol = solve_wls(X, y, w)
    if not sol["ok"]:
        return {"ok": False}
    dof = max(y.size - 3, 1)
    return {"ok": True, "chi2": sol["w_rss"] / dof,
            "beta": sol["beta"], "condition": sol["condition"],
            "rank": sol["rank"], "residual": sol["residual"]}


def _permuted_chi2_batch(X_full: np.ndarray, perms: list[np.ndarray],
                         valid_idx: np.ndarray, y: np.ndarray,
                         sigma: np.ndarray) -> np.ndarray:
    """向量化计算多个角度排列在同一有效子集上的 reduced chi2。

    排列只重排设计矩阵行；权重对所有排列相同，3x3 法方程批量求解，
    避免逐排列 Python 求逆（整幅诊断的主要耗时来源）。
    """
    if not perms:
        return np.empty(0)
    Xp = np.stack([X_full[p][valid_idx] for p in perms])  # (P,n,3)
    w = 1.0 / np.maximum(sigma ** 2, 1e-12)
    XtW = Xp * w[None, :, None]
    gram = np.einsum("pni,pnj->pij", XtW, Xp)
    rhs = np.einsum("pni,n->pi", XtW, y)
    try:
        beta = np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("pij,pj->pi", np.linalg.pinv(gram), rhs)
    resid = y[None, :] - np.einsum("pni,pi->pn", Xp, beta)
    dof = max(y.size - 3, 1)
    return np.sum(resid ** 2 * w[None, :], axis=1) / dof


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4 邻域连通块标注（仅依赖 numpy）。"""
    labels = np.full(mask.shape, -1, dtype=np.int32)
    n = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            stack = [(r, c)]
            labels[r, c] = n
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and mask[nr, nc] and labels[nr, nc] < 0):
                        labels[nr, nc] = n
                        stack.append((nr, nc))
            n += 1
    return labels, n


# --------------------------------------------------------------------------
# 全局逐帧增益（曝光）证据：留一帧稳健估计
# --------------------------------------------------------------------------

def estimate_frame_gains(intensity: np.ndarray,
                         predicted: np.ndarray,
                         valid: np.ndarray,
                         noise_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """在基线拟合的预测值上逐帧做截尾比值，估计 g_k 与稳健散布 MAD。

    诊断目的是判别而非精确标定：曝光故障表现为**整帧、低散布**的常数
    增益，角度错配产生的调变错位在帧内有正有负，截尾均值回到 ~1 且
    MAD 大；局部运动只污染少量像素，被截尾剔除。因此同时返回点估计与
    MAD，显著性判据使用 |g-1| 相对 MAD 而不是裸点估计。
    """
    k = intensity.shape[-1]
    flat_i = intensity.reshape(-1, k)
    flat_p = predicted.reshape(-1, k)
    flat_v = valid.reshape(-1, k)
    good = flat_v & np.isfinite(flat_i) & np.isfinite(flat_p) & (flat_p > 50)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(good, flat_i / np.maximum(flat_p, 1e-9), np.nan)
    gains = np.ones(k)
    mads = np.zeros(k)
    for j in range(k):
        vals = ratio[:, j]
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            continue
        lo, hi = np.percentile(vals, [20, 80])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        med = float(np.median(trimmed))
        gains[j] = med
        mads[j] = float(np.median(np.abs(vals - med)))
    return gains, mads


# --------------------------------------------------------------------------
# 逐像素诊断
# --------------------------------------------------------------------------

def diagnose_pixel(observed: np.ndarray,
                   angles_rad: np.ndarray,
                   valid: np.ndarray,
                   sigma: float | np.ndarray,
                   global_gains: np.ndarray,
                   config: DiagnosisConfig,
                   rng: np.random.Generator,
                   result_state: str,
                   global_gain_mad: np.ndarray | None = None,
                   motion_known: np.ndarray | None = None,
                   fit_beta: np.ndarray | None = None,
                   fit_cond: float | None = None,
                   fit_rank: int | None = None,
                   fit_chi2: float | None = None,
                   ) -> dict:
    """对单个采样点跑三类反事实检验并形成带证据的假设结论。"""
    k = angles_rad.size
    v = np.asarray(valid, dtype=bool)
    y_all = np.asarray(observed, dtype=np.float64)
    sig_all = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (k,))
    X_full = design_matrix(angles_rad)
    reasons = exclusion_reasons(
        y_all[None, :], saturation=np.inf,
        motion_mask=(None if motion_known is None
                     else np.asarray(motion_known, bool)[None, :]))[0]

    base = {
        "valid": v.tolist(), "exclusion_reasons": [str(x) for x in reasons],
        "n_valid": int(v.sum()), "design_rank": fit_rank,
        "condition_number": (None if fit_cond is None
                             else float(fit_cond) if np.isfinite(fit_cond)
                             else None),
        "A": None, "B": None, "C": None,
        "reduced_chi2": (None if fit_chi2 is None else float(fit_chi2)),
        "residual_rms": None,
        "angle_perm_best": None, "angle_chi2_base": None,
        "angle_chi2_best": None, "angle_relative_improvement": None,
        "gain_estimates": [float(g) for g in global_gains],
        "gain_max_abs_dev": float(np.max(np.abs(global_gains - 1.0))),
        "gain_chi2_corrected": None, "gain_relative_improvement": None,
        "joint_chi2": None, "joint_relative_improvement": None,
        "hot_angles_deg": [],
        "predicted": [None] * k, "residual": [None] * k,
        "standardized_residual": [None] * k,
    }

    n_valid = int(v.sum())
    if n_valid < 3:
        base.update(hypothesis=HYP_UNFITTABLE,
                    candidates=[HYP_UNFITTABLE], evidence=[],
                    suggestion="有效观测不足 3 个，模型不可辨识，无法做异常区分")
        return base

    X = X_full[v]
    y = y_all[v]
    sig = np.asarray(sig_all, dtype=np.float64)[v]
    sol = _wls_chi2(X, y, sig)
    if not sol["ok"]:
        base.update(hypothesis=HYP_UNFITTABLE,
                    candidates=[HYP_UNFITTABLE], evidence=[],
                    suggestion="设计矩阵奇异，无法求解")
        return base

    beta = (np.asarray(fit_beta, dtype=np.float64)
            if fit_beta is not None else sol["beta"])
    pred_all = X_full @ beta
    resid_all = y_all - pred_all
    std_all = resid_all / np.asarray(sig_all, dtype=np.float64)
    rms = float(np.sqrt(np.mean(sol["residual"] ** 2)))
    base.update(
        A=float(beta[0]), B=float(beta[1]), C=float(beta[2]),
        reduced_chi2=float(sol["chi2"]), residual_rms=rms,
        design_rank=int(sol["rank"]),
        condition_number=(float(sol["condition"])
                          if np.isfinite(sol["condition"]) else None),
        predicted=[float(x) if np.isfinite(x) else None for x in pred_all],
        residual=[float(x) if np.isfinite(x) else None for x in resid_all],
        standardized_residual=[float(x) if np.isfinite(x) else None
                               for x in std_all],
        hot_angles_deg=[float(np.rad2deg(angles_rad[j]))
                        for j in range(k)
                        if v[j] and abs(std_all[j]) > config.residual_sigma_hot])

    chi2_base = sol["chi2"]
    evidence: list[str] = []
    candidates: list[str] = []
    perms = _permutations(k, rng, config.max_permutations)
    valid_idx = np.flatnonzero(v)

    # --- 检验 1：角度重配（排列重拟合，向量化批量求解） ---
    perm_chi2 = _permuted_chi2_batch(X_full, perms, valid_idx, y, sig)
    if perm_chi2.size:
        best_idx = int(np.argmin(perm_chi2))
        best = float(perm_chi2[best_idx])
        best_perm = perms[best_idx]
    else:
        best, best_perm = None, None
    angle_imp = (chi2_base - best) / max(chi2_base, 1e-12) if best else 0.0
    base.update(angle_perm_best=([int(x) for x in best_perm]
                                 if best_perm is not None else None),
                angle_chi2_base=float(chi2_base),
                angle_chi2_best=best,
                angle_relative_improvement=float(angle_imp))

    # --- 检验 2：曝光增益校正（使用全局稳健增益，避免单像素过拟合） ---
    y_gain = y_all[v] / global_gains[v]
    gain_sol = _wls_chi2(X, y_gain, sig / global_gains[v])
    gain_imp = 0.0
    if gain_sol["ok"]:
        gain_imp = (chi2_base - gain_sol["chi2"]) / max(chi2_base, 1e-12)
        base.update(gain_chi2_corrected=float(gain_sol["chi2"]),
                    gain_relative_improvement=float(gain_imp))
    gain_dev = base["gain_max_abs_dev"]

    # --- 检验 3：先增益校正、再在其上做角度重配（联合反事实） ---
    joint_chi2_arr = _permuted_chi2_batch(
        X_full, perms, valid_idx, y_gain, sig / global_gains[v])
    joint_best = (float(joint_chi2_arr.min())
                  if joint_chi2_arr.size and gain_sol["ok"] else None)
    joint_imp = ((chi2_base - joint_best) / max(chi2_base, 1e-12)
                 if joint_best is not None else 0.0)
    if joint_best is not None:
        base.update(joint_chi2=joint_best,
                    joint_relative_improvement=float(joint_imp))

    # 门控：基线残差本身不显著时，排列过拟合不得作为角度故障证据；
    # 全局增益需同时满足点偏离与低散布（相对 MAD 的 z 值）才算曝光故障。
    if global_gain_mad is not None:
        mad_floor = 1e-3
        gain_z = np.max(np.abs(global_gains - 1.0)
                        / np.maximum(global_gain_mad, mad_floor))
    else:
        gain_z = 0.0
    angle_strong = (angle_imp >= config.angle_rel_gain
                    and chi2_base >= config.angle_min_chi2)
    gain_strong = (gain_imp >= config.angle_rel_gain
                   and gain_dev >= config.exposure_gain_tol
                   and gain_z >= config.exposure_mad_z)
    joint_strong = (joint_imp >= config.angle_rel_gain
                    and angle_strong and gain_strong)

    if angle_strong:
        evidence.append(
            f"角度重配后 reduced chi2 {chi2_base:.2f} -> {best:.2f}"
            f"（相对下降 {angle_imp:.0%}），最佳排列 {base['angle_perm_best']}")
        candidates.append(HYP_ANGLE)
    if gain_strong:
        bad = int(np.argmax(np.abs(global_gains - 1.0)))
        evidence.append(
            f"帧 {np.rad2deg(angles_rad[bad]):.0f}° 稳健增益 "
            f"{global_gains[bad]:.3f}（|g-1|={gain_dev:.3f}），"
            f"增益校正后 chi2 -> {gain_sol['chi2']:.2f}"
            f"（相对下降 {gain_imp:.0%}）")
        candidates.append(HYP_EXPOSURE)

    # --- 结论判定：先看残差是否可由噪声解释，再区分结构与局部运动 ---
    if chi2_base <= config.chi2_elevated and not angle_strong and not gain_strong:
        hypothesis = HYP_CLEAN
        evidence.insert(0,
                        f"reduced chi2={chi2_base:.2f} 未超过阈值 "
                        f"{config.chi2_elevated}，残差与噪声水平相容")
    elif joint_strong and joint_imp > max(angle_imp, gain_imp) + config.weak_evidence_margin:
        hypothesis = HYP_JOINT
        evidence.append("联合（增益+重配）修正显著优于任一单因修正")
    elif angle_strong and gain_strong:
        if abs(angle_imp - gain_imp) <= config.weak_evidence_margin:
            hypothesis = HYP_AMBIGUOUS
            evidence.append("角度重配与增益校正收益相当，单像素证据无法区分")
        else:
            hypothesis = HYP_JOINT
            evidence.append("两类证据同时显著，按共同存在处理")
    elif angle_strong:
        hypothesis = HYP_ANGLE
    elif gain_strong:
        hypothesis = HYP_EXPOSURE
    else:
        # 残差异常但角度/曝光的全局反事实都解释不了 -> 局部运动候选，
        # 由 diagnose_stack 的空间连通性确认或降级为歧义
        hypothesis = HYP_MOTION
        hot = ", ".join(f"{a:.0f}°" for a in base["hot_angles_deg"]) or "（无单帧超 4σ）"
        evidence.append(
            f"chi2={chi2_base:.2f} 超阈值但角度重配/全局增益均无显著修正，"
            f"热点帧 {hot}，符合局部、单帧偏离特征")
        candidates.append(HYP_MOTION)

    if hypothesis in (HYP_ANGLE, HYP_EXPOSURE) and HYP_MOTION not in candidates:
        # 若同时带单帧局部热点，保留运动作为次要候选
        if base["hot_angles_deg"] and hypothesis == HYP_EXPOSURE:
            candidates.append(HYP_MOTION)
    if not candidates:
        candidates = [hypothesis]
    elif hypothesis not in candidates:
        candidates.insert(0, hypothesis)

    base.update(hypothesis=hypothesis, candidates=candidates, evidence=evidence,
                suggestion=SUGGESTIONS.get(hypothesis, SUGGESTIONS[HYP_AMBIGUOUS]))
    return base


# --------------------------------------------------------------------------
# 整幅 / 区域诊断
# --------------------------------------------------------------------------

def diagnose_stack(intensity: np.ndarray,
                   angles_rad: np.ndarray,
                   result,
                   config: DiagnosisConfig | None = None,
                   region: tuple | None = None,
                   record_fraction: float = 0.02,
                   rng_seed: int = 20261113,
                   ) -> DiagnosisReport:
    """对拟合结果运行可复跑诊断。

    result : engine.FitResult（提供基线参数、valid、状态、条件数等，
             诊断只读不改）。
    region : (r0, r1, c0, c1)，仅诊断并记录该窗口；None 表示整幅。
    record_fraction : 正常对照像素的最大记录比例（控制逐像素记录体积），
             异常像素始终全量记录。
    """
    config = config or DiagnosisConfig()
    rng = np.random.default_rng(rng_seed)
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    spatial, multichannel = _spatial_shape(intensity)

    sigma = float(np.mean(result.config.noise_sigma)) if np.ndim(
        result.config.noise_sigma) else float(result.config.noise_sigma)
    global_gains, global_mad = estimate_frame_gains(
        intensity, result.predicted, result.valid, sigma)
    finite_chi2 = result.reduced_chi2[np.isfinite(result.reduced_chi2)]
    baseline_med = float(np.median(finite_chi2)) if finite_chi2.size else np.nan

    r0, r1, c0, c1 = region or (0, spatial[0], 0, spatial[1])
    win_shape = ((r1 - r0, c1 - c0, spatial[2]) if multichannel
                 else (r1 - r0, c1 - c0))
    # 区域诊断只返回窗口形状的假设图；连通性分析也限定在窗口内
    hyp_map = np.full(win_shape, HYP_UNFITTABLE, dtype=object)
    pre_motion = np.zeros(win_shape[:2], dtype=bool)
    prelim: dict[tuple, dict] = {}

    all_indices = list(_iter_spatial(win_shape, multichannel))

    for r, c, ch in all_indices:
        rr, cc = r + r0, c + c0
        key = (r, c) if ch is None else (r, c, ch)
        obs = _take(intensity, rr, cc, ch)
        val = _take(result.valid, rr, cc, ch)
        st = str(_take(result.state, rr, cc, ch))
        beta = np.array([_take(result.A, rr, cc, ch),
                         _take(result.B, rr, cc, ch),
                         _take(result.C, rr, cc, ch)])
        cond = _take(result.condition_number, rr, cc, ch)
        rank = _take(result.design_rank, rr, cc, ch)
        chi = _take(result.reduced_chi2, rr, cc, ch)
        motion_known = (None if result.motion_mask is None
                        else _take(result.motion_mask, rr, cc, ch))
        rec = diagnose_pixel(
            obs, angles_rad, val, sigma, global_gains, config, rng,
            result_state=st, global_gain_mad=global_mad,
            motion_known=motion_known,
            fit_beta=(beta if np.isfinite(beta).all() else None),
            fit_cond=(None if not np.isfinite(cond) else float(cond)),
            fit_rank=(None if not np.isfinite(rank) else int(rank)),
            fit_chi2=(None if not np.isfinite(chi) else float(chi)))
        prelim[key] = rec
        if rec["hypothesis"] == HYP_MOTION:
            pre_motion[r, c] = True

    # 帧级故障（角度/曝光）必须在全图普遍出现；局部运动只占小连通块。
    n_diag = max(sum(1 for k in prelim
                     if prelim[k]["hypothesis"] != HYP_UNFITTABLE), 1)
    angle_prev = sum(1 for k in prelim
                     if HYP_ANGLE in prelim[k]["candidates"]) / n_diag
    exposure_prev = sum(1 for k in prelim
                        if HYP_EXPOSURE in prelim[k]["candidates"]) / n_diag
    angle_global = angle_prev >= config.global_prevalence
    mad_floor = 1e-3
    gain_z_global = float(np.max(
        np.abs(global_gains - 1.0) / np.maximum(global_mad, mad_floor)))
    exposure_global = (exposure_prev >= config.global_prevalence
                       and float(np.max(np.abs(global_gains - 1.0)))
                       >= config.exposure_gain_tol
                       and gain_z_global >= config.exposure_mad_z)

    # 区分"仅曝光"与"角度+曝光共存"：若逐帧增益已把全图中位残差校正回
    # 噪声水平，则角度排列只是增益残差上的过拟合，不承认角度共存；
    # 若增益校正后全图中位 chi2 仍显著高于噪声阈值，则存在增益解释不了
    # 的结构性失配，结合角度普遍性承认共存。
    if exposure_global:
        gain_chi2_all = [
            rec["gain_chi2_corrected"] for rec in prelim.values()
            if rec.get("gain_chi2_corrected") is not None]
        median_gain = float(np.median(gain_chi2_all)) if gain_chi2_all else np.inf
        angle_coexists = (angle_prev >= config.global_prevalence
                          and median_gain >= config.angle_min_chi2)
    else:
        angle_coexists = angle_global
    prevalence_note = (
        f"帧级普遍性: 角度候选 {angle_prev:.0%}（门控 "
        f"{config.global_prevalence:.0%}，{'通过' if angle_global else '不通过'}），"
        f"曝光候选 {exposure_prev:.0%}"
        f"（{'通过' if exposure_global else '不通过'}）")

    for rec in prelim.values():
        angle_allowed = angle_coexists if exposure_global else angle_global
        if HYP_ANGLE in rec["candidates"] and not angle_allowed:
            rec["candidates"] = [c for c in rec["candidates"]
                                 if c != HYP_ANGLE]
            rec["evidence"] = rec["evidence"] + [
                "角度证据未通过全图普遍性门控，单像素重配收益不提升为全局结论"]
        # 曝光是显著的帧级现象且不承认共存时，增益校正已解释的像素
        # 不再保留角度候选（排列只是对增益残差的过拟合）。
        if (HYP_ANGLE in rec["candidates"] and exposure_global
                and rec.get("gain_chi2_corrected") is not None
                and rec["gain_chi2_corrected"] < config.angle_min_chi2):
            rec["candidates"] = [c for c in rec["candidates"]
                                 if c != HYP_ANGLE]
            rec["evidence"] = rec["evidence"] + [
                "全局曝光增益校正后该像素残差已回到噪声水平，"
                "角度重配收益判定为增益残差上的过拟合"]
        if HYP_EXPOSURE in rec["candidates"] and not exposure_global:
            rec["candidates"] = [c for c in rec["candidates"]
                                 if c != HYP_EXPOSURE]
            rec["evidence"] = rec["evidence"] + [
                "曝光证据未通过普遍性/增益门控，降级为局部候选"]

    # 空间连通性确认运动块；孤立热点降级为歧义，避免把噪声当运动
    motion_confirmed = np.zeros(win_shape[:2], dtype=bool)
    if pre_motion.any():
        labels_cc, n_labels = _label_components(pre_motion)
        for lab in range(n_labels):
            size = int((labels_cc == lab).sum())
            if size >= config.motion_min_pixels:
                motion_confirmed |= labels_cc == lab

    records: list[PixelRecord] = []
    rng_record = np.random.default_rng(rng_seed + 1)
    clean_kept = 0
    clean_budget = max(1, int(record_fraction * len(all_indices)))
    counts: dict[str, int] = {}

    for r, c, ch in all_indices:
        rr, cc = r + r0, c + c0
        key = (r, c) if ch is None else (r, c, ch)
        rec = dict(prelim[key])
        hypothesis = rec["hypothesis"]

        # 普遍性门控后按剩余候选重定最终标签，证据全部保留在 evidence 中
        if hypothesis not in (HYP_CLEAN, HYP_UNFITTABLE):
            has_a = HYP_ANGLE in rec["candidates"]
            has_e = HYP_EXPOSURE in rec["candidates"]
            has_m = HYP_MOTION in rec["candidates"]
            if has_a and has_e:
                hypothesis = HYP_JOINT
            elif has_a:
                hypothesis = HYP_ANGLE
            elif has_e:
                hypothesis = HYP_EXPOSURE
            elif has_m and motion_confirmed[r, c]:
                hypothesis = HYP_MOTION
            else:
                hypothesis = HYP_AMBIGUOUS
                if HYP_AMBIGUOUS not in rec["candidates"]:
                    rec["candidates"] = rec["candidates"] + [HYP_AMBIGUOUS]
                rec["evidence"] = rec["evidence"] + [prevalence_note]
                rec["suggestion"] = SUGGESTIONS[HYP_AMBIGUOUS]
            rec["hypothesis"] = hypothesis

        if hypothesis == HYP_MOTION and not motion_confirmed[r, c]:
            hypothesis = HYP_AMBIGUOUS
            rec["hypothesis"] = HYP_AMBIGUOUS
            rec["candidates"] = sorted(set(rec["candidates"] + [HYP_AMBIGUOUS]))
            rec["evidence"] = rec["evidence"] + [
                "热点未形成满足最小尺寸的连通运动块，保留为歧义而非断言运动"]
            rec["suggestion"] = SUGGESTIONS[HYP_AMBIGUOUS]
        hyp_map[key] = hypothesis
        counts[hypothesis] = counts.get(hypothesis, 0) + 1

        is_clean = hypothesis == HYP_CLEAN
        keep = (not is_clean) or clean_kept < clean_budget
        if is_clean and keep:
            # 正常对照再做一层固定抽样，防止整幅正常像素撑大诊断包
            if rng_record.random() > min(1.0, clean_budget
                                         / max(1, counts.get(HYP_CLEAN, 1))):
                keep = False
        if not keep:
            continue
        if is_clean:
            clean_kept += 1
        records.append(PixelRecord(
            row=rr, col=cc, channel=ch, state=str(
                _take(result.state, rr, cc, ch)),
            angles_deg=[float(x) for x in np.rad2deg(angles_rad)],
            observed=[float(x) if np.isfinite(x) else None for x
                      in _take(intensity, rr, cc, ch)],
            predicted=rec["predicted"], residual=rec["residual"],
            standardized_residual=rec["standardized_residual"],
            valid=rec["valid"], exclusion_reasons=rec["exclusion_reasons"],
            n_valid=rec["n_valid"], design_rank=rec["design_rank"],
            condition_number=rec["condition_number"],
            A=rec["A"], B=rec["B"], C=rec["C"],
            reduced_chi2=rec["reduced_chi2"],
            residual_rms=rec["residual_rms"],
            angle_perm_best=rec["angle_perm_best"],
            angle_chi2_base=rec["angle_chi2_base"],
            angle_chi2_best=rec["angle_chi2_best"],
            angle_relative_improvement=rec["angle_relative_improvement"],
            gain_estimates=rec["gain_estimates"],
            gain_max_abs_dev=rec["gain_max_abs_dev"],
            gain_chi2_corrected=rec["gain_chi2_corrected"],
            gain_relative_improvement=rec["gain_relative_improvement"],
            joint_chi2=rec["joint_chi2"],
            joint_relative_improvement=rec["joint_relative_improvement"],
            hot_angles_deg=rec["hot_angles_deg"],
            hypothesis=hypothesis, candidates=rec["candidates"],
            evidence=rec["evidence"], suggestion=rec["suggestion"]))

    equivalence = verify_clean_equivalence(intensity, angles_rad, result,
                                           hyp_map, region)
    return DiagnosisReport(
        hypothesis_map=hyp_map, records=records, angles_rad=angles_rad,
        config=config, global_gains=global_gains, global_gain_mad=global_mad,
        baseline_chi2_median=baseline_med, noise_sigma=sigma, region=region,
        component_equivalence=equivalence, counts=counts)


def verify_clean_equivalence(intensity: np.ndarray,
                             angles_rad: np.ndarray,
                             result,
                             hyp_map: np.ndarray,
                             region: tuple | None) -> dict:
    """核查正常观测经过诊断路径后仍得到原有分量结果。

    在被判为 consistent_with_model 的像素上，用诊断路径自己的 WLS
    （engine.solve_wls + 同一有效遮罩与 sigma）重新求解，
    与原 FitResult 的 A/B/C 逐元素比对。
    """
    X_full = design_matrix(angles_rad)
    sigma = float(np.mean(result.config.noise_sigma)) if np.ndim(
        result.config.noise_sigma) else float(result.config.noise_sigma)
    spatial, multichannel = _spatial_shape(intensity)
    r0, r1, c0, c1 = region or (0, spatial[0], 0, spatial[1])
    max_abs_diff = 0.0
    checked = 0
    for r, c, ch in _iter_spatial(
            (r1 - r0, c1 - c0, spatial[2]) if multichannel
            else (r1 - r0, c1 - c0), multichannel):
        rr, cc = r + r0, c + c0
        key = (r, c) if ch is None else (r, c, ch)
        if hyp_map[key] != HYP_CLEAN:
            continue
        v = _take(result.valid, rr, cc, ch)
        if int(v.sum()) < 3:
            continue
        y = _take(intensity, rr, cc, ch)[v]
        sig = np.full(intensity.shape[-1], sigma)[v]
        sol = solve_wls(X_full[v], y, 1.0 / np.maximum(sig ** 2, 1e-12))
        if not sol["ok"]:
            continue
        orig = np.array([_take(result.A, rr, cc, ch),
                         _take(result.B, rr, cc, ch),
                         _take(result.C, rr, cc, ch)])
        max_abs_diff = max(max_abs_diff,
                           float(np.max(np.abs(sol["beta"] - orig))))
        checked += 1
    tol = 1e-8 * max(1.0, float(np.nanmax(np.abs(result.A))))
    return {
        "checked_clean_pixels": checked,
        "max_abs_beta_diff": max_abs_diff,
        "tolerance": tol,
        "equivalent": bool(max_abs_diff <= tol),
    }
