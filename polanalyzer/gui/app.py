"""偏振估计图形界面（图形入口）。

启动方式：
    python -m polanalyzer.gui.app
    或使用安装后的 pol-gui 命令。

界面结构
--------
- 左上：图像恢复窗口，可选择分析角显示该角度的观测图，点击任意像素；
- 左下：单像素观测曲线（有效样本、被排除样本及原因、拟合曲线与不确定带），
  异常像素同时显示诊断假设、证据与建议；
- 右侧：分量图（A / R / 偏振角 / DoLP / 残差 / 可靠性状态 / 诊断假设）；
- 所有图来自 engine 的真实拟合结果；诊断在后台线程运行，状态条可见，
  异常以界面消息条提示，不静默吞掉。

诊断与图像恢复窗口联动：选择分析角并点击像素时，同时给出该像素的
分量图位置、观测曲线与对应异常原因（附证据）。诊断只读观测与拟合结果，
不重排角度、不夹紧负值；确实无法区分时界面保留候选解释。

说明边界（界面文案与 docs 保持一致）：
- 恒定分量 A 不命名为"无反光真实颜色"，未偏振镜面反射与其不可区分；
- 弱响应区给出 weak_response 状态，不承诺偏振角/偏振度可解释。
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from .. import faults as fault_lib
from ..bundle import extract_bundle
from ..diagnostics import (DIAGNOSIS_LABELS, DiagnosisConfig,
                           diagnose_stack)
from ..engine import FitConfig, fit_pixels
from ..masks import STATE_WEAK
from ..synthetic import demo_truth, make_scene
from ..viz import (STATE_COLORS, STATE_LABELS, angle_hsv_image,
                   component_arrays, pixel_curve_data, state_overlay)

# 诊断假设的界面着色（与 viz.STATE_COLORS 区分，避免混淆拟合状态与诊断结论）
HYP_COLORS = {
    "consistent_with_model": (0.20, 0.65, 0.30),
    "angle_order_mismatch": (0.85, 0.25, 0.25),
    "exposure_inconsistency": (0.95, 0.55, 0.10),
    "angle_and_exposure": (0.70, 0.25, 0.75),
    "local_motion": (0.20, 0.45, 0.85),
    "ambiguous": (0.95, 0.75, 0.15),
    "not_estimable": (0.55, 0.55, 0.55),
}

FAULT_SCENES = [
    ("正常观测（噪声/遮挡/已知运动）", "clean"),
    ("仅分析角顺序错配", "angle_only"),
    ("仅照片曝光不一致", "exposure_only"),
    ("角度错配 + 曝光变化", "angle_and_exposure"),
    ("局部场景运动（未知遮罩）", "motion_only"),
]


def _import_qt():
    from ..fonts import configure_fonts
    configure_fonts()
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PySide6 import QtCore, QtGui, QtWidgets
    return (matplotlib, FigureCanvasQTAgg, NavigationToolbar2QT, Figure,
            QtCore, QtGui, QtWidgets)


def _apply_qt_font(QtGui, QtWidgets, app) -> None:
    """QApplication 存在后载入 CJK 字体并应用（过早调用会崩溃）。"""
    from ..fonts import find_cjk_font_path
    font_path = find_cjk_font_path()
    if font_path is None:
        return
    font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
    if font_id >= 0:
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QtGui.QFont(families[0], 10))


COMPONENT_SPECS = [
    ("A_恒定分量", "gray", "强度量纲"),
    ("R_调变幅度", "gray", "强度量纲"),
    ("phi_偏振角_rad", "angle", "弧度，仅可靠像素着色"),
    ("DoLP_线偏振度", "viridis", "0..1"),
    ("残差RMS", "magma", "强度量纲"),
    ("归一化卡方", "cividis", "无量纲"),
    ("可靠性状态", "state", "分类"),
]


class PolarizationWindow:
    """由 main() 构造，持有场景、拟合结果与全部控件。"""

    def __init__(self, argv):
        (mpl, FigureCanvas, NavToolbar, Figure,
         QtCore, QtGui, QtWidgets) = _import_qt()
        self._qt = (QtCore, QtGui, QtWidgets)
        self.FigureCanvas = FigureCanvas
        self.NavToolbar = NavToolbar
        self.Figure = Figure

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
        self.app = app
        _apply_qt_font(QtGui, QtWidgets, app)

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("多角度线性偏振分量分析器")
        self.window.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.window.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

        # --- 分析角选择 + 图像恢复窗口 ---
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("分析角:"))
        self.angle_box = QtWidgets.QComboBox()
        controls.addWidget(self.angle_box)
        controls.addWidget(QtWidgets.QLabel("受控场景:"))
        self.scene_box = QtWidgets.QComboBox()
        for label, _ in FAULT_SCENES:
            self.scene_box.addItem(label)
        controls.addWidget(self.scene_box)
        self.fit_button = QtWidgets.QPushButton("重新拟合")
        controls.addWidget(self.fit_button)
        self.diag_button = QtWidgets.QPushButton("运行异常诊断")
        controls.addWidget(self.diag_button)
        self.export_button = QtWidgets.QPushButton("导出 NPZ（保留量纲）")
        controls.addWidget(self.export_button)
        self.bundle_button = QtWidgets.QPushButton("提取诊断包")
        controls.addWidget(self.bundle_button)
        controls.addStretch(1)
        left.addLayout(controls)

        self.image_fig = Figure(figsize=(6, 4))
        self.image_canvas = FigureCanvas(self.image_fig)
        self.image_ax = self.image_fig.add_subplot(111)
        self.image_cid = None
        left.addWidget(self.image_canvas)
        left.addWidget(NavToolbar(self.image_canvas, self.window))

        self.curve_fig = Figure(figsize=(6, 3.2))
        self.curve_canvas = FigureCanvas(self.curve_fig)
        left.addWidget(self.curve_canvas)

        # --- 分量图网格 ---
        self.comp_fig = Figure(figsize=(6, 8))
        self.comp_canvas = FigureCanvas(self.comp_fig)
        right.addWidget(self.comp_canvas)

        # --- 像素信息与消息条 ---
        self.info = QtWidgets.QLabel("点击左侧图像选择像素")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-family: monospace; padding: 4px;")
        left.addWidget(self.info)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            "background:#f5f5dc; padding:6px; border:1px solid #aaa;")
        left.addWidget(self.status)

        # 偏振估计/诊断运行状态（独立于消息条，长时间任务期间持续可见）
        self.run_state = QtWidgets.QLabel("运行状态：空闲")
        self.run_state.setStyleSheet(
            "background:#e8eef7; padding:4px; border:1px solid #99b;")
        left.addWidget(self.run_state)

        self.angle_box.currentIndexChanged.connect(self._show_angle_frame)
        self.scene_box.currentIndexChanged.connect(self._switch_scene)
        self.fit_button.clicked.connect(self._refit)
        self.diag_button.clicked.connect(self._run_diagnosis)
        self.export_button.clicked.connect(self._export)
        self.bundle_button.clicked.connect(self._save_bundle)

        self.diagnosis = None
        self.diagnosis_worker = None
        self._load_demo_scene()
        self._run_fit()
        self._populate_angles()
        self._show_angle_frame()
        self._draw_components()
        self._bind_pick()

    # ---------- 数据 ----------

    def _base_scene(self):
        A, B, C = demo_truth()
        return make_scene(
            A, B, C, [0, 30, 60, 90, 120, 150],
            noise_sigma=8.0,
            occluder_cells=[(slice(12, 22), slice(14, 26), [0, 1])],
            motion_cells=[(slice(48, 58), slice(60, 74), [3])])

    def _load_demo_scene(self):
        try:
            self.scene = self._base_scene()
            # 故障场景（FaultScenario）不携带饱和/噪声量纲，统一由基础场景保存
            self._base_saturation = self.scene.saturation
            self._base_noise_sigma = self.scene.noise_sigma
            # 界面默认场景仍把已知运动遮罩交给引擎，保持原有演示行为
            self.known_motion = self.scene.motion_mask
            self.known_missing = self.scene.missing_mask
            self.unit = "合成线性传感器灰度（0..4095），噪声 sigma=8"
            self._notify("已载入内置合成场景（含已知噪声、局部遮挡与运动区）")
        except Exception as exc:  # 界面内提示，不静默
            self._fatal("场景载入失败", exc)

    def _switch_scene(self):
        """切换受控故障场景。只重建观测副本，绝不修改原场景数组。"""
        try:
            kind = FAULT_SCENES[self.scene_box.currentIndex()][1]
            base = self._base_scene()
            self.scene = base
            self._base_saturation = base.saturation
            self._base_noise_sigma = base.noise_sigma
            self.known_motion = None
            self.known_missing = None
            if kind == "clean":
                self.known_motion = base.motion_mask
                self.known_missing = base.missing_mask
            elif kind == "angle_only":
                sc = fault_lib.inject_angle_permutation(
                    base.intensity, base.angles_rad,
                    permutation=(1, 0, 2, 3, 4, 5))
                self.scene = sc
            elif kind == "exposure_only":
                sc = fault_lib.inject_exposure_gains(
                    base.intensity, base.angles_rad,
                    gains=(1.0, 1.0, 1.0, 1.6, 1.0, 1.0))
                self.scene = sc
            elif kind == "angle_and_exposure":
                self.scene = fault_lib.combine(
                    fault_lib.inject_angle_permutation(
                        base.intensity, base.angles_rad,
                        permutation=(1, 0, 2, 3, 4, 5)),
                    fault_lib.inject_exposure_gains(
                        base.intensity, base.angles_rad,
                        gains=(1.0, 1.0, 1.0, 1.6, 1.0, 1.0)))
            elif kind == "motion_only":
                # 不提供运动遮罩：诊断必须仅凭残差结构识别局部运动
                self.scene = fault_lib.inject_local_motion(
                    base.intensity, base.angles_rad,
                    slice(48, 58), slice(60, 74), 3, shift=150.0)
            self.diagnosis = None
            self._run_fit()
            self._show_angle_frame()
            self._draw_components()
            if hasattr(self, "_selected"):
                self._draw_curve(*self._selected)
            self._notify(f"已切换受控场景：{FAULT_SCENES[self.scene_box.currentIndex()][0]}")
        except Exception as exc:
            self._fatal("场景切换失败", exc)

    def _run_fit(self):
        s = self.scene
        saturation = getattr(s, "saturation", self._base_saturation)
        noise_sigma = getattr(s, "noise_sigma", self._base_noise_sigma)
        self.config = FitConfig(saturation=saturation,
                                noise_sigma=noise_sigma)
        self.result = fit_pixels(
            s.intensity, s.angles_rad, self.config,
            motion_mask=getattr(self, "known_motion", None),
            missing_mask=getattr(self, "known_missing", None))
        self.components = component_arrays(self.result)

    def _run_diagnosis(self):
        """后台线程运行诊断，运行状态在界面持续可见，不阻塞图像窗口。"""
        if self.diagnosis_worker is not None:
            self._notify("诊断仍在运行，请等待完成")
            return
        from PySide6 import QtCore

        class DiagnosisWorker(QtCore.QObject):
            finished = QtCore.Signal(object)
            failed = QtCore.Signal(str)

            def __init__(self, scene, result):
                super().__init__()
                self.scene = scene
                self.result = result

            @QtCore.Slot()
            def run(self):
                try:
                    report = diagnose_stack(
                        self.scene.intensity, self.scene.angles_rad,
                        self.result, config=DiagnosisConfig())
                    self.finished.emit(report)
                except Exception as exc:  # 传回主线程提示，不静默
                    self.failed.emit(f"{type(exc).__name__}: {exc}")

        from PySide6 import QtCore
        self._diag_thread = QtCore.QThread()
        self.diagnosis_worker = DiagnosisWorker(self.scene, self.result)
        self.diagnosis_worker.moveToThread(self._diag_thread)
        self._diag_thread.started.connect(self.diagnosis_worker.run)
        self.diagnosis_worker.finished.connect(self._on_diagnosis_done)
        self.diagnosis_worker.failed.connect(self._on_diagnosis_failed)
        self.diagnosis_worker.finished.connect(self._diag_thread.quit)
        self.diagnosis_worker.failed.connect(self._diag_thread.quit)
        self._diag_thread.finished.connect(self._cleanup_diag_thread)
        self.run_state.setText("运行状态：正在执行异常诊断（角度重配/增益/运动检验）…")
        self.diag_button.setEnabled(False)
        self._diag_thread.start()

    def _cleanup_diag_thread(self):
        self.diagnosis_worker = None
        self.diag_button.setEnabled(True)

    def _on_diagnosis_done(self, report):
        self.diagnosis = report
        self.run_state.setText(
            "运行状态：诊断完成。假设计数 "
            + ", ".join(f"{DIAGNOSIS_LABELS[k].split('（')[0]}={v}"
                        for k, v in sorted(report.counts.items(),
                                           key=lambda kv: -kv[1])))
        self._draw_components()
        self._show_angle_frame()
        if hasattr(self, "_selected"):
            self._draw_curve(*self._selected)
        self._notify("异常诊断完成：证据与建议见像素信息栏，假设图见右侧")

    def _on_diagnosis_failed(self, message):
        self.run_state.setText("运行状态：诊断失败")
        self._fatal("异常诊断失败", RuntimeError(message))

    def _save_bundle(self):
        if self.diagnosis is None:
            self._notify("请先运行异常诊断再提取诊断包")
            return
        try:
            from PySide6 import QtWidgets
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self.window, "选择诊断包输出目录")
            if not path:
                return
            bundle_path = extract_bundle(
                self.diagnosis, self.result, self.scene.intensity, path,
                name="polarization_diagnostic",
                extra_note=f"GUI 受控场景: "
                           f"{FAULT_SCENES[self.scene_box.currentIndex()][0]}")
            self._notify(
                f"诊断包已提取: {bundle_path}（含 manifest/data/rebuild.py，"
                "可在本界面之外离线重建同一结论）")
        except Exception as exc:
            self._fatal("诊断包提取失败", exc)

    def _refit(self):
        try:
            self._run_fit()
            self.diagnosis = None  # 数据/配置变化后旧诊断失效，需重新运行
            self.run_state.setText("运行状态：拟合完成，诊断待运行")
            self._show_angle_frame()
            self._draw_components()
            if hasattr(self, "_selected"):
                self._draw_curve(*self._selected)
            counts = self._state_counts()
            self._notify("拟合完成: " + ", ".join(
                f"{STATE_LABELS[k].split(' ')[0]}={v}" for k, v in counts.items()))
        except Exception as exc:
            self._fatal("拟合失败", exc)

    def _state_counts(self):
        states, cnt = np.unique(self.result.state, return_counts=True)
        return dict(zip(states, cnt))

    # ---------- 控件填充 ----------

    def _populate_angles(self):
        self.angle_box.blockSignals(True)
        self.angle_box.clear()
        for deg in np.rad2deg(self.scene.angles_rad):
            self.angle_box.addItem(f"{deg:.0f}°")
        self.angle_box.blockSignals(False)

    def _bind_pick(self):
        if self.image_cid is not None:
            self.image_canvas.mpl_disconnect(self.image_cid)
        self.image_cid = self.image_canvas.mpl_connect(
            "button_press_event", self._on_pick)

    def _on_pick(self, event):
        if event.inaxes != self.image_ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        h, w = self.result.A.shape[:2]
        if not (0 <= row < h and 0 <= col < w):
            return
        self._selected = (row, col)
        try:
            self._draw_curve(row, col)
        except Exception as exc:
            self._fatal("观测曲线绘制失败", exc)

    # ---------- 绘制 ----------

    def _hypothesis_overlay(self):
        """把诊断假设图着色为 RGB（无诊断时返回 None）。"""
        if self.diagnosis is None:
            return None
        hmap = self.diagnosis.hypothesis_map
        out = np.zeros((*hmap.shape, 3))
        for hyp, color in HYP_COLORS.items():
            out[hmap == hyp] = color
        return out

    def _show_angle_frame(self):
        idx = self.angle_box.currentIndex()
        if idx < 0:
            return
        frame = self.scene.intensity[..., idx].copy()
        self.image_fig.clear()
        self.image_ax = self.image_fig.add_subplot(111)
        vmax = np.nanpercentile(frame[np.isfinite(frame)], 99)
        self.image_ax.imshow(frame, cmap="gray", vmin=0, vmax=vmax,
                             interpolation="nearest")
        self.image_ax.set_title(
            f"图像恢复窗口 — 分析角 {np.rad2deg(self.scene.angles_rad[idx]):.0f}°\n"
            "点击像素查看观测曲线与诊断原因")
        if hasattr(self, "_selected"):
            row, col = self._selected
            self.image_ax.plot(col, row, "ro", ms=8, mfc="none", mew=1.5)
        # 在原图上定位诊断异常区域（红框/红点），不修改原图数组
        if self.diagnosis is not None:
            from ..diagnostics import HYP_CLEAN, HYP_UNFITTABLE
            hmap = self.diagnosis.hypothesis_map
            abnormal = ~np.isin(hmap, [HYP_CLEAN, HYP_UNFITTABLE])
            if abnormal.any():
                rr, cc = np.where(abnormal)
                self.image_ax.scatter(cc, rr, s=4, facecolors="none",
                                      edgecolors="red", linewidths=0.4,
                                      label="诊断异常像素")
                rows, cols = np.where(abnormal)
                r0, r1 = rows.min(), rows.max() + 1
                c0, c1 = cols.min(), cols.max() + 1
                import matplotlib.patches as mpatches
                rect = mpatches.Rectangle(
                    (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                    fill=False, edgecolor="yellow", lw=1.2,
                    label="异常区域包围盒")
                self.image_ax.add_patch(rect)
                self.image_ax.legend(fontsize=7, loc="upper right")
        self.image_canvas.draw_idle()
        self._bind_pick()

    def _draw_components(self):
        fig = self.comp_fig
        fig.clear()
        hyp_overlay = self._hypothesis_overlay()
        n_rows = 4 if hyp_overlay is None else 5
        axes = fig.subplots(n_rows, 2)
        for ax, (name, cmap, note) in zip(axes.flat, COMPONENT_SPECS):
            if name == "可靠性状态":
                img = state_overlay(self.result.state)
                ax.imshow(img, interpolation="nearest")
            elif name == "phi_偏振角_rad":
                img = angle_hsv_image(self.result.phi_rad, self.result.R,
                                      self.result.state)
                ax.imshow(img, interpolation="nearest")
            else:
                data = self.components[name]
                finite = data[np.isfinite(data)]
                vmax = (np.nanpercentile(finite, 98)
                        if finite.size else None)
                ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                          interpolation="nearest")
            ax.set_title(f"{name}\n({note})", fontsize=9)
            ax.axis("off")

        legend_ax = axes.flat[len(COMPONENT_SPECS) - 1]
        legend_ax.axis("off")
        legend_ax.set_xlim(0, 1)
        legend_ax.set_ylim(0, len(STATE_COLORS))
        from matplotlib.patches import Rectangle
        for i, (st, color) in enumerate(STATE_COLORS.items()):
            y = len(STATE_COLORS) - 1 - i
            legend_ax.add_patch(
                Rectangle((0.02, y + 0.2), 0.1, 0.6, color=color))
            legend_ax.text(0.16, y + 0.5, STATE_LABELS[st],
                           fontsize=8, va="center")

        if hyp_overlay is not None:
            ax_hyp = axes.flat[len(COMPONENT_SPECS)]
            ax_hyp.imshow(hyp_overlay, interpolation="nearest")
            ax_hyp.set_title("诊断假设（异常原因）\n(证据见像素信息栏)",
                             fontsize=9)
            ax_hyp.axis("off")
            leg2 = axes.flat[len(COMPONENT_SPECS) + 1]
            leg2.axis("off")
            leg2.set_xlim(0, 1)
            leg2.set_ylim(0, len(HYP_COLORS))
            for i, (hyp, color) in enumerate(HYP_COLORS.items()):
                y = len(HYP_COLORS) - 1 - i
                leg2.add_patch(
                    Rectangle((0.02, y + 0.2), 0.08, 0.6, color=color))
                leg2.text(0.13, y + 0.5, DIAGNOSIS_LABELS[hyp],
                          fontsize=7, va="center")
        fig.tight_layout()
        self.comp_canvas.draw_idle()

    def _find_record(self, row, col):
        if self.diagnosis is None:
            return None
        for rec in self.diagnosis.records:
            if rec.row == row and rec.col == col \
                    and rec.channel in (None, 0):
                return rec
        return None

    def _draw_curve(self, row: int, col: int):
        d = pixel_curve_data(self.result, row, col)
        fig = self.curve_fig
        fig.clear()
        ax = fig.add_subplot(111)

        ax.fill_between(d["fit_angles_deg"], d["band_lower"],
                        d["band_upper"], color="tab:blue", alpha=0.15,
                        label="R 不确定带")
        ax.plot(d["fit_angles_deg"], d["fit_curve"], "-",
                color="tab:blue", lw=1.5, label="模型拟合")

        valid = d["valid"]
        ax.plot(d["angles_deg"][valid], d["observed"][valid], "o",
                color="tab:green", ms=7, label="有效观测")
        excluded = ~valid
        if excluded.any():
            reason_text = ", ".join(
                f"{a:.0f}°:{r}" for a, r in zip(
                    d["angles_deg"][excluded],
                    d["exclusion_reasons"][excluded]))
            ax.plot(d["angles_deg"][excluded],
                    np.nan_to_num(d["observed"][excluded], nan=0.0),
                    "x", color="tab:red", ms=9, mew=2,
                    label="被排除样本")
        else:
            reason_text = "无"

        # 标出诊断识别出的残差热点帧（标准化残差超阈值），与排除样本区分
        rec = self._find_record(row, col)
        if rec is not None and rec.hot_angles_deg:
            hot = np.array(rec.hot_angles_deg)
            obs = d["observed"]
            hot_y = [obs[np.argmin(np.abs(d["angles_deg"] - a))]
                     for a in hot]
            ax.plot(hot, np.nan_to_num(hot_y, nan=0.0), "D",
                    color="tab:orange", ms=8, mfc="none", mew=1.6,
                    label="诊断残差热点帧")

        title = f"像素 (row={row}, col={col}) 观测曲线"
        if rec is not None:
            title += f" — {DIAGNOSIS_LABELS[rec.hypothesis]}"
        ax.set_xlabel("分析角 (度)")
        ax.set_ylabel("强度（原始量纲）")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.curve_canvas.draw_idle()

        phi_txt = ("不可解释（弱响应）" if d["state"] == STATE_WEAK
                   else f"{d['phi_deg']:.1f} ± {d['sigma_phi_deg']:.1f}°")
        diag_text = ""
        rec = self._find_record(row, col)
        if rec is not None:
            diag_text = (
                f"\n--- 异常诊断 ---\n"
                f"假设: {DIAGNOSIS_LABELS[rec.hypothesis]}\n"
                f"候选解释: {', '.join(DIAGNOSIS_LABELS[c] for c in rec.candidates)}\n"
                f"求解条件: 有效观测 {rec.n_valid}, 秩 {rec.design_rank}, "
                f"条件数 {rec.condition_number}\n"
                f"证据:\n" + "\n".join(f"  - {e}" for e in rec.evidence)
                + f"\n建议（附证据，不自动改数据）: {rec.suggestion}")
        elif self.diagnosis is not None:
            hyp = self.diagnosis.hypothesis_map[row, col]
            diag_text = (f"\n--- 异常诊断 ---\n假设: {DIAGNOSIS_LABELS[hyp]}"
                         "（该像素记录未入包，细节见同区域代表记录）")
        self.info.setText(
            f"像素 ({row},{col})  状态: {STATE_LABELS[d['state']]}\n"
            f"A = {d['A']:.2f} ± {d['sigma_A']:.2f}    "
            f"R = {d['R']:.2f} ± {d['sigma_R']:.2f}\n"
            f"phi = {phi_txt}    DoLP = {d['dolp']:.3f} "
            f"± {d['sigma_dolp']:.3f}\n"
            f"有效角度 {d['n_valid']}/{d['angles_deg'].size}    "
            f"残差RMS = {d['residual_rms']:.2f}    "
            f"卡方 = {d['reduced_chi2']:.2f}\n"
            f"排除样本: {reason_text}\n"
            f"量纲: {self.unit}\n"
            "注: A 为角度恒定分量，含未偏振镜面反射，不等于无反光真实颜色"
            + diag_text)

    # ---------- 导出与消息 ----------

    def _export(self):
        try:
            from PySide6 import QtWidgets
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.window, "导出拟合结果", "polarization_fit.npz",
                "NPZ (*.npz)")
            if not path:
                return
            from ..io_stack import export_fit_npz
            export_fit_npz(path, self.result, self.unit)
            self._notify(f"已导出 {path}（分量、不确定度、状态、遮罩与预测强度，浮点量纲保留）")
        except Exception as exc:
            self._fatal("导出失败", exc)

    def _notify(self, text: str):
        self.status.setText(text)

    def _fatal(self, title: str, exc: Exception):
        self.status.setText(f"[异常] {title}: {exc}")
        traceback.print_exc()

    def show(self):
        self.window.show()
        return self.app.exec()


def main(argv=None):
    win = PolarizationWindow(argv if argv is not None else sys.argv)
    raise SystemExit(win.show())


if __name__ == "__main__":
    main()
