"""偏振估计图形界面（图形入口）。

启动方式：
    python -m polanalyzer.gui.app
    或使用安装后的 pol-gui 命令。

界面结构
--------
- 左上：图像恢复窗口，可选择分析角显示该角度的观测图，点击任意像素；
- 左下：单像素观测曲线（有效样本、被排除样本及原因、拟合曲线与不确定带）；
- 右侧：分量图（A / R / 偏振角 / DoLP / 残差 / 可靠性状态）；
- 所有图来自 engine 的真实拟合结果；异常以界面消息条提示，不静默吞掉。

说明边界（界面文案与 docs 保持一致）：
- 恒定分量 A 不命名为"无反光真实颜色"，未偏振镜面反射与其不可区分；
- 弱响应区给出 weak_response 状态，不承诺偏振角/偏振度可解释。
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from ..engine import FitConfig, fit_pixels
from ..masks import STATE_WEAK
from ..synthetic import demo_truth, make_scene
from ..viz import (STATE_COLORS, STATE_LABELS, angle_hsv_image,
                   component_arrays, pixel_curve_data, state_overlay)


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
        self.fit_button = QtWidgets.QPushButton("重新拟合")
        controls.addWidget(self.fit_button)
        self.export_button = QtWidgets.QPushButton("导出 NPZ（保留量纲）")
        controls.addWidget(self.export_button)
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

        self.angle_box.currentIndexChanged.connect(self._show_angle_frame)
        self.fit_button.clicked.connect(self._refit)
        self.export_button.clicked.connect(self._export)

        self._load_demo_scene()
        self._run_fit()
        self._populate_angles()
        self._show_angle_frame()
        self._draw_components()
        self._bind_pick()

    # ---------- 数据 ----------

    def _load_demo_scene(self):
        try:
            A, B, C = demo_truth()
            self.scene = make_scene(
                A, B, C, [0, 30, 60, 90, 120, 150],
                noise_sigma=8.0,
                occluder_cells=[(slice(12, 22), slice(14, 26), [0, 1])],
                motion_cells=[(slice(48, 58), slice(60, 74), [3])])
            self.unit = "合成线性传感器灰度（0..4095），噪声 sigma=8"
            self._notify("已载入内置合成场景（含已知噪声、局部遮挡与运动区）")
        except Exception as exc:  # 界面内提示，不静默
            self._fatal("场景载入失败", exc)

    def _run_fit(self):
        s = self.scene
        self.config = FitConfig(saturation=s.saturation,
                                noise_sigma=s.noise_sigma)
        self.result = fit_pixels(
            s.intensity, s.angles_rad, self.config,
            motion_mask=s.motion_mask, missing_mask=s.missing_mask)
        self.components = component_arrays(self.result)

    def _refit(self):
        try:
            self._run_fit()
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
            "点击像素查看观测曲线")
        if hasattr(self, "_selected"):
            row, col = self._selected
            self.image_ax.plot(col, row, "ro", ms=8, mfc="none", mew=1.5)
        self.image_canvas.draw_idle()
        self._bind_pick()

    def _draw_components(self):
        fig = self.comp_fig
        fig.clear()
        axes = fig.subplots(4, 2)
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

        legend_ax = axes.flat[-1]
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
        fig.tight_layout()
        self.comp_canvas.draw_idle()

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

        ax.set_xlabel("分析角 (度)")
        ax.set_ylabel("强度（原始量纲）")
        ax.set_title(f"像素 (row={row}, col={col}) 观测曲线")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.curve_canvas.draw_idle()

        phi_txt = ("不可解释（弱响应）" if d["state"] == STATE_WEAK
                   else f"{d['phi_deg']:.1f} ± {d['sigma_phi_deg']:.1f}°")
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
            "注: A 为角度恒定分量，含未偏振镜面反射，不等于无反光真实颜色")

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
