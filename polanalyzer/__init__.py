"""多角度线性偏振照片分量分析器。

核心能力：
- model：声明的成像模型与设计矩阵
- masks：饱和 / 运动 / 缺失角度的排除规则
- engine：逐像素加权最小二乘拟合、残差与可靠性状态
- io_stack：观测栈的载入与量纲保留
- synthetic：可控角度响应的合成观测生成（含噪声与局部遮挡）

图形界面入口见 polanalyzer.gui.app，命令行仅为辅助。
"""

from .model import MODEL_FORMULA, design_matrix, predict, derive_components

__all__ = ["MODEL_FORMULA", "design_matrix", "predict", "derive_components"]
