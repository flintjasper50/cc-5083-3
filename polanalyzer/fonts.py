"""matplotlib 中文字体与负号显示配置。

按以下顺序查找可用 CJK 字体：
1. 环境变量 POL_FONT 指向的字体文件；
2. ~/.local/fonts 下的文泉驿/Noto 字体；
3. matplotlib 自带字体中已含 CJK 字形者。

找不到时回退默认字体并保留英文标签，不抛异常。
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
from matplotlib import font_manager

_CJK_CANDIDATES = (
    "wqy-microhei.ttc", "wqy-zenhei.ttc",
    "NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf",
)


def find_cjk_font_path() -> Path | None:
    """返回搜索到的第一个 CJK 字体文件路径，供 Qt QFontDatabase 使用。"""
    env_font = os.environ.get("POL_FONT")
    if env_font and Path(env_font).exists():
        return Path(env_font)
    for base in (Path.home() / ".local" / "fonts",):
        if not base.exists():
            continue
        for candidate in _CJK_CANDIDATES:
            hits = list(base.rglob(candidate))
            if hits:
                return hits[0]
    return None


def configure_fonts() -> str | None:
    """注册用户目录 CJK 字体并设置 rcParams，返回实际选中的字体名。"""
    search_dirs = [Path.home() / ".local" / "fonts"]
    env_font = os.environ.get("POL_FONT")
    if env_font:
        search_dirs.insert(0, Path(env_font).parent)

    chosen_path = None
    for base in search_dirs:
        if not base.exists():
            continue
        for candidate in _CJK_CANDIDATES:
            hits = list(base.rglob(candidate))
            if hits:
                chosen_path = hits[0]
                break
        if chosen_path:
            break
    if env_font and Path(env_font).exists():
        chosen_path = Path(env_font)

    if chosen_path is None:
        return None

    font_manager.fontManager.addfont(str(chosen_path))
    name = font_manager.FontProperties(fname=str(chosen_path)).get_name()
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name
