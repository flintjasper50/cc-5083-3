#!/usr/bin/env bash
# 偏振估计图形界面启动脚本（图形入口）。
# 若 Qt 依赖库安装在用户目录（无 root 环境下的本地 .deb 解包），
# 通过 LD_LIBRARY_PATH 注入；正常系统安装时该目录不存在则跳过。
set -euo pipefail

SYS_LIB="${HOME}/.local/syslib/usr/lib/$(uname -m)-linux-gnu"
if [ -d "${SYS_LIB}" ]; then
    export LD_LIBRARY_PATH="${SYS_LIB}:${LD_LIBRARY_PATH:-}"
fi

# 可选：通过 POL_FONT 指定自定义 CJK 字体文件。
export QT_API="${QT_API:-pyside6}"

exec python3 -m polanalyzer.gui.app "$@"
