# 使用指南

## 1. 图形界面（推荐入口）

```bash
./scripts/run_gui.sh        # 或 python -m polanalyzer.gui.app / pol-gui
```

启动后自动载入一个内置合成场景，包含：

- 6 个分析角（0°–150°，间隔 30°）；
- 已知高斯噪声 σ=8（12 bit 灰度，饱和值 4095）；
- 左上角局部遮挡（0°、30° 两角度缺失）；
- 中部运动条带（90° 注入平移并在运动遮罩标记）；
- 一条弱响应带（R 与噪声同量级）与一个高反射斑。

操作：

1. 顶部下拉框切换分析角，图像恢复窗口同步更新；
2. 点击图像任意像素 → 观测曲线与参数面板刷新；
3. 被排除样本显示红叉，信息栏列出每个样本的排除原因；
4. “重新拟合”使用当前配置重跑引擎；
5. “导出 NPZ”保存全部结果（见第 4 节）。

## 2. 真实图像清单格式

`manifest.json`：

```json
{
  "frames": [
    {
      "path": "frames/a000.tif",
      "angle_deg": 0,
      "exposure_s": 1.0,
      "gain": 1.0,
      "saturation": 65535
    },
    {
      "path": "frames/a045.tif",
      "angle_deg": 45,
      "exposure_s": 2.0,
      "gain": 1.0,
      "saturation": 65535
    }
  ]
}
```

- `saturation` 可省略：8 bit 图默认 255，16 bit 图默认 65535；
- `exposure_s`、`gain` 省略时视为 1.0；
- 角度无需预先排序，载入时按角度排序；
- 支持单通道（灰度）与多通道（RGB，数组形状 H×W×3×K）。

命令：

```bash
python -m polanalyzer.cli fit manifest.json \
    --read-noise 6 --register --out result.npz
```

## 3. 合成数据核对

```bash
python -m polanalyzer.cli synthetic \
    --angles 0 30 60 90 120 150 \
    --noise 8 --saturation 4095 --weak-snr 3 \
    --out synthetic_fit.npz --report report.json
```

输出同时保存 `*_truth.npz`（生成用的 A/B/C 真值与无噪声理想强度），
便于离线核对估计偏差，而不是拿拟合结果自证。

## 4. NPZ 导出内容（保留量纲）

所有数组为拟合时的浮点精度，不做 8 bit 压缩：

| 键 | 内容 |
|----|------|
| A, B, C, R, phi_rad, dolp | 拟合分量与派生量 |
| sigma_A, sigma_R, sigma_phi_rad, sigma_dolp | 不确定度 |
| residual_rms, reduced_chi2, max_abs_residual | 残差诊断 |
| n_residual_outliers, n_valid | 异常计数与有效角度数 |
| state | 每像素状态字符串 |
| valid | 逐样本有效遮罩（H×W×K） |
| angles_rad | 分析角 |
| observed, predicted | 观测与模型预测强度 |
| saturation, noise_sigma | 拟合使用的标定值 |
| unit | 量纲说明字符串 |

读取：

```python
import numpy as np
d = np.load("result.npz", allow_pickle=True)
print(d["unit"], d["state"].shape, d["angles_rad"])
```

## 5. 离线环境准备（无 root 的 Debian 容器）

本仓库开发环境的做法，可复现：

1. 用户态安装 pip 与 Python 依赖：
   ```bash
   python get-pip.py --user --break-system-packages
   python -m pip install --user --break-system-packages \
       numpy matplotlib pillow PySide6
   ```
2. Qt6 运行需要 libEGL/libxkbcommon 等系统库。无 apt 权限时，从
   `deb.debian.org` 下载对应 arm64/amd64 的 `.deb`，`ar x` +
   `tar xf data.tar.*` 解包到 `~/.local/syslib`，并把
   `~/.local/syslib/usr/lib/<arch>-linux-gnu` 加入
   `LD_LIBRARY_PATH`（`run_gui.sh` 自动处理）。
3. 中文字体下载 `fonts-wqy-microhei` 解包到 `~/.local/fonts`；
   界面同时配置 matplotlib 与 Qt 的字体。也可用环境变量
   `POL_FONT=/path/to/font.ttf` 指定。
4. 无显示服务器时以离屏平台验证逻辑：
   `QT_QPA_PLATFORM=offscreen ./scripts/run_gui.sh`（仅自动化测试用，
   人工分析需要真实显示）。

## 6. 测试

```bash
python -m unittest discover -s tests
```

覆盖：设计矩阵正交性、分量往返、饱和/运动/缺失排除、真值参数核对、
φ 的 90° 歧义、弱响应状态、物理违例检测、多通道拟合、曝光归一、
配准符号与越界遮罩、NPZ 量纲保留。
