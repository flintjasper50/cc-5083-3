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

### 1.1 异常诊断与图像恢复窗口联动

顶部“受控场景”下拉框可在五个场景间切换：正常观测、仅分析角顺序错配、
仅曝光不一致、角度+曝光、局部场景运动（未知遮罩）。故障在观测副本上
注入，原始场景数组不被修改。

1. 点击“运行异常诊断”，诊断在后台线程执行，顶部“运行状态”条持续显示
   进度，完成后给出各假设的像素计数，不阻塞图像窗口；
2. 图像恢复窗口上异常像素以红色散点标出，异常区域以黄色包围盒定位；
3. 右侧分量图新增“诊断假设”一行，按假设类别着色并附图例；
4. 点击任意像素，观测曲线除有效/被排除样本与拟合带外，用橙色菱形标出
   热点角度，信息栏追加“异常诊断”块：最终假设、候选解释、设计秩与
   条件数、证据列表与建议；
5. 点击“提取诊断包”选择目录，导出可在本界面之外离线重建结论的诊断包。

诊断只读观测与拟合结果：不会自动重排照片、不会夹紧负值；无法区分时
界面保留候选解释（证据不足）而不是强行归因。

### 1.2 窗口启动步骤

```bash
# 1. 安装依赖（无 root 时加 --user --break-system-packages）
python -m pip install numpy matplotlib pillow PySide6

# 2. 有显示服务器时直接启动（脚本会自动注入本地 Qt 库路径）
./scripts/run_gui.sh

# 3. 无显示服务器（如远程/容器）仅做离屏逻辑自检
QT_QPA_PLATFORM=offscreen ./scripts/run_gui.sh
```

`run_gui.sh` 检测到 `~/.local/syslib/usr/lib/<arch>-linux-gnu` 存在时
自动加入 `LD_LIBRARY_PATH`；正常系统安装了 Qt 运行库时该目录不存在，
脚本透明跳过。离屏模式仅供自动化验证，人工分析需要真实显示。


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

## 4. 受控故障对照与异常诊断（命令行）

终端只辅助偏振估计，交互式像素检查仍以图形界面为主。

### 4.1 受控错误发生器与四类对照

`fault-demo` 在同一已知真值场景上分别注入故障并核对诊断结论，刻意构造
“只有角度错误”“只有曝光变化”“两者共同存在”以及局部运动对照，避免把
所有残差都归咎于噪声：

```bash
python -m polanalyzer.cli fault-demo \
    --height 48 --width 56 \
    --permutation 1 0 2 3 4 5 \
    --gains 1 1 1 1.6 1 1 \
    --motion-shift 150 \
    --bundle-dir examples/bundles --report fault_report.json
```

注入器（`polanalyzer/faults.py`）全部在副本上操作：角度故障只重排角度
轴、曝光故障只乘整帧常数增益、运动故障只在局部切片单帧叠加平移，
互不污染。退出码为 0 表示全部对照的主导假设与注入故障一致。

### 4.2 对真实清单运行诊断

```bash
python -m polanalyzer.cli diagnose manifest.json \
    --read-noise 8 \
    --region 20,40,25,45 \
    --bundle-dir out_bundles --bundle-name my_case
```

`--region r0,r1,c0,c1` 可把诊断限制在一个窗口，控制逐像素记录体积；
缺省整幅诊断。命令打印各假设计数、逐帧稳健增益与 MAD、正常观测分量
等价核查，以及证据最重的示例像素。诊断过程不修改任何输入图像与清单。

### 4.3 诊断包的离线重建

诊断包含 `manifest.json`、`data.npz`、`rebuild.py`。在任意机器上
（仅需 numpy 与 polanalyzer 包）：

```bash
python3 out_bundles/my_case/rebuild.py
# 逐像素假设一致: N/N
# 结论: 重建同一结论
```

脚本会逐级向上查找 polanalyzer 包，找不到则使用已安装版本，因此诊断包
可整体复制到脱离原仓库/原界面的位置独立复算。

## 5. 仓库内置示例数据

`examples/` 由 `examples/generate_examples.py` 一次性生成并纳入版本
管理，可直接走完整的“PNG + 清单 → 载入 → 诊断 → 诊断包”链路：

```bash
python3 examples/generate_examples.py           # 重新生成（确定性随机种子）
python3 -m polanalyzer.cli diagnose \
    examples/angle_only/manifest.json --read-noise 8 \
    --bundle-dir /tmp/bundles --bundle-name angle
```

目录内容：

| 路径 | 内容 |
|------|------|
| `clean/` | 正常对照：6 帧 16-bit PNG + manifest.json |
| `angle_only/` | 仅角度错配（帧内容按排列重排，清单仍按原角度声明） |
| `exposure_only/` | 仅曝光不一致（90° 帧未声明的 1.6 倍增益） |
| `angle_and_exposure/` | 两类故障共同存在 |
| `bundles/` | 各场景提取并已通过离线复算的诊断包 |
| `clean_reference.npz` | 无噪声真值与含噪观测，供离线核对 |

注意示例清单的 `gain` 都写成 1.0：要演示的是**未声明、未被共同曝光
解释捕获**的曝光偏差；若把真实增益写进清单，载入阶段就会被归一，故障
在进入诊断前即消失。

## 6. NPZ 导出内容（保留量纲）

所有数组为拟合时的浮点精度，不做 8 bit 压缩：

| 键 | 内容 |
|----|------|
| A, B, C, R, phi_rad, dolp | 拟合分量与派生量 |
| sigma_A, sigma_R, sigma_phi_rad, sigma_dolp | 不确定度 |
| residual_rms, reduced_chi2, max_abs_residual | 残差诊断 |
| condition_number, design_rank | 加权 Gram 矩阵条件数与设计秩（求解条件） |
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

## 7. 离线环境准备（无 root 的 Debian 容器）

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

## 8. 测试

```bash
python -m unittest discover -s tests
```

覆盖：设计矩阵正交性、分量往返、饱和/运动/缺失排除、真值参数核对、
φ 的 90° 歧义、弱响应状态、物理违例检测、多通道拟合、曝光归一、
配准符号与越界遮罩、NPZ 量纲保留、清单相对路径解析，以及受控故障
注入下的四类诊断对照、不可变输入（不重排/不夹紧）、逐点记录内容与
体积封顶、诊断包离线重建、区域窗口诊断与不可估计像素。
