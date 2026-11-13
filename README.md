# 多角度线性偏振照片分量分析器

对同一静态场景在多个检偏器（分析角）下拍摄的强度照片做配准、共同曝光
解释后，按声明的成像模型逐像素拟合随角度变化的分量与恒定分量，给出拟合
残差、可靠区域、分量图与单像素观测曲线。

## 这是什么，不是什么

**它做的是真实的最小二乘拟合**：每个像素用有效角度样本求解
`I(θ) = A + B·cos2θ + C·sin2θ`，输出参数、不确定度、残差与可靠性状态，
并可由参数重建任意角度的预测强度。它**不是**把多张照片取最暗值然后宣称
完成了物理分离。

**它不承诺做不到的事**：

- 恒定分量 **A 不命名为"完全无反光的真实物体颜色"**。A 包含所有在旋转
  线偏振片下不可分辨的贡献，未偏振（或圆偏振）的镜面反射与漫反射一样
  落入 A，本模型无法把二者区分开。
- 本工具**不承诺去除不可区分的未偏振反射**。能估计的只是线偏振部分
  （R、φ、DoLP）及其置信度。
- 弱响应区（偏振幅度与噪声同量级）给出可解释的 `weak_response` 状态，
  偏振角与 DoLP 标记为不可解释，而不是强行输出一个数字。

## 图形入口（偏振估计的主界面）

文档以图形界面为主，命令行仅辅助批处理。

```bash
# 方式一：启动脚本（自动处理无 root 环境下的本地 Qt 库路径）
./scripts/run_gui.sh

# 方式二：模块入口
python -m polanalyzer.gui.app

# 方式三：安装后
pol-gui

# 无显示服务器时的离屏逻辑自检（人工分析仍需真实显示）
QT_QPA_PLATFORM=offscreen ./scripts/run_gui.sh
```

首次启动需先安装依赖 `python -m pip install numpy matplotlib pillow
PySide6`；无 root Debian 容器的 Qt 库与中文字体离线准备见
[`docs/usage.md`](docs/usage.md) 第 7 节。

界面操作：

1. 左上角 **分析角下拉框** 选择要查看的角度帧（图像恢复窗口）；
2. **点击图像任意像素**，左下区域显示该像素的观测曲线：
   - 绿点：进入方程的有效观测；
   - 红叉：被排除样本，图例与信息栏给出原因
     （饱和 / 运动 / 缺失 / 非法）；
   - 蓝线：由拟合参数计算的模型曲线，浅蓝带为 R 的不确定带；
3. 右侧为实际运算得到的分量图：A、R、偏振角（HSV 编码，非可靠像素
   消色）、DoLP、残差 RMS、归一化卡方、可靠性状态分类；
4. 信息栏列出该像素的参数与不确定度、有效角度数、残差、量纲；
5. “导出 NPZ”保留浮点量纲、全部不确定度、遮罩与预测强度。

任何异常在界面底部消息条提示，不静默吞掉。

### 异常诊断与窗口联动

“受控场景”下拉框提供正常、仅角度错配、仅曝光不一致、角度+曝光、
局部运动五个可复现对照。点击“运行异常诊断”后（后台线程，运行状态条
可见）：

- 图像恢复窗口用红色散点标出异常像素、黄色包围盒定位异常区域；
- 分量图新增“诊断假设”行，区分分析角顺序错配、照片曝光不一致、
  局部场景运动、二者共同存在、证据不足（保留候选）与不可估计；
- 点击像素时观测曲线标注热点角度，信息栏给出假设、候选解释、设计秩/
  条件数、证据列表与建议。

诊断只在副本上重拟合，**不自动重排照片、不夹紧负值**；“提取诊断包”
导出 `manifest.json + data.npz + rebuild.py`，可脱离本界面离线重建
同一结论。方法细节见 [`docs/methodology.md`](docs/methodology.md) 第 8 节。

## 命令行（辅助）

```bash
# 合成观测 + 已知噪声/遮挡/运动，拟合并与真值核对
python -m polanalyzer.cli synthetic --out fit.npz --report report.json

# 真实角度图像：按 JSON 清单载入、曝光归一、可选配准后拟合
python -m polanalyzer.cli fit manifest.json --register --out fit.npz

# 读取导出结果打印状态报告
python -m polanalyzer.cli report fit.npz

# 受控故障对照（只有角度 / 只有曝光 / 二者联合 / 局部运动），
# 自动核对诊断结论与注入故障是否一致，可顺带导出诊断包
python -m polanalyzer.cli fault-demo --bundle-dir examples/bundles

# 对清单运行可复跑诊断并提取自包含诊断包（不修改输入）
python -m polanalyzer.cli diagnose manifest.json --read-noise 8 \
    --region 20,40,25,45 --bundle-dir out_bundles
python3 out_bundles/diagnostic_bundle/rebuild.py   # 离线重建同一结论
```

清单 `manifest.json` 格式见 `docs/usage.md`。仓库内置示例（PNG 帧、
清单与已复算的诊断包）在 `examples/`，由
`examples/generate_examples.py` 确定性生成。

## 成像模型与可靠性

数学模型、参数含义、排除规则与可靠性状态见
[`docs/methodology.md`](docs/methodology.md)。

## 测试

```bash
python -m unittest discover -s tests
```

测试用已知真值的合成观测核对估计参数与预测强度，显式覆盖饱和、运动、
缺失角度的排除以及弱响应状态，不允许用拟合结果自证。诊断测试
（`tests/test_diagnostics.py`）另覆盖四类受控故障的归因对照、输入
不可变（不重排/不夹紧）、逐点记录与体积封顶、诊断包离线重建及正常
观测经诊断路径后分量逐元素不变。

## 环境备注

无 root 的 Debian 容器中，Qt 运行库（libEGL 等）与中文字体可安装到
`~/.local` 下，`scripts/run_gui.sh` 会自动识别。细节见
[`docs/usage.md`](docs/usage.md) 的“离线环境准备”。
