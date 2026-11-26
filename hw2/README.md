# HW2 学生实验指南

本指南与 `hw2.pdf` 配套使用：PDF 给出题目与评分要求，本文件说明如何准备环境、补全代码、运行实验和整理提交材料。**所有必做实验均可在 CPU 上完成，不需要 GPU、外部数据或 AlphaFold 权重。** 数据来源与许可见 `NOTICE.md`。

## 1. 先看整体流程

作业的实验主线是：**Part Two 预测残基对距离 → Part Three 冻结 CNN，用 GNN 修正预测 → Part Four 在同一修正任务上比较优化器 → Part Five 解释哪些蛋白质得到改善。** Part One 的手算要求请直接阅读 PDF。

| 阶段 | 需要做什么 | 训练次数与预算 | 下一步使用的产物 |
|---|---|---|---|
| 准备 | 安装依赖，检查数据 | 不训练 | 可运行的环境 |
| Part Two | 补全 CNN 的 2 个方法，比较 `plain` / `dilated` | 2 次，各 300 步 | 选中的 **plain CNN** 预测缓存 |
| Part Three | 补全 GNN 的 3 个方法，比较 `sequence` / `predicted` 图 | 2 次，各 300 步 | 两种图的结果与分析 |
| Part Four | 补全 `Adam.step`，比较 SGD / Momentum / Adam | 3 次，各 200 步 | 优化器结果、逐蛋白验证 CSV |
| Part Five | 复用验证结果，提出并检验一个解释 | **不增加训练** | 一次分析及其代码 |

必做训练合计 **7 次、1,800 次参数更新**。无需搜索学习率、多随机种子重复训练，也无需为 Part Four 额外训练 CNN。先完成对应方法并通过诊断，再运行该单元的训练命令。

解压后应看到以下结构。后文所有命令均在 **`HW2_student/` 根目录**执行，即同时包含 `CNN/`、`GNN/` 和 `data/` 的目录；不要先切换到 `CNN/` 再运行命令。

```text
HW2_student/
├── hw2.pdf                  # 正式题面
├── README.md                # 本指南与实现规范
├── requirements.txt         # 环境依赖
├── config.json              # 必做实验的固定设置说明
├── check_data.py            # 数据完整性检查
├── CNN/model.py             # 补全 2 个方法
├── GNN/model.py             # 补全 3 个方法
├── Optimization/optimizers.py # 补全 1 个方法；SGD/Momentum 已提供
├── common/                  # 已完成的数据、特征、损失、训练和评估工具
├── data/                    # 数据、初始权重、采样计划、备用预测与诊断样例
└── reference_logs/task_optimizers/ # Part Four 使用的 CNN 训练/验证日志
```

## 2. 准备环境并检查数据

### 2.1 安装依赖

使用 **Python 3.11 或以上版本**；参考环境为 Python 3.13.5、PyTorch 2.9.1、macOS arm64。依赖版本以 `requirements.txt` 为准。推荐使用独立环境。

```sh
python -m venv .venv
```

激活环境，按操作系统选择一条：

```sh
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

然后安装依赖、检查数据并记录环境：

```sh
python -m pip install -r requirements.txt
python check_data.py
python -m pip freeze > environment.txt
```

`check_data.py` 成功时输出 JSON，其中 `passed` 为 `true`，训练/验证/测试链数为 **200 / 40 / 40**。数据已全部提供，不需要另行下载。保留输出中的 `dataset_id`，报告中也要记录该标识。若检查失败，先重新解压原始数据；不要修改 `checksums.json` 绕过检查。

### 2.2 统一运行设置

本指南的命令使用 `--device cpu`，默认使用 4 个 CPU 计算线程。如需降低 CPU 占用，可给诊断、训练或导出命令添加 `--threads 1` 至 `--threads 4`。统一实验使用 seed 416 和 float32；初始化、采样顺序、学习率、梯度裁剪、步数与选点逻辑均已提供，**无需修改训练循环或 `config.json`**。

## 3. 明确需要补全的代码

只需在以下 **3 个文件中补全 6 个函数体**，保持类名、参数和返回接口不变。`common/` 中的特征、损失、采样与评估已经完成；构造函数、`DistanceCNN.raw`、GNN 外层流程、SGD 和 Momentum 也已完成。

下表中的 `B` 为批大小，`H,W` 为窗口的两个空间尺寸，`L` 为残基数，`E` 为有向边数，`Q` 为查询残基对数。

| 对应题目 | 文件与方法 | 输入 → 返回 | 实现重点 |
|---|---|---|---|
| 2.1 | `CNN/model.py`：`ResidualBlock.forward(self, x)` | `[B,16,H,W]` → 同形状 | 按题面完成两次卷积、激活与残差相加；相加后还有 ReLU |
| 2.1 | `CNN/model.py`：`DistanceCNN.forward(self, x)` | `[B,20,H,W]` → `[B,16,H,W]` | 调用已提供的 `raw`，实现共享权重的空间转置平均；返回 logits |
| 3.1 | `GNN/model.py`：`build_edges(residue_index, prior_logits, mode='predicted', k=8)` | `[L]` 与 `[L,L,16]` → `long` 型 `[2,E]` | 支持两种图，按原始序号判断距离与打破并列；去重、去自环并补反向边 |
| 3.1 | `GNN/model.py`：`MessagePassingLayer.forward(self, h, edge_index, e)` | `[L,32]`、`[2,E]`、`[E,4]` → `[L,32]` | 生成消息，在接收节点处取均值，完成带残差的状态更新；处理空图与孤立节点 |
| 3.1 | `GNN/model.py`：`PairHead.forward(self, hi, hj, prior)` | `[Q,32]`、`[Q,32]`、`[Q,16]` → `[Q,16]` | 构造端点交换不变的表示，将 MLP 修正量加到 prior；返回 logits |
| 4.1 | `Optimization/optimizers.py`：`Adam.step(self)` | 读取各参数的 `.grad` → 原地更新参数，无需返回值 | 保留一、二阶矩与步数，做偏差修正；epsilon 位于平方根外 |

实现时注意：

- CNN 只交换最后两个空间轴，不能交换批或通道轴；两次调用共享同一组参数。不要在 CNN 或 GNN 的返回值上额外加 softmax。
- `build_edges` **用原始 `residue_index` 选边，但返回当前张量的节点行号**。例如节点行经过置换后，行号和原始序号不再相同。第 0 行为 sender，第 1 行为 receiver；空图仍返回 `[2,0]`，并保持正确的设备和整数类型。
- 图输入只使用原始序号与冻结 CNN 的预测；真实距离标签和坐标有效性掩码不能用于选边或构造输入。
- 消息在 receiver 处平均，不能把求和误当均值；孤立节点的平均消息为零，但其后续状态更新未必为零。PairHead 对 prior 只加一次修正量。
- Part Two/Three 使用的 Adam 已提供；Part Four 才考察自写 Adam。实现 `Adam.step` 时不能调用 `torch.optim`；诊断工具使用它作为对照不影响这一要求。

精确的特征、维度和更新约定见后面的 **Implementation Reference**，原理阅读见题面 References。补全前运行诊断出现 `NotImplementedError` 是预期现象。通过诊断后仍需完成题目要求的实验与报告，诊断本身不是全部答案。

## 4. Part Two：实现、检查并训练 CNN

### 第一步：完成两个 forward，通过公开诊断

编辑 `CNN/model.py` 后运行：

```sh
python CNN/diagnostics.py --check --device cpu
```

结果写入 `results/CNN/diagnostics.json`。应确认顶层 `passed` 为 `true`，各检查为 `PASS`。诊断包含输出/梯度对照、转置一致性和中心窗口与完整链的上下文一致性检查。

题目 2.1 的报告中需要记录：测试输入/输出形状、`transpose_error` 和 `central_context_error`。诊断 JSON 不单列形状，请从测试输入/输出张量自行记录；尤其要覆盖矩形与偏离对角线的窗口。误差用科学计数法保留原始量级，非零值至少保留 3 位有效数字；若原始值恰为零，可如实写 0。偏离对角线的单个输出窗口不要求自身对称。

### 第二步：各训练 300 步

```sh
python CNN/train.py --mode all --device cpu
```

该命令依次运行 `plain` 和 `dilated`，每次从相同的给定初值开始。每步使用四个 32×32 窗口，只监督中心 20×20 区域。完整链评估由程序完成，不需自行改写损失或添加窗口边界裁剪。

程序每 50 步在 **全部 40 条验证链**上计算 NLL，并保存验证 NLL 最低的候选检查点；即使较早的检查点最好，也会完成 300 步。保存的 `plain.pt` / `dilated.pt` 是所选检查点，不一定是第 300 步的权重。

### 第三步：整理结果，保留供 Part Three 使用的缓存

查看 `results/CNN/`：

- `curves.png`：两种 CNN 的训练损失与完整验证 NLL 曲线。
- `summary.csv`：两种 CNN 的选中步数、验证/测试指标与参数更新耗时。
- `summary.json`：更完整的记录，`baseline` 字段包含题目要求的 **sequence-separation baseline**。
- `plain_priors/cache.json` 及逐链 `.npy`：由选中的 **plain CNN** 自动导出，包含所有训练/验证/测试链的冻结预测。`dilated` 的预测不用于 Part Three。

完成题目 2.2 的曲线、三方指标对照（baseline、plain、dilated）与简短解释。再完成 2.3：复用 Part One 的参数量和感受野计算，统计单个 32×32 窗口的 raw CNN MACs 与转置封装两次调用的总 MACs；可写小脚本辅助统计。**不要求 CNN 一定优于基线。**

## 5. Part Three：冻结 CNN，训练 GNN 修正

### 第一步：完成三个方法，通过公开诊断

编辑 `GNN/model.py` 后运行：

```sh
python GNN/diagnostics.py --check --device cpu
```

检查 `results/GNN/diagnostics.json`。公开样例覆盖选边、消息与梯度、pair head 对称性、重建图后的置换一致性等。此诊断使用固定样例，**不需要先完成或训练 CNN**。

### 第二步：使用 Part Two 的同一份 CNN 缓存，训练两种图

正常流程下，先确认 `results/CNN/plain_priors/cache.json` 已存在，再运行：

```sh
python GNN/train.py --mode all --device cpu
```

程序读取上述缓存，分别训练 `sequence` 和 `predicted`，各 300 步。两者都使用同一个冻结 CNN、相同初值和查询计划；比较的是是否增加预测接触边。CNN 不再更新。图、边特征、损失、检查点选择与完整链评估均由现成工具串联。

若 CNN 使用了自定义输出目录，或尚未完成，请先阅读第 8 节的缓存选项。不要把不同 CNN 的预测用于两个 GNN 配置。

### 第三步：整理对照，并补做训练后检查

查看 `results/GNN/summary.json` 中的 `prior_source`、`baseline` 和 `cnn_baseline_check`：

- `prior_source` 说明当前 CNN 来源，包括检查点哈希与选中步数。报告中应明确来源，并提交对应的 `cache.json`。
- `baseline` 是**这份输入缓存对应的 CNN**；应与其两种 GNN 修正结果一起比较，不能换成 Part Two 的 dilated CNN。
- 自己导出的 CNN 缓存会与源 CNN 记录核对指标；备用缓存则核对来源及预测哈希，并在本次运行中计算基线。两种检查的含义不同，按实际输出说明即可。

使用 `curves.png` 和汇总文件，报告 CNN、CNN + sequence GNN、CNN + predicted GNN 的验证/测试 NLL 与 Precision@L，并解释差异。**不要求 GNN 一定改善预测，也不要求 predicted 图一定优于 sequence 图。**

题目 3.1/3.2 还要求以下内容，公开诊断没有替你全部完成：

1. **实际图规模。** 用实际链构建两种图，记录节点数和边数，说明统计的是有向边还是无向边，以及所用链或汇总方式。不要将 top-8 误解为最终度数上限。
2. **训练后模型的置换检查。** 加载一个已训练并选中的 GNN 检查点，在同一条链上比较置换前后的同一组查询对。将节点特征、原始序号和 prior 两个残基轴同步置换，重新构图；用逆置换重映射查询行号，再比较 logits 的最大绝对差。另包含一个 prior 接触分数并列的情形，检查按原始序号打破并列是否正确。此检查只需推理，不增加训练。

## 6. Part Four：补全 Adam，在相同 GNN 任务上比较优化器

### 第一步：完成 Adam 并通过诊断

编辑 `Optimization/optimizers.py` 的 `Adam.step` 后运行：

```sh
python Optimization/diagnostics.py --check --device cpu
```

查看 `results/Optimization/diagnostics.json`。报告 Adam 的 pass/fail 与 `max_parameter_error`；诊断使用多步、多参数张量轨迹检查状态与偏差修正。该诊断中的小样例可使用与正式训练不同的 betas/epsilon，应使用构造函数传入的值，不要写死训练常数。

### 第二步：每种优化器训练 200 步

```sh
python Optimization/run.py --mode all --device cpu
```

本单元使用**已提供的完整 predicted-edge GNN 与图构造器**，不依赖你补全的 `GNN/model.py`。每个优化器从同一零修正初值重新训练，**不接着 Part Three 的 300 步检查点训练**；CNN 缓存与 Part Three 保持一致。

| 优化器 | 学习率 | 其他设置 |
|---|---:|---|
| SGD | 0.05 | 已提供 |
| Momentum | 0.01 | 已提供，momentum 0.9 |
| Adam | 0.0001 | 需补全，betas `(0.9, 0.999)`，epsilon `1e-8` |

程序固定为每次 200 步，按完整验证 NLL 各自选点。`results/Optimization/curves.png`、`summary.csv` 和 `summary.json` 提供曲线、指标和耗时；`cnn_baseline_validation.csv` 与各优化器的 `*_validation.csv` 供 Part Five 使用。

### 第三步：与已提供的 CNN 优化器日志比较

阅读 `reference_logs/task_optimizers/README.md`、`summary.csv`、各优化器 JSON 及 `*_history.csv`，比较 CNN 与 GNN 上的**验证 NLL 排序**。这些是已有的 200 步 CNN 训练/验证记录，**不需要重新训练三次 CNN**。日志中的 CNN Adam 学习率为 `0.001`，本单元 GNN Adam 为 `0.0001`；比较结论应限定在给定设置与预算下。

报告提交损失/验证曲线和指标汇总（含 `update_seconds`），再用不超过 **200 英文词或 350 中文字**讨论跨任务比较并给出一个局限。`update_seconds` 仅统计预热后的参数更新循环，不能当作整个程序的运行时间。

## 7. Part Five：用现有验证结果完成一次分析

本单元不要求补全新的模型函数，也不增加训练。写一个可运行的分析脚本（例如 `analysis/part5_analysis.py`），可使用标准库 `csv` / `json` 和已安装的 NumPy、Matplotlib。

1. **选一个问题。** 结合前面结果提出可检验的假设，例如链长与修正收益是否有关，或不同优化器是否改善同一批蛋白质。按题面完成一次调查即可。
2. **读验证数据。** 使用 `results/Optimization/cnn_baseline_validation.csv`，以及至少两个优化器的 `sgd_validation.csv`、`momentum_validation.csv`、`adam_validation.csv`；三者齐全时使用全部三个。CSV 含 `chain_id`、`length`、`nll`、`precision_at_l`、`valid_pairs` 与 `long_range_candidates`。
3. **核对可比性。** 从对应优化器 JSON 核对 `prior_source`、`updates` 与 `selected_step`。必须使用同一 CNN 来源、相同 200 步预算和选点规则；允许不同优化器选中不同步数。按 `chain_id` 连接，不能只按 CSV 行号相减。
4. **计算与展示。** 对每条链计算 `GNN NLL - CNN NLL`，负数表示改善。围绕假设做一次逐链或分组分析，用**至多一张图或一张表**给出证据。
5. **解释结果。** 说明假设是否得到支持，联系 CNN–GNN–优化器主线，并给出局限。假设被否定也可以获得完整评价；不要用测试集挑假设。

正文不超过 **300 英文词或 500 中文字**，附可运行分析代码与 AI Use Statement。没有必要额外重训或调参。若 Adam 尚未完成，可按下一节分别运行已经提供的 SGD 与 Momentum，支持这一分析。

## 8. 缓存路径、独立完成与重跑方式

### 8.1 CNN 输出目录改变后，如何衔接后续实验

默认流程不需要路径参数。若确需更改保存位置，可在**第一次正式运行时**使用以下命令（它们是默认命令的替代方案，不是一组额外必做实验）：

```sh
python CNN/train.py --mode all --device cpu --output results/my_CNN
python GNN/train.py --mode all --device cpu --prior-dir results/my_CNN/plain_priors
python Optimization/run.py --mode all --device cpu --prior-dir results/my_CNN/plain_priors
```

`--output` 控制本单元结果目录；`--prior-dir` 控制读取哪一份冻结 CNN 缓存，两者含义不同。改变 CNN 输出目录不会自动改变后两个命令的默认缓存路径。

若已有选中的 `plain.pt`，只需重建缓存，可运行下列导出工具；同时保留与该权重匹配的 `plain.json`，放在权重同一目录：

```sh
python CNN/export_priors.py --checkpoint results/CNN/plain.pt --output results/CNN/plain_priors --device cpu
```

导出只增加推理，不重新训练。缓存检查失败时，应核对来源或重新导出，不要手动修改预测数组或 `cache.json`。

### 8.2 CNN 尚未完成时，如何独立进行后续实验

Part Three/Four 可以显式使用 `data/priors/` 中的备用 plain CNN 预测：

```sh
python GNN/train.py --mode all --device cpu --use-reference-prior
python Optimization/run.py --mode all --device cpu --use-reference-prior
```

第一条仍要求完成 GNN 的三个方法，第二条只要求完成 Adam。报告中须注明使用了备用 CNN，所有后续比较使用该来源对应的基线。备用入口不能代替 Part Two 本身的作答。若后来改用自己的 CNN，应确保报告中的各组比较来自同一份缓存，不要混用已有结果。

若 Adam 尚未完成，分别运行 SGD 和 Momentum 即可生成各自结果，并保留 Part Five 所需的基线 CSV：

```sh
python Optimization/run.py --mode sgd --device cpu --use-reference-prior
python Optimization/run.py --mode momentum --device cpu --use-reference-prior
```

若已有自己的缓存，移除上面的 `--use-reference-prior`，或统一换为同一个 `--prior-dir`。`--prior-dir` 与 `--use-reference-prior` 不能同时使用。不要在 Adam 仍未完成时运行 `--mode all`：它会在 Adam 处中断，本次最终汇总与基线 CSV 可能尚未写出。

### 8.3 输出文件与覆盖规则

默认输出为 `results/CNN/`、`results/GNN/`、`results/Optimization/`。`<name>` 对应 `plain/dilated`、`sequence/predicted` 或 `sgd/momentum/adam`。

| 文件 | 用途 |
|---|---|
| `diagnostics.json` | 本单元公开诊断的逐项结果与误差 |
| `<name>.json` | 该配置的来源、环境、步数、选中点、验证/测试指标、耗时及历史 |
| `<name>_history.csv` | 每步训练损失；每 50 步有完整验证 NLL，其余行该列为空 |
| `<name>_validation.csv` / `<name>_test.csv` | 选中检查点的逐蛋白指标 |
| `<name>.pt` | 选中检查点，用于本地推理检查；提交时不打包 |
| `summary.json` | 本次调用各配置的汇总、基线及来源检查 |
| `summary.csv` | 本次调用的简洁指标表；基线另见 `summary.json` |
| `curves.png` | 本次调用各配置的训练/验证曲线 |
| `plain_priors/cache.json` | 仅 CNN 输出：供后续单元使用的缓存说明，随报告提交 |
| `cnn_baseline_validation.csv` | 仅 GNN/Optimization 输出：同源 CNN 的逐蛋白验证基线 |

同一配置在同一目录重跑会覆盖其结果。**每次调用都会重写 `summary.json`、`summary.csv` 和 `curves.png`，且只包含本次调用的配置。** 单独运行 Momentum 后，已有 SGD 的独立 JSON/CSV 仍在，但当前汇总图不包含 SGD。

正常流程建议各单元只运行一次 `--mode all`。若已分别完成各配置，可从各自的 JSON/历史 CSV 合并作图和汇总，不必只为重新生成总图而重复训练。需要并存不同尝试时使用不同的 `--output` 目录，并清楚记录缓存来源。

## 9. 整理报告与提交

报告按题号作答，检查以下内容是否齐全：

| 部分 | 报告与材料检查项 |
|---|---|
| Part One | PDF 题面的三个 CNN 手算问题，独立手写推导的照片或扫描件 |
| Part Two | 两个方法；诊断与形状；两模型的训练/验证曲线、两模型与基线的指标；参数量、感受野与 MACs |
| Part Three | 三个方法；诊断与训练后置换检查；图规模；同源 CNN 与两个 GNN 的对照；来源 `cache.json` |
| Part Four | Adam；诊断；三优化器的曲线、指标与更新耗时；和现有 CNN 日志的短讨论 |
| Part Five | 一条假设、一次验证分析、至多一张图或一张表、结论与局限；分析代码 |
| 全文 | 环境、CPU 线程数、seed、各配置步数、数据标识；末尾 AI Use Statement |

将提交文件整理为 `{StudentID}_{Name}_hw2.zip`，例如：

```text
{StudentID}_{Name}_hw2/
├── report.pdf
├── CNN/model.py
├── GNN/model.py
├── Optimization/optimizers.py
├── analysis/                 # Part Five 与训练后检查等可运行分析代码
├── results/
│   ├── CNN/                  # 相关 JSON、CSV、图片及 plain_priors/cache.json
│   ├── GNN/                  # 相关 JSON、CSV、图片
│   └── Optimization/         # 相关 JSON、CSV、图片
└── environment.txt
```

只提交报告所需的精简结果与代码；**不打包**原始 `data/`、`.venv/`、CNN 预测 `.npy` 数组或完整 `.pt` 检查点。自定义路径和分析脚本的运行方式请在报告或一个简短说明中写清楚。实验失败也应如实记录现象和已有证据。

手算与核心代码须遵守题面规定：手算独立完成，AI 不得生成手写推导或未标记任务的核心代码；Part Five 允许 AI 辅助分析。在报告末尾用 3–5 句话给出 AI Use Statement，注明工具、涉及任务与核验方式；未使用也须说明。

以下两节集中保留数据、评估和模型的精确定义，供实现或排查结果时查阅，**不增加额外题目或手算要求**。


## Shared Experimental Setup

The following definitions fix the shared protocol. Data loaders and evaluation are provided; use the step-by-step guide above for the required commands.

### Data and targets

The staff supplies **200/40/40 train/validation/test complete short chains**, each 48-96 residues, derived from ProteinNet CASP7 `training_50` ([ProteinNet documentation](https://github.com/aqlaboratory/proteinnet)). CATH 4.3.0 homologous superfamilies, shared PDB IDs, and identical sequences define connected groups before splitting (88/22/11 groups). This is a course split of original training records, not the official CASP7 test benchmark. `data/manifest.json` fixes IDs, source hashes, groups, units, and preprocessing. No chain or recorded group crosses splits; see `NOTICE.md` for sources and limitations.

Keep residues with missing target coordinates as input nodes. Before permutation, `residue_index` is the unique, zero-based sequence position $0,\ldots,L-1$; it is not a PDB author residue number. Here $L$ is the full sequence length, including unresolved residues.

The input contains no target coordinates. The dataset and prediction cache use the following fields:

| Field | Shape | Meaning |
|---|---|---|
| `node_features` | $(L,21)$ | Staff-prepared nonnegative normalized descriptor, as defined below |
| `residue_index` | $(L,)$ | Original positions along the chain |
| `target_bin` | $(L,L)$ | Integer distance-bin labels, used only in losses and evaluation |
| `coord_mask` | $(L,)$ | Whether the target Cα position is resolved; used only to mask supervision and evaluation |
| `prior_logits` | $(L,L,16)$ | Symmetric, frozen log probabilities from the selected Part Two plain CNN |

In `common.data.load_chains`, each returned chain dictionary uses the aliases `a`, `idx`, `y`, `mask` and `prior` for these five fields, respectively, and `id` for the chain identifier. `prior` is loaded only with `with_prior=True`; `features` is added only with `with_pairs=True`. `queries` contains valid upper-triangle supervised pairs for losses and evaluation. Use this mapping when writing your trained-checkpoint or graph-statistics analysis; target labels and masks remain excluded from model inputs.

The raw ProteinNet field contains **20 PSSM scores plus information content** ([record format](https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md)). The provided preprocessing drops information content, normalizes each nonnegative positive-sum score vector, and appends a zero as dimension 21. Missing or zero-sum rows use the query residue's 21-class one-hot, including an unknown class. Thus $a_i\ge0$ and $\sum_k a_{ik}=1$. This is a normalized descriptor, not an observed MSA frequency.

Distances use **Cα atoms and ångströms** throughout, including glycine. The staff extracts the Cα in each ProteinNet N-Cα-C triplet and divides raw picometers by 100. Coordinate validity is checked before labels are generated; unresolved pairs have label -1. This differs from AF1's Cβ-based definition. The 15 finite bin boundaries are

$$
(4,5,6,7,8,9,10,11,12,13,14,15,16,18,20)\ \text{Å}.
$$

They define 16 bins: below 4 Å; twelve width-one bins from 4 to 16 Å; two width-two bins from 16 to 20 Å; and at least 20 Å. Finite intervals are left-closed and right-open. Consequently, contact probability for $d<8$ Å is the sum of probabilities in bins **0–4**.

### Frozen CNN prediction cache

A cache stores float32 symmetric distance log probabilities for every pair in all 280 chains. `cache.json` records the source CNN checkpoint hash, selected step, data manifest and per-file hashes. Export reads node features and original sequence positions; target coordinates, labels and masks do not determine inputs or which predictions are written. The loader rejects incomplete, modified or mismatched caches.

Both graph configurations use the same cache. Normal own-CNN exports also include source metrics for a consistency check. The supplied fallback contains predictions and provenance without precomputed source metrics; its CNN baseline is evaluated in your run. Source-checkpoint paths in the data manifest describe provenance, not additional files you need to download.

The CNN and GNN train on the same 200 chains in this two-stage pipeline. Training-chain CNN predictions are **in-sample**, not out-of-fold, which can cause fitting bias; assess generalization on held-out chains. No extra cross-validation training is required. The course GNN is inspired by residue relationship reasoning and does not reproduce AlphaFold 2's architecture or recycling.

`target_bin` and `coord_mask` may determine supervision, supervised-query sampling and evaluation; they must not construct model inputs, edges or edge features.

### Training budget and evaluation

Use seed **416**. CNN uses provided Adam with learning rate $10^{-3}$; GNN uses provided Adam with learning rate $10^{-4}$. Both use betas $(0.9,0.999)$, $\epsilon=10^{-8}$, no decay or schedule. Clip the norm of the **complete mini-batch gradient across all parameters** to 1, once after backward and before the update. Every compared run reloads the same supplied initial `state_dict`; the CNN stays frozen during GNN training; fixed window/chain/query lists use separate random streams. Setting one seed and constructing models consecutively is not a substitute for copying the initial state.

**Initialization is provided.** CNN head weights are zero and biases are the log training-class frequencies, averaged equally over training chains using valid upper-triangle pairs with sequence separation at least 6. Plain/dilated CNNs share these values and the remaining seed-416 weights; the supplied CNN optimizer reference logs use the same initialization. The GNNs in Parts Three and Four share a zero-initialized final correction layer, so their initial output equals the frozen CNN prior; their other weights use the same supplied random state. No initialization code is required.

| Required experiment | Configuration | Updates per run | Runs |
|---|---|---:|---:|
| CNN | `plain`, `dilated` | 300 | 2 |
| GNN | `sequence`, `predicted` | 300 | 2 |
| Optimization | Same predicted-edge GNN; `sgd`, `momentum`, `adam` | 200 | 3 |
| **Total** | **Fixed configurations** | **1,800** | **7** |

Use **float32 and at most four CPU compute threads**, with one interop thread. Small numerical checks use float64. Features, priors, graphs, and sample lists are prepared outside timed update loops; evaluation handles one chain at a time. Multi-seed reruns and learning-rate searches are not required.

Every 50 updates, evaluate **all 40 complete validation chains** using the mean per-chain NLL defined below, including all valid upper-triangle pairs and sequence ends. Eligible checkpoints are **50,100,...,T only; step 0 is not eligible**. Select minimum full-validation NLL, earliest step on an exact tie. Validation uses `eval` and no gradients; no contact ranking is needed during selection. Finish all prescribed updates even if an earlier checkpoint is selected. At the selected checkpoint, report full validation metrics and evaluate test metrics once. Training window/query losses and the full-validation NLL have different reductions; keep them clearly labelled.

Use validation data or the provided **training/validation logs** for exploratory choices. Test scores must not select configurations, checkpoints, graphs, or investigations.

For complete-chain evaluation, write $m_i=\texttt{coord\_mask}_i$ and let $q$ denote the softmax probabilities of the model being evaluated, rather than necessarily the frozen prior. Let

$$
\mathcal E_n=\{(i,j):i<j,\ \Delta_{ij}\ge6,\ m_i=m_j=1\}.
$$

The main distance metric is the **mean per-chain negative log-likelihood**:

$$
\operatorname{NLL}=\frac1N\sum_{n=1}^{N}
\frac1{|\mathcal E_n|}\sum_{(i,j)\in\mathcal E_n}
-\log q_{ij,y_{ij}}.
$$

Here $N$ counts chains with nonempty $\mathcal E_n$; skip empty sets and report the number scored. Do not pool all pairs across proteins. For long-range contact Precision@L, restrict to $\Delta_{ij}\ge24$ and resolved pairs in the upper triangle. Rank by contact probability and take $\min(L,\text{number of candidates})$ pairs, where $L$ is the chain's full residue count. Break score ties by the original residue-index pair. Average precision across eligible chains; skip a chain with no candidates and report the number scored. A supplied sequence-separation prior is the Part Two baseline. Parts Three and Four use their exact input CNN as the baseline. Do not report coordinate RMSD or model-predicted confidence as if either were a measured structure error for these distance-only models.

## Implementation Reference

本节供实现时查阅，集中记录课程模型的精确定义，不要求额外手算或推导。题面各单元的示意图与任务清单说明需要完成什么；已有 helper 负责的特征和损失不需重写。理论阅读见题面 References：CNN [1,8,9]、消息传递 [10]、Momentum [11]、Adam [6] 的 Algorithm 1。原论文帮助理解原理；课程具体结构、采样和评估以题面及本文件为准。

### CNN: features, transpose wrapper, and window loss

The provided feature builder uses a fixed projection $s_i=a_iR\in\mathbb R^8$ of the 21-dimensional profile $a_i$. $R$ is supplied and is not trained. It constructs 20 symmetric channels:

$$
F_{ij}=\operatorname{concat}\bigl(s_i+s_j,\ |s_i-s_j|,\ r_{ij},\ c_{ij}\bigr).
$$

Here $\Delta_{ij}=|\texttt{residue\_index}_i-\texttt{residue\_index}_j|$,

$$
r_{ij}=\left(\frac{\min(\Delta_{ij},32)}{32},\frac{\log(1+\Delta_{ij})}{\log 97}\right),
\qquad
c_{ij}=\left(a_i^\top a_j,\frac{H(a_i)+H(a_j)}{2\log21}\right),
$$

with $H(a)=-\sum_k a_k\log a_k$ and $0\log0=0$. These are inexpensive profile descriptors, **not pairwise evolutionary couplings inferred from a full MSA**. Their construction is provided and is not assessed.

Let $f_\theta$ be the CNN in Task 1.1, with ReLU after its input projection. Use spatial transposition only, leaving batch and channel axes unchanged, and return

$$
\ell_\theta(F)=\tfrac12\bigl(f_\theta(F)+f_\theta(F^\top)^\top\bigr).
$$

The same weights are reused in both calls. For any rectangular or off-diagonal window, test $\ell(F^\top)^\top=\ell(F)$; a single off-diagonal output window need not be symmetric by itself. Full-chain inputs are symmetric, so their outputs are symmetric. Required MAC counts use both calls, even if a separate full-chain inference optimization can reuse one call.

Training uses four $32\times32$ windows per update. Each axis is a contiguous 32-residue fragment within a full short chain; the axes may be separated, allowing long-range pairs. **Both CNNs supervise only the central $20\times20$ area**, local rows/columns 6 through 25. The common six-position guard band removes artificial window-boundary context from supervised outputs without increasing input size. Only resolved pairs with separation at least six are valid; no upper-triangle restriction is imposed on training windows.

For each window, the staff sampler chooses an eligible training chain uniformly, then resamples window offsets within that chain until its central area has a valid pair. Its fixed window list is shared across compared runs. Loss is the **mean over valid central pairs in each window, then the mean over four windows**. Select valid entries before cross-entropy; -1 labels never enter it. The supplied CNN optimizer reference logs use this same training-loss definition. Checkpoint selection uses the complete-chain validation NLL, rather than a center-window loss. Full-chain final evaluation uses all valid upper-triangle pairs, including sequence ends, without gradients; no guard band is applied there.

### GNN: graph, message passing, and pair correction

The graph has one node per residue, initial hidden states $h_i^0=\operatorname{ReLU}(a_iW_{\mathrm{in}}+b_{\mathrm{in}})\in\mathbb R^{32}$ with $W_{\mathrm{in}}\in\mathbb R^{21\times32}$, and one of two edge sets:

- **`sequence`:** connect positions whose original sequence indices differ by one.
- **`predicted`:** start with sequence edges; for each node choose $\min(8,\text{eligible candidates})$ highest-prior-contact neighbors at separation at least six. Break ties by original sequence index. Take the union with reverse edges, remove duplicate directed pairs and self-edges. With no eligible neighbors retain sequence edges.

Store `edge_index` as two rows: sender, then receiver. **The final degree may exceed 10**, because other nodes can select the same residue. Both graph configurations use the same prior in edge features and the pair head; the comparison isolates added predicted edges, not the presence of a prior.

The graph builder and edge features use only sequence indices and the frozen prior. Write $q_{ij}=\operatorname{Softmax}(\ell^0_{ij})$ and $p_{ij}=\sum_{b=0}^4q_{ijb}$. Four symmetric edge features are provided:

$$
e_{ij}=\left(p_{ij},\frac{H(q_{ij})}{\log16},
\frac{\min(\Delta_{ij},32)}{32},\mathbf1[\Delta_{ij}=1]\right).
$$

For each of two layers, a message MLP maps 68 to 32 to 32 dimensions, using ReLU between its linear layers. With directed edge $j\to i$,

$$
m_{j\to i}^{t}=M_t([h_j^t,h_i^t,e_{ij}]),\qquad
\bar m_i^t=\frac{\sum_{j\to i}m_{j\to i}^{t}}{\max(1,\deg(i))},
$$

$$
h_i^{t+1}=h_i^t+\operatorname{ReLU}\bigl([h_i^t,\bar m_i^t]W_{U,t}+b_{U,t}\bigr).
$$

The aggregate of an isolated node is the zero vector. The update map has $W_{U,t}\in\mathbb R^{64\times32}$ and a 32-dimensional bias. All linear layers, including the message and pair MLPs, have biases. The two message-passing layers have independent parameters. A symmetric pair head, an MLP of dimensions $80\to32\to16$ with an intermediate ReLU, returns

$$
\ell_{ij}=\ell^0_{ij}+\operatorname{PairHead}
\bigl([h_i^2+h_j^2,\ |h_i^2-h_j^2|,\ \ell^0_{ij}]\bigr).
$$

The staff supplies a common initial `state_dict`, the prior, and feature construction; **students implement edge selection and the assessed forward methods**. Each update chooses an eligible training protein uniformly, then up to 128 valid unordered pairs uniformly without replacement. Loss is the mean cross-entropy over those queries. Graph selection and prior are frozen. Final evaluation decodes at most 128 query pairs at a time. The prior is the **frozen prediction of your selected Part Two plain CNN**, or the explicitly selected staff fallback. Compare refinement with that exact CNN baseline. The provided baseline check compares cache metrics with the source CNN record; minor floating-point differences can occur.

### Optimizers: exact update conventions

Part Four trains the predicted-edge GNN from Part Three, with its CNN input frozen. It supplies a complete GNN and graph builder as TorchScript artifacts, so the optimizer task does not import your CNN or GNN implementation. All three methods use the same zero-correction initial state and the first 200 entries of the GNN query plan. SGD and Momentum are provided; only Adam.step is assessed. SGD uses learning rate 0.05; Momentum uses learning rate 0.01 and momentum 0.9; Adam uses learning rate 0.0001, betas (0.9, 0.999), and epsilon 1e-8. There is no weight decay, dampening, Nesterov term, or schedule. The supplied loop clips the complete mini-batch gradient once before the update. Every assessed parameter receives a gradient tensor at every update; a shared Adam counter or consistent per-parameter counters are both acceptable.

Initialize $v_0=m_0=u_0=0$ and the update counter to zero. Use the following updates; squares, square roots, and divisions are elementwise:

$$
\begin{aligned}
\text{SGD:}\quad &\theta_t=\theta_{t-1}-\eta g_t;\\
\text{Momentum:}\quad &v_t=\mu v_{t-1}+g_t,\qquad
\theta_t=\theta_{t-1}-\eta v_t;\\
\text{Adam:}\quad &m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
&u_t=\beta_2u_{t-1}+(1-\beta_2)g_t^2,\\
&\widehat m_t=m_t/(1-\beta_1^t),\qquad
\widehat u_t=u_t/(1-\beta_2^t),\\
&\theta_t=\theta_{t-1}-\eta\widehat m_t/(\sqrt{\widehat u_t}+\epsilon).
\end{aligned}
$$

Momentum uses the current gradient without an extra factor $(1-\mu)$. These formulas are implementation references; no extra hand calculations are required. Small float64 diagnostic fixtures may specify their own betas and epsilon; required GNN training settings are fixed above and in Part Four. The provided CNN reference logs use Adam learning rate 0.001, with SGD/Momentum rates unchanged.


