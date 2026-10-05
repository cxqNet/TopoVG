# TopoVG

基于语言解析树和 Stable Diffusion 注意力的遥感目标定位实现。提供三个入口：`llm_parser.py` 补全解析树、`generate_features.py` 提取特征、`evaluate.py` 从已保存的特征执行定位与评测。

一次运行处理指定划分的全部样本。生成程序逐条执行一次前向；评测支持多进程，默认 4 个 worker。当前注意力捕获器的模型 batch 为同一个样本的 `[uncond, cond]`，不支持把多个不同样本合并成一个模型前向。

## 目录与模块

从项目根目录运行下文命令。入口脚本和 `topovg/` 包必须一起保留。

| 文件或目录 | 职责 |
|---|---|
| `llm_parser.py` | 解析入口，保留原有增量补全流程 |
| `generate_features.py` | 整个划分的单次特征提取入口 |
| `evaluate.py` | 已有特征的独立评测入口 |
| `build_rrsis_test_from_refs.py` | 从原始 test refs 和 XML 构建标准测试 JSON |
| `topovg/config.py` | 模型路径默认值、图像开关与既有解码常量 |
| `topovg/parser/prompts.py` | 原始 Qwen system prompt 与 few-shot 示例 |
| `topovg/parser/` | Qwen 推理、JSON 验证、缓存匹配和补全 |
| `topovg/data/` | 划分加载、图像查找、共享特征路径 |
| `topovg/diffusion/pipeline.py` | SD1.5 加载、编码、加噪与一次 U-Net 前向 |
| `topovg/diffusion/ptp_utils.py` | 当前生效的注意力捕获和聚合 |
| `topovg/diffusion/generation.py` | 样本遍历、已有文件跳过、保存与资源释放 |
| `topovg/grounding/semantic_executor.py` | 原有解析树到注意力图的语义执行 |
| `topovg/decoding/tfbd.py` | 原有跨尺度融合、稀疏场细化、亲和力定位 |
| `topovg/evaluation/` | 指标、并行 worker、可视化和 TXT 报告 |
| `topovg/utils/` | 原子保存、文件锁、特征格式检查和延迟依赖加载 |
| `tests/` | 注意力、生成流程、parser、输出和并行回归检查 |

数据布局如下：

```text
Data/RRSIS-D/
├── JPEGImages/
│   ├── 03600.jpg
│   └── ...
├── split/
│   └── test.json
└── features/
    ├── 03600_000.pt
    └── ...
```

原始 `refs(unc).p` 中的 RRSIS-D test 应得到 **3481 条样本、1824 条去重表达**，每个引用里的每条 sentence 独立保留。不能把 test 图像对应 XML 中所有 object 都扩展进测试集，也不能按文本去重决定 `.pt` 数量。当前这份原始 refs 的 3481 条测试引用恰好对应 3481 个不同图像；加载器也支持同一图像的多个不同表达，每条使用唯一样本 ID。

## 环境

建议沿用已跑通的 DiffVG 环境，Python 3.9 或更高版本。需要单独安装适合本机的 PyTorch；建议 PyTorch 2.1 或更高版本，生成阶段使用 CUDA 构建。已有特征的评测在 CPU 上执行。

仅评测已有特征时安装：

```bash
python -m pip install -r requirements-eval.txt
```

需要生成特征或运行 Qwen 时安装：

```bash
python -m pip install -r requirements.txt
```

这些范围表示依赖接口要求，不是从原训练机器导出的锁定环境。模型默认仅从本地加载；代码不自动下载 SD 或 Qwen。网盘数据、特征和模型权重均不包含在代码包中。

## 路径一：下载已有特征并直接评测

**预计算特征下载链接：待仓库维护者填入实际网盘地址及提取码。**

将下载的 `features/` 解压到 `./Data/RRSIS-D/`，使其与 `split/`、`JPEGImages/` 同级。避免出现 `features/features/` 的重复嵌套。样本 ID 和当前标准 `test.json` 必须一致。

```bash
python evaluate.py --save-images false
```

保存定位图和热力图：

```bash
python evaluate.py --workers 4 --save-images true
```

评测无需 SD、Qwen 或 `JsonTree.json`。默认读取 `./Data/RRSIS-D/features/`，输出到 `./outputs/RRSIS-D_evaluation/`。

终端逐样本显示，TXT 中保存相同的样本行：

```text
14/3481  |  11595_002：IoU=0.564443，pred_bbox_xyxy: [544.4, 581.9, 800.0, 655.6], The small plane is situated on a dry, barren landscape.
```

全部完成后显示并保存：

```text
最终指标：Pr@0.3=... | Pr@0.5=... | Pr@0.7=... | mIoU=... | oIoU=...
```

输出结构：

```text
outputs/RRSIS-D_evaluation/
├── evaluation.txt
├── 03600_000/                 # 仅 save-images=true 时创建
│   ├── localization.png       # GT 绿色框，预测红色框
│   └── heatmap.png            # TFBD 细化后的热力图
└── ...
```

默认 `--save-images true`；使用命令行 `true/false` 切换，也可以修改 `topovg/config.py` 的 `SAVE_IMAGES_DEFAULT`。`false` 不写入图像，不会删除以前已有的 PNG。热力图使用固定 `[0,1]` 色标。完整报告只有样本序号、ID、IoU、预测框、表达和最终指标，不保存 GT 框等附加字段。

每完成 50 条样本，原子更新一次 `evaluation.txt`，最终再写入完整报告。失败或中断保留已完成的行并标记未完成，不将部分样本结果标为最终指标。同一输出路径再次评测会更新 TXT；图像开启时更新对应样本 PNG。输入 `.pt`、划分 JSON 和原图不会被评测修改。

多进程按实际完成顺序输出，因此进度序号不代表 `test.json` 中的原始位置；样本 ID 用于对应结果。使用 `--workers 1` 可以按划分顺序输出。可通过 `--results-dir` 指定结果目录；结果目录必须位于输入特征目录之外。

指标定义保持既有实现：单样本 IoU 范围 `[0,1]`；最终 Pr/mIoU/oIoU 为百分制。`Pr@t` 使用 `IoU >= t`，mIoU 为各样本 IoU 的平均值，oIoU 为所有框的交集面积之和除以并集面积之和。框面积使用连续坐标 `xmax-xmin`、`ymax-ymin`，不额外加 1。

预计算特征与自行单次提取的特征使用相同文件格式。若发布的是此前使用测试 GT 评价候选并选择最高 IoU 的特征，应在下载说明中标明该生成方式；该预计算结果不代表这里单次随机提取的指标。本版本生成器不使用 GT 选取特征，也不保证重新提取后达到历史候选选择的分数。

## 路径二：自己生成特征

### 1. 准备标准 split/test.json

已有已验证的标准测试 JSON 时可直接使用。需要从原始文件重建时：

```bash
python build_rrsis_test_from_refs.py \
  --refs './Data/refs(unc).p' \
  --ann-dir './Data/RRSIS-D/ann_split' \
  --data-root './Data/RRSIS-D' \
  --output-json './Data/RRSIS-D/split/test.json'
```

`--ann-dir` 可以指向数据集 XML 实际所在位置。工具按照原始 `split='test'` 引用和表达匹配 XML 目标对象，保留原来的 `object_index` 和 `image_objectIndex` ID 格式。全部匹配、bbox、图像存在性和 ID 检查成功后才写 JSON；已有 JSON 先备份为时间戳 `.bak`。不复制图像、不按 XML 中的 object 总数扩展测试样本。

例如：

```json
{
  "id": "03600_000",
  "image": "03600",
  "width": 800,
  "height": 800,
  "category": "windmill",
  "expression": "The gray small windmill",
  "bbox_xyxy": [348, 272, 389, 389],
  "bbox_xywh": [348, 272, 41, 117],
  "object_index": 0,
  "depth": 3,
  "source_object_id": "50"
}
```

特征生成只消费 ID、图像和表达，不读取 bbox 或计算指标。GT 和图像尺寸在独立评测阶段使用。

### 2. 检查或补全解析树

沿用已有 `JsonTree.json`，先检查覆盖率：

```bash
python llm_parser.py --check-only
```

该命令不加载 Qwen、不修改文件；完整覆盖返回 0，存在待补全表达返回 2。首次从空缓存开始且还没有 `JsonTree.json` 时，可复制 `examples/JsonTree.example.json` 到 `JsonTree.json`。已有缓存无需复制。

运行补全：

```bash
python llm_parser.py \
  --model-path "$HOME/VG2/AImodel/qwen/Qwen2.5-7B-Instruct"
```

已有有效表达解析直接跳过；只处理缺失或无效的去重表达。每条成功后保存，首次实际修改前备份。未指定 `--output-json` 时更新 `--parser-json` 指定的 JSON，不修改 `.py` 脚本。失败条目存入 `JsonTree.fill_failures.json`；再次运行继续补全。需要单独输出时设置 `--output-json`。

system prompt 和 few-shot 示例位于 `topovg/parser/prompts.py`，拆分时内容保持不变。当前代码依旧支持 `relation: null`、有 anchor 的相对位置，以及 `anchor: null` 的空间描述；空对象 `{}` 不能代替有效 anchor。

### 3. 提取一次特征

使用当前本地 SD1.5 快照：

```bash
python generate_features.py \
  --model-path "$HOME/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
```

一次遍历整个测试集，每个待处理样本一次随机加噪、一次 U-Net 前向，然后保存一个 `.pt`。生成阶段没有单样本 IoU、GT 搜索、候选排名或目标指标停止条件；不再支持旧的 `--target-pr05`、`--target-miou`、`--trials` 等搜索参数。

默认输出 `./Data/RRSIS-D/features/<sample_id>.pt`。已存在的 `.pt` 默认跳过，包括下载的预计算特征。只有显式传入 `--overwrite` 才重新提取并替换；重启程序通过已有 `.pt` 自动跳过完成的样本。生成器按文件是否存在跳过，已有文件的内容与格式由评测器检查。

可先执行 `python generate_features.py --check-only`，检查图像、样本 ID 和待生成样本的 parser 覆盖率，不加载模型或写文件。没有待生成样本时无需模型和 parser。生成只保留 `.pt`，不写候选、GT、预测框、搜索状态 JSON 或热力图。目录中的 `.generate.lock` 是防止多个进程同时写同一目录的文件锁。

既有提取参数保持：SD1.5、`a photograph of ` 前缀、512 像素 PIL 预处理、VAE 均值缩放 `0.18215`、CLIP 77 token、默认 timestep 100、query chunk 512、条件分支注意力按 head 加权聚合。未固定随机种子；长文本超过 77 token 时明确报错。

### 4. 独立评测

```bash
python evaluate.py --save-images true
```

## `.pt` 文件契约

每个文件只包含如下三个键；均为 CPU `torch.float32`：

| 键 | 张量形状 |
|---|---|
| `semantic_mask_8` | `(8, 8)` |
| `semantic_mask_16` | `(16, 16)` |
| `self_64` | `(4096, 4096)` |

不重命名键，不添加 GT、解析树或额外元数据，不压缩/量化已有数值。评测先校验这些键、dtype、shape 和有限数值，再执行原有定位算法。因此符合该契约的原有候选选择 `.pt` 可直接读取，无需转换。

完整 `self_64` FP32 矩阵约 64 MiB/样本；3481 个样本约 218 GiB。模块拆分和取消多候选不会缩小单个特征文件。实际空间还包括 `.pt` 开销、数据和可选 PNG；逐条生成只需暂存当前样本的一个文件。

## 自定义路径与旧路径兼容

| 参数 | 默认/用途 |
|---|---|
| `--data-root` | 生成和评测均默认 `./Data/RRSIS-D` |
| `--split` | 生成和评测均默认 `test`，也支持 `train`、`val` |
| `--feature-dir` | 生成/评测共同使用的直接特征目录，默认 `data-root/features` |
| `--parser-json` | 生成/解析默认 `./JsonTree.json` |
| `--split-json` | parser 默认 `./Data/RRSIS-D/split/test.json`，自定义数据集时自行设置 |
| `--model-path` | 对应入口使用的本地 SD 或 Qwen 模型目录 |
| `--results-dir` | 评测默认 `./outputs/<数据集目录名>_evaluation` |
| `--save-images` | 评测的 `true/false`，同时控制两种 PNG |
| `--workers` | 评测 CPU 进程数，默认 4 |
| `--overwrite` | 生成时显式替换已有 `.pt` |

模型路径还可通过 `TOPOVG_SD_MODEL` 和 `TOPOVG_QWEN_MODEL` 环境变量设置；未设置则分别使用 `./models/stable-diffusion-v1-5` 和 `./models/Qwen2.5-7B-Instruct`。

原来文件还在 `./outputs_RRSIS-D/features/` 时，可直接指定：

```bash
python evaluate.py \
  --feature-dir './outputs_RRSIS-D/features' \
  --save-images false
```

无需移动或重新生成。旧 `--output './outputs_RRSIS-D'` 同样作为“包含 features 的父目录”保留兼容；生成阶段显式使用旧 `--output` 时也输出到该父目录下的 `features/`。`--output` 与 `--feature-dir` 不能同时传入。不传旧参数时，两阶段统一使用数据集目录中的 features。

## 验证与实现说明

```bash
python -m unittest discover -s tests -v
```

本次交付已通过 23 项回归检查，包括条件注意力只选择一次、query 切片保持完整 softmax、head 加权聚合、完整划分单次生成、已有特征保护、无 GT 输入、parser 缓存与重试、实际 spawn 进程池、终端/TXT 格式、图像开关和实际 PNG 生成。迁移时还对提示词、语义执行器、捕获器、TFBD、指标及可视化进行源代码 AST 等价核对。

交付环境没有 PyTorch、OpenCV、实际数据或 GPU 模型，因此未执行真实 SD/Qwen 推理和真实 `.pt` 解码。依赖模型的端到端运行需要在已跑通的 DiffVG 环境完成；上述测试中的生成器和评测 worker 使用注入的模拟依赖，注意力数学测试使用 NumPy。

提供的下载版 `ptp_utils.py` 还包含 Compel、notebook 展示、attention editing 和旧版 forward monkeypatch。当前生效模块保留已跑通的 Diffusers attention processor，避免同时注册两套捕获器；文本编码、条件 head 选择和数值流程均沿用当前实现。不需要额外安装 Compel、einops 或 IPython。相关来源说明见 `NOTICE.md`。

发布代码前，填写实际网盘链接，并将标准数据准备说明、实际预计算特征来源和你选择的项目许可证补齐。代码包不包含数据、解析缓存、模型、预计算特征或新的真实数据指标。
