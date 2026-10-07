<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.md">English</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="docs/logo.png" alt="ai-eyes" width="360">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/ai-eyes-mcp/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/ai-eyes-mcp/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
</p>

# ai-eyes-mcp

**版本：** 1.2.0

基于视觉的评估器 MCP 服务器。通过 SigLIP2 为 Claude 提供客观的图像判断——它*衡量*，而不是叙述，因此不会产生幻觉。

## 问题

当 Claude 需要验证图像中的内容时——“这个精灵是否持有剑？”、“是否有登录按钮？”——生成式视觉语言模型（LLaVA、GPT-4V）会自信地给出错误的答案。它们完成的是叙述，而不是观察。LLaVA 13B 在没有武器的图像上报告说“角色拿着一把大剑”，置信度为 0.90，并且对每个裁剪后的图像都如此。

## 解决方案

SigLIP2 是一个判别式视觉模型。它不生成文本——而是衡量图像和文本描述之间的相似性，并返回一个校准后的 sigmoid 分数。当存在武器时，分数比不存在时高 10-100 倍。当它无法判断时，分数会很低。它不会产生幻觉。

此 MCP 服务器将 SigLIP2 作为工具进行封装，任何 Claude 工作流程都可以调用这些工具。

## 工具

| 工具 | 其作用 |
|------|-------------|
| `image_contains` | “这张图像是否包含 X？”→ sigmoid 分数 |
| `image_classify` | 根据 N 个候选标签对图像进行评分 |
| `image_compare` | 两个图像之间的余弦相似度，相对于调用方提供的“不同”阈值 |
| `image_score_batch` | 将 N 个图像与一个查询进行比较 |
| `image_verify` | 诚实的相对判断：目标与替代方案 → 决策 + 差值 + 置信度 |
| `image_rank` | 根据一个参考对象对 N 个候选对象进行排序 → 前 k 个，并显示差值 |
| `eyes_selftest` | 在捆绑的参考图像上进行自测（证明安装和校准） |
| `eyes_status` | 健康检查：模型、设备、加载状态 |

### 何时使用其他方法

ai-eyes 回答 **“关于像素的主张是否为真？”** 它对你提供的假设进行评分——它无法告诉你图像中有什么，除非你已经描述过该图像。

对于 **标题、描述或从图像中读取的文本**，这是生成式工作，需要使用不同的工具：**[plain-sight](https://github.com/mcp-tool-shop-org/plain-sight)**（Florence-2）。其输出可以通过构造来产生细节——将任何关键内容带回此处并使用 `image_verify` 进行衡量。

这是一个有意的组合，并且每个工具的指导都指向另一个：**plain-sight 描述，ai-eyes 衡量。**

## 快速入门

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

或者作为模块运行：`python -m ai_eyes_mcp`

### Claude 代码配置

```json
{
  "mcpServers": {
    "ai-eyes": {
      "command": "ai-eyes-mcp",
      "env": {
        "AI_EYES_MODEL_DIR": "/path/to/model/cache"
      }
    }
  }
}
```

## 配置

| 环境变量 | 默认值 | 用途 |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | HuggingFace 模型 |
| `AI_EYES_MODEL_REVISION` | 固定的提交 SHA | 模型版本。**必须是 40 个字符的十六进制提交 SHA。** `main`、标签或空值会导致加载失败，而不是回退——请参阅下面的*可重复性*。 |
| `AI_EYES_MODEL_DIR` | HF 默认缓存 | 模型缓存目录 |
| `AI_EYES_DEVICE` | 如果可用，则为 `cuda`，否则为 `cpu` | torch 设备。设置一个字面设备（`cuda`、`cpu`、`cuda:1`）——没有 `auto` 值；`AI_EYES_DEVICE=auto` 会引发错误。 |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | `image_contains` 的默认阈值 |
| `AI_EYES_LOG_LEVEL` | `WARNING` | 日志详细程度：`DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `AI_EYES_EAGER_LOAD` | 未设置 | 如果为真，则在启动时加载模型，以便出现故障的模型/缓存能够快速失败（而不是在第一次调用工具时）。 |
| `AI_EYES_DTYPE` | 全精度 | `float16` / `bfloat16` 以减少一半的 VRAM |
| `AI_EYES_EMBED_CACHE` | `64` | 内存中的图像嵌入备忘录大小。以路径 + mtime + 大小为键，因此重写的文件的分数会重新计算，而不会提供过时的结果。没有磁盘，也没有辅助文件。 |

### 可重复性——权重已固定

只有当你知道哪个权重生成了该分数时，该分数才有意义。模型版本**固定为特定的提交 SHA**，并传递给每次加载，并且**在包含数字的每个有效负载中都会报告**。两个相隔几个月安装的版本对相同的输入返回相同的分数。

如果未设置修订版，则会解析到固定的版本——而不是浮动分支。只有当它是一个*不同*的 40 个字符的 SHA（操作员意图）时，才会接受覆盖；`main`、标签或空字符串会导致加载失败并显示可操作的消息，而不是默默地漂移。

**日志记录：**服务器在 `ai_eyes_mcp` 日志记录器下将日志记录到 **stderr**（stdout 是 MCP 协议通道）。使用 `AI_EYES_LOG_LEVEL`（上面）设置级别，或者附加你自己的处理程序到 `logging.getLogger("ai_eyes_mcp")`。

**第一次调用：**模型会延迟加载——**第一个**图像工具调用会下载/加载 SigLIP2（GPU 上大约需要 10-20 秒；首次下载时间更长），后续的调用大约需要 100 毫秒。设置 `AI_EYES_EAGER_LOAD=1` 以在服务器启动时加载，或者调用 `eyes_status`，它会报告 `loaded` 而不会触发加载。**在冷服务器上并非免费**——第一次调用需要一次性导入库（测量时间约为 10 秒；预热后为 2 毫秒）。

## 分数的工作原理

SigLIP2 使用 **sigmoid** 分数，而不是 softmax。每个图像-文本对都会获得一个独立的概率（0-1）：

仅为粗略的说明——**这些范围不能在查询或图像样式之间转移**，并且下一部分解释了为什么你不应该在此基础上进行构建：

- **高**（>0.1）：强视觉匹配
- **低**（<0.01）：弱或无匹配
- **中等**（0.01-0.1）：模糊

分数不是相对的。多个查询可以对同一图像进行高分（例如，一张同时包含剑和盾牌的图像）。

### ⚠ 查询措辞很重要——为了获得可靠的决策，请使用 `image_classify`

SigLIP2 sigmoid 分数**对查询措辞敏感**：对于*相同的*图像，绝对分数会因措辞而发生很大变化（匹配样式的短语可以比通用短语高 10-100 倍）。因此，固定的 `threshold` 需要针对每个用例进行查询工程，并且阈值不能跨图像样式转移。

对于各种输入，进行可靠的“是/否”决策时，请优先使用 **`image_classify`** ——它*对候选标签进行排序比较*，并且不受绝对分数大小的影响。只有在您能够控制查询措辞和图像风格的情况下，才可以使用经过调整的阈值的 `image_contains`。 `eyes_status` 在其 `scoring_guidance` 字段中也体现了这一点。

默认阈值（`0.02`）是一个宽松的下限，而不是一个通用的截止点——请根据您的查询和图像风格进行调整，或者使用 `image_classify`。

## 架构

```
engine.py          Standalone SigLIP2 wrapper — no MCP dependency.
                   Lazy-loads model on first inference call.
                   Importable directly for non-MCP use cases.

server.py          FastMCP wrapper that exposes engine methods as MCP tools.
                   Thin layer: input validation, error shaping, tool metadata.

__main__.py        Entry point for `python -m ai_eyes_mcp`.
```

`engine.py` 是核心——它负责模型加载、设备选择以及所有推理逻辑。 `server.py` 从不直接操作 torch；它将所有内容委托给引擎。这意味着您可以 `from ai_eyes_mcp.engine import SigLIPEngine`，并在任何 Python 脚本中使用它，而无需引入 FastMCP。

ai-eyes-mcp 仅评估图像，不做其他操作。它对您如何使用该数字没有意见——例如，编目、精灵流水线、生成资产的 CI 门控——这些都属于消费者的范畴。

## 工具参考

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

检查图像是否包含查询描述的内容。返回一个独立的 sigmoid 分数（0-1）。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `image_path` | 字符串 | 是 | 图像文件的绝对路径 |
| `query` | 字符串 | 是 | 要查找的内容（例如，“拿着剑的人”） |
| `threshold` | 浮点数 | no | 存在判定的分数阈值（默认值为 0.02） |

返回值：`{present, score, threshold, query, truncated, revision, elapsed_ms}`

`truncated: true` 表示您的查询超出了文本编码器的 64 个令牌容量，并且**该分数仅反映了前 64 个令牌**——将其视为不完整的结果，而不是一个数字。

### `image_classify`

```
image_classify(image_path, labels)
```

对图像与多个候选标签进行评分。返回独立的 sigmoid 分数——不是 softmax。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `image_path` | 字符串 | 是 | 图像文件的绝对路径 |
| `labels` | 字符串[] | 是 | 要评分的候选标签（最多 20 个） |

返回值：`{scores, best, best_score, truncated, revision, elapsed_ms}`

`best` 是从全精度分数中选择的；显示的值仅四舍五入到与该选择一致的程度。

### `image_compare`

```
image_compare(image_a, image_b, baselines=None)
```

使用 SigLIP2 嵌入的余弦相似度计算两个图像之间的视觉相似度。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `image_a` | 字符串 | 是 | 第一个图像的绝对路径 |
| `image_b` | 字符串 | 是 | 第二个图像的绝对路径 |
| `baselines` | 字符串[][] | no | 在您的风格中**不**匹配的图像对。只有当 A-B 的差异超过此下限时，才将其计为分离。 |

**如果没有 `baselines`，则该数字是一个测量值，而不是一个判决**——有效负载包含 `incomplete: true`。相似度下限不会跨图像风格进行转移，因此该工具不会自行设定一个。在一种精灵风格中，对六个*不同*角色的图像进行测量：**0.698–0.836**，对于同一图像与其自身进行比较则为 1.0。从这些数据中选择固定的截止点对于照片、屏幕截图或渲染图来说是不合适的——因此您需要提供对比度，就像 `image_verify` 接受替代方案一样。

返回值：`{similarity, separated, incomplete, margin, baseline_max, confidence, image_a, image_b, revision, elapsed_ms}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

对单个查询评分多个图像。每次调用最多可以处理 100 个图像。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `image_paths` | 字符串[] | 是 | 绝对图像路径列表 |
| `query` | 字符串 | 是 | 要查找的内容 |
| `threshold` | 浮点数 | no | 分数阈值（默认值为 0.02） |

返回值：`{query, threshold, total, scored, present, absent, errors, error_details?, results: [{path, score, present}], truncated, revision, elapsed_ms}`

### `image_verify`

```
image_verify(image_path, target, alternatives)
```

诚实的**相对**判决——将 `target` 与调用者提供的 `alternatives` 进行比较（必需，≥1），并返回一个决策 + 范围 + 置信度。由于它是相对的，而不是绝对阈值，因此它对 SigLIP 的查询措辞敏感性具有鲁棒性。要获取原始分数，请使用 `image_contains`；要进行完整的排序，请使用 `image_classify`。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `image_path` | 字符串 | 是 | 图像的绝对路径 |
| `target` | 字符串 | 是 | 要验证的假设 |
| `alternatives` | 字符串[] | 是 | 用于排序的对比替代方案（≥1） |

返回值：`{present, target, target_score, best_alternative, best_alternative_score, margin, confidence, truncated, revision, elapsed_ms}`——`confidence` 是 `high` / `moderate` / `low — inconclusive`，描述了测量的差距。

### `image_rank`

```
image_rank(reference, candidates, k=5, baselines=None)
```

通过相似度对一个参考图像进行排序。它只对参考图像进行编码**一次**，而不是对每个候选图像都进行编码。

| 参数 | 类型 | 必需 | 说明 |
|-----------|------|----------|-------------|
| `reference` | 字符串 | 是 | 参考图像的绝对路径 |
| `candidates` | 字符串[] | 是 | 要排序的候选图像路径 |
| `k` | 整数 | no | 返回的最大匹配数量（默认值为 5） |
| `baselines` | 字符串[][] | no | 在您的风格中**不**匹配的图像对——低于此值的任何内容都不被认为是“接近”。 |

返回值：`{matches, nothing_close, incomplete, baseline_max, k, reference, revision, elapsed_ms}`

**如果没有 `baselines`，则排序是一个测量值，而不是“这些匹配”**（`incomplete: true`）。有了基线之后，只有高于调用者提供的下限的候选对象才会被视为匹配——如果没有任何对象超过该下限，则 `matches` 将是**空的**。始终返回 k 个结果的排序动词是在没有信号的情况下，可以自信地回答；这个工具会拒绝这样做。

### `eyes_status`

```
eyes_status()
```

检查服务器状态。不会触发模型加载。

返回值：`{model_id, revision, device, dtype, loaded, cache_dir, parameters?, vram_mb?, scoring_guidance, note?}`

当 `loaded` 为真时，它还会返回 `parameters`（`1136M`）和 `vram_mb`（仅限 CUDA）。

### `eyes_selftest`

```
eyes_selftest()
```

在几个捆绑的参考图像上运行模型，并确认预期的排序关系是否成立——证明安装已正确加载并且 SigLIP2 已校准。如果尚未加载模型，则会加载模型。

返回值：`{passed, checks: [{name, expected, measured_a, measured_b, ok}], model_id, device, torch_version, transformers_version}`

`measured_a` / `measured_b` 以完整分辨率报告——一个很小的非零分数将以科学计数法显示（例如，`4.3813e-08`），而不是坍缩到 `0`。

## 安全与信任

此工具**仅在本地运行**。

- **涉及的数据：** 本地图像文件（只读）、HuggingFace 模型缓存（下载一次）
- **网络：** 所有*推理*都在本地进行——没有数据外发，绝对不会。唯一的例外是**第一次运行**，它会从 HuggingFace 下载约 1.6 GB 的模型权重。此后，该工具将不再访问网络。
- **对于隔离或审计外发的部署：** 预先填充缓存，并指向 `AI_EYES_MODEL_DIR`。然后，该工具在任何时候都不会进行任何网络调用。
- **不处理任何敏感信息**——不会读取、存储或传输凭据或 API 密钥
- **没有遥测数据**——不会收集或发送任何数据
- **不会修改文件**——图像文件以只读方式打开，绝不会被修改
- **不会执行危险操作**——没有删除、终止或重新启动的操作
- **仅返回结构化错误**——堆栈跟踪绝不会暴露给客户端

## 要求

- Python >= 3.10
- CUDA GPU recommended. **Measured VRAM: ~4.3 GB at the default `float32`**, ~2.2 GB with `AI_EYES_DTYPE=float16`. The model is 1136M parameters — "SO400M" names the vision tower, not the assembled model.
- CPU fallback available (slower, ~10x)
- Model downloads ~1.6GB on first use

## 开发说明

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run CI-safe tests (no model required) — this is what CI runs
pytest -m "not dogfood"

# Run everything, including GPU tests that need the model
pytest

# Full verify: imports, tool registration, cold-status gate, wheel build, CI-safe tests
bash verify.sh
```

请使用**控制台脚本**`pytest`，而不是`python -m pytest`。后者会将工作目录设置为`sys.path`，并且可能会隐藏一个导入失败的情况，而 CI 系统会检测到这一点。

## 许可证

MIT 许可

---

由 [MCP Tool Shop](https://mcp-tool-shop.github.io/) 构建
