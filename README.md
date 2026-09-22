# rustlaya

面向 [Laya](https://huggingface.co/convaiinnovations/laya) 非自回归类型化决策模型的
原生 **candle**（Metal 后端）推理。输入一个状态（state）和一组类型化问题
（`choice`、`score`、`noul`），一次前向传播即可返回经过校准的答案。

本 crate 移植自官方 Python 运行时 [`laya`](https://pypi.org/project/laya/) 以及 MLX
参考实现 [`laya_mlx`](https://github.com/mizorewww/laya-mlx)。

## 架构

```
src/
  lib.rs      Laya 门面：load() + predict()，仅做串联
  hub.rs      把本地目录或 Hugging Face 仓库解析为文件路径
  encode.rs   输入侧：QuestionType/Question/Criteria、Python 兼容渲染、特殊 token 与分词后的序列（不含张量）
  model.rs    candle 网络：ModernBERT 编码器 + 决策头 + 打分头 + 动作头，检查点加载/重映射，以及批处理
  decode.rs   输出侧：校准数学（softmax、温度、熵、动作解码）与类型化结果文档（不含张量后端）
```

数据流：`Laya::predict` 为每个问题构建一条序列（`encode.rs`），把它们 padding 成
一个批（`model.rs`），执行一次前向传播（`model.rs`）并**只返回原始 logits**，再解码
为类型化答案（`decode.rs`）。

设计约束：

- `encode.rs` 与 `decode.rs` 不依赖 candle；它们是纯函数，可独立测试。
- `model.rs` 只返回 logits，绝不负责输出格式化。检查点命名全部收敛在其
  `checkpoint_name` / `load_var_builder` 中。
- 所有 softmax / 温度 / 熵 / 舍入都在 `decode.rs`（只有一份实现）。
- 错误统一为 `anyhow::Result`；`candle_core::Error` 在边界处转换。

## 用法

```bash
# CLI 示例（模型由 LAYA_MODEL 指定：本地目录或 Hugging Face 仓库 id；缺省为官方仓库）
cargo run --example demo
```

```rust
use serde_json::json;
use rustlaya::{Criteria, Laya, Question, QuestionType};

let laya = Laya::load("convaiinnovations/laya")?;
let state = json!("I was billed twice. Please refund the duplicate today.");
let questions = vec![Question {
    id: "department".to_string(),
    question_type: QuestionType::Choice,
    instructions: "Which department should handle this request?".to_string(),
    criteria: Criteria::Choice(vec![
        ("billing".to_string(), Some(json!("invoices, payments"))),
        ("technical".to_string(), Some(json!("bugs"))),
    ]),
}];

let result = laya.predict(&state, &questions)?; // -> Response (Serialize)
```

### 结果 schema

```jsonc
{
  "model": "laya-rl-agent",            // 固定值，与参考实现一致
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 0.96, "technical": 0.02 },
      "confidence": 0.84,              // 1 - H(p)/log(k)
      "action": { "act_probability": 1.0 }
    },
    "urgency": { "type": "score", "score": 1.36, "legend": { "0": "low", "1": "high" }, ... },
    "refund":  { "type": "noul", "noul": 0.82, "confidence": 0.82, "action": { ... } }
  },
  "usage": { "input_tokens": 132, "output_tokens": 0 }
}
```

`action.act_probability` 即参考实现的 `softmax(act_logits)[..., 0]`，也就是空操作
`answer` 动作的概率。`noul` 的 confidence 是最大类概率 `max(p[1], 1 - p[1])`，而非
`choice` / `score` 所用的熵度量。

## 与 Python 参考实现的一致性

行为标准是官方 [`laya`](https://pypi.org/project/laya/) 0.3.5 包
（`laya/agent.py`、`laya/common.py`）。

刻意对齐的点：

- 选项渲染与 `common.py::render_criterion` / `render_options` 一致：字符串原样透传，
  结构化值（数字、布尔、数组、对象）渲染为 `json.dumps(..., ensure_ascii=False)` 的
  紧凑 JSON，且只有 `null` / `""` 表示“无描述”——`0` 与 `false` 都是合法取值。
- 状态序列化使用 `json.dumps(..., ensure_ascii=False)` 的空格风格
  （`{"a": 1, "b": [1, 2]}`）。
- 温度按 `clamp_temperature` 裁剪到 `[0.5, 5.0]`。
- 输出 schema 与 `Agent.system_one` 一致：`model` 为 `"laya-rl-agent"`，每个答案
  都带 `action.act_probability`，`noul` 也报告 `confidence`。
- 动作头消费 top-2 概率、熵与 `k/255` 特征，与 `common.py` 完全一致。
- 编码器使用与参考实现相同的全局/滑动窗口交替注意力，以及按层类型的 RoPE base。

已知偏差：

- 默认在 **Metal** 上以 **F16** 推理（出于性能）：既能原生加速，也与检查点的存储
  格式一致，且（与 BF16 不同）无需逐算子模拟；声明为 `fp32` 的检查点会以 F32 保留。
  参考实现则以 F32 跑在 CPU/MPS 上，因此四位小数概率最多相差约 ~0.004，argmax 不会
  改变；如需严格对齐 F32，可用 `Laya::load_with_dtype(model, DType::F32)`。

## 环境变量

| 变量 | 作用 |
|---|---|
| `LAYA_MODEL` | 示例程序的默认检查点（目录或仓库 id） |
| `LAYA_REVISION` | 固定 Hugging Face 版本号 |
| `LAYA_DEBUG` | 把每个问题的 logits 打印到 stderr |
