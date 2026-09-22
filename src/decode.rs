//! 输出侧：校准数学与类型化结果文档。
//!
//! 解码刻意与后端无关：它只消费主机端的 `f32` 切片，并集中负责所有
//! softmax / 温度裁剪 / 熵 / 舍入决策，因此实现只有一份。
//! 移植自官方 `laya` 0.3.5 的 `agent.py::Agent.system_one` 与 `common.py`
//! （`clamp_temperature`、`confidence_from_probs`）。

use std::collections::HashMap;

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::Value;

use crate::encode::{Criteria, Question, QuestionType, Sequence};

/// 有序的 `label -> value` 映射，序列化为 JSON 对象。
type OrderedMap<V> = IndexMap<String, V>;

/// 构建类型化答案并包装成结果文档。
///
/// `logits` 形状为 `[batch, num_markers]`，`action_logits` 形状为
/// `[batch, action_stride]`，二者均为行主序。温度表按引用传入，decode 因此
/// 不需要依赖配置类型本身。
pub(crate) fn build_response(
    questions: &[Question],
    sequences: &[Sequence],
    logits: &[f32],
    action_logits: &[f32],
    num_markers: usize,
    input_tokens: i32,
    action_stride: usize,
    temperature_by_type: &[f32],
    temperature_by_options: &HashMap<String, f32>,
) -> Response {
    let debug = std::env::var("LAYA_DEBUG").is_ok();
    let mut answers = OrderedMap::new();

    for (row, question) in questions.iter().enumerate() {
        let sequence = &sequences[row];
        let num_options = sequence.markers.len();

        // 对本问题的选项 logits 做温度校准后的 softmax。
        let temperature = temperature(
            temperature_by_type,
            temperature_by_options,
            question.question_type,
            num_options,
        );

        let row_logits = &logits[row * num_markers..row * num_markers + num_options];
        let scaled = row_logits
            .iter()
            .map(|&logit| logit / temperature)
            .collect::<Vec<_>>();
        let probabilities = softmax(&scaled);

        if debug {
            eprintln!(
                "{}: options={num_options} length={} markers={:?} \
                 temperature={temperature} logits={row_logits:?}",
                question.id,
                sequence.token_ids.len(),
                sequence.markers
            );
        }

        let action =
            action_answer(&action_logits[row * action_stride..row * action_stride + action_stride]);
        let answer = decode_answer(question, &probabilities, action);

        answers.insert(question.id.clone(), answer);
    }

    Response {
        model: MODEL_NAME.to_string(),
        answers,
        usage: Usage {
            input_tokens,
            output_tokens: 0,
        },
    }
}

/// 结果文档的 `model` 值。参考实现在 `Agent.system_one` 中硬编码该字符串，
/// 而非读取 `rl_agent_config.json` 里的 `model_name`。
const MODEL_NAME: &str = "laya-rl-agent";

/// 解码动作头：只保留参考实现输出的 `act_probability`。
///
/// 动作索引 0 即“直接作答”（`answer`）的概率，对应 `system_one` 中的
/// `softmax(act)[:, 0]`。
fn action_answer(action_logits: &[f32]) -> ActionAnswer {
    let probabilities = softmax(action_logits);
    ActionAnswer {
        act_probability: round4(probabilities.first().copied().unwrap_or(0.0)),
    }
}

/// 把单个问题校准后的分布解码为类型化答案。
fn decode_answer(question: &Question, probabilities: &[f32], action: ActionAnswer) -> Answer {
    let num_options = probabilities.len();
    match &question.criteria {
        Criteria::Choice(options) => {
            let labels: Vec<&str> = options.iter().map(|(label, _)| label.as_str()).collect();
            let best = argmax(probabilities);
            Answer::Choice {
                choice: labels.get(best).copied().unwrap_or_default().to_string(),
                probabilities: label_distribution(labels.iter().copied(), probabilities),
                confidence: round4(confidence(probabilities, num_options)),
                action,
            }
        }
        Criteria::Score(_) => {
            let score: f32 = probabilities
                .iter()
                .enumerate()
                .map(|(index, &probability)| index as f32 * probability)
                .sum();
            Answer::Score {
                score: round4(score),
                legend: question
                    .legend()
                    .into_iter()
                    .enumerate()
                    .map(|(index, level)| (index.to_string(), level))
                    .collect(),
                probabilities: label_distribution(
                    (0..num_options).map(|index| index.to_string()),
                    probabilities,
                ),
                confidence: round4(confidence(probabilities, num_options)),
                action,
            }
        }
        Criteria::Noul { .. } => {
            // `noul` 的 confidence 是最大类概率（`max(p[1], 1 - p[1])`），
            // 而非 choice/score 的归一化熵。
            let positive = probabilities.get(1).copied().unwrap_or(0.0);
            Answer::Noul {
                noul: round4(positive),
                confidence: round4(positive.max(1.0 - positive)),
                action,
            }
        }
    }
}

/// 把标签与四舍五入后的概率配对，保持顺序。
fn label_distribution<L: Into<String>>(
    labels: impl Iterator<Item = L>,
    probabilities: &[f32],
) -> OrderedMap<f64> {
    labels
        .zip(probabilities)
        .map(|(label, &probability)| (label.into(), round4(probability)))
        .collect()
}

/// 数值稳定的 softmax。
fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exponentials: Vec<f32> = logits
        .iter()
        .map(|&logit| (logit - maximum).exp())
        .collect();
    let sum: f32 = exponentials.iter().sum();
    if sum > 0.0 {
        for exponential in exponentials.iter_mut() {
            *exponential /= sum;
        }
    }
    exponentials
}

/// 归一化的香农熵置信度：`1 - H(p) / log(k)`。
fn confidence(probabilities: &[f32], num_options: usize) -> f32 {
    if num_options < 2 {
        return 1.0;
    }
    let num_options = num_options.min(probabilities.len());
    let entropy: f32 = probabilities[..num_options]
        .iter()
        .map(|&probability| {
            let clamped = probability.clamp(1e-12, 1.0);
            -clamped * clamped.ln()
        })
        .sum();
    let result = 1.0 - entropy / (num_options as f32).ln();
    result.clamp(0.0, 1.0)
}

/// 四舍五入到四位小数，与参考实现的输出 schema 一致。
fn round4(value: f32) -> f64 {
    ((value as f64) * 10000.0).round() / 10000.0
}

/// 最大值所在的索引（并列时取第一个）。
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}

/// 某问题类型在拥有 `num_options` 个选项时的后验温度，已按参考实现裁剪。
/// 温度属于校准数学，因此实现留在 decode 侧，只接收两张温度表。
fn temperature(
    by_type: &[f32],
    by_options: &HashMap<String, f32>,
    question_type: QuestionType,
    num_options: usize,
) -> f32 {
    let bucket = temperature_bucket(question_type, num_options);
    let raw = match by_options.get(&bucket) {
        Some(temperature) => *temperature,
        None => by_type.get(question_type as usize).copied().unwrap_or(1.0),
    };
    clamp_temperature(raw)
}

/// 校准分桶，例如 `choice:3-5`、`noul:2`。
fn temperature_bucket(question_type: QuestionType, num_options: usize) -> String {
    let size = if num_options <= 2 {
        "2"
    } else if num_options <= 5 {
        "3-5"
    } else if num_options <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", question_type.as_str(), size)
}

/// 参考 `common.py::clamp_temperature`：把拟合温度限制在 `[0.5, 5.0]`，
/// 非有限值回落到 `1.0`。低于 1 的温度会锐化 logits，导致置信度虚高。
fn clamp_temperature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.5, 5.0)
    } else {
        1.0
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: OrderedMap<Answer>,
    pub usage: Usage,
}

/// 单个类型化答案。序列化时带 `type` 判别字段以匹配参考 schema。
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        action: ActionAnswer,
    },
    Score {
        score: f64,
        legend: OrderedMap<Value>,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        action: ActionAnswer,
    },
    Noul {
        noul: f64,
        confidence: f64,
        action: ActionAnswer,
    },
}

/// 动作头的解码结果：只保留参考实现输出的 `act_probability`。
#[derive(Clone, Debug, Serialize)]
pub struct ActionAnswer {
    pub act_probability: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
}
