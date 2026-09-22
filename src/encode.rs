//! 输入侧：类型化问题、用于构建 prompt 的 Python 兼容渲染、特殊 token 解析
//! 以及序列构造。
//!
//! 移植自官方 `laya` 0.3.5 的 `common.py`（`QTYPES`、`render_options`、
//! `build_sequence`）。渲染刻意对齐其语义：字符串原样透传，结构化值走
//! `json.dumps(..., ensure_ascii=False)`，且只有 `None` / `""` 表示“无描述”。
//! `common.py` 的 `render_criterion` 与 `serialize_state` 实现相同，这里统一由
//! `python::to_text` 承担。

use anyhow::{anyhow, bail};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

/// 序列格式：`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`。
/// 单个选项文本在其 `[MASK]` marker 之后保留的最大 token 数。
const OPTION_TOKENS_MAX: usize = 48;
/// 剩余选项预算低于此值时，各选项被均匀收缩。
const OPTION_BUDGET_MIN: usize = 16;
/// 收缩时每个选项保留 token 数的下界。
const OPTION_TOKENS_MIN: usize = 4;
/// 为问题头部保留的最小 token 数。
const HEAD_TOKENS_MIN: usize = 8;

/// 为每个问题构建一条分词后的序列，并校验每个问题的选项能放进 `head_max_len`。
/// 它是 [`crate::decode::build_response`] 在输入侧的对应物。
pub(crate) fn build_sequences(
    tokenizer: &Tokenizer,
    special_tokens: &SpecialTokens,
    state: &Value,
    questions: &[Question],
    max_len: usize,
    head_max_len: usize,
) -> anyhow::Result<Vec<Sequence>> {
    if questions.is_empty() {
        bail!("no questions supplied");
    }

    let encoder = SequenceEncoder::new(tokenizer, special_tokens);
    let mut sequences = Vec::with_capacity(questions.len());
    for question in questions {
        let sequence = encoder.build_sequence(state, question, max_len, head_max_len)?;
        if sequence.markers.len() != question.render_options().len() {
            bail!(
                "question {:?}: options do not fit in head_max_len={head_max_len} tokens",
                question.id
            );
        }
        sequences.push(sequence);
    }
    Ok(sequences)
}

/// 基于分词器与已解析的特殊 token，把问题与状态构建成模型输入序列。
struct SequenceEncoder<'a> {
    tokenizer: &'a Tokenizer,
    special: &'a SpecialTokens,
}

impl<'a> SequenceEncoder<'a> {
    fn new(tokenizer: &'a Tokenizer, special: &'a SpecialTokens) -> Self {
        Self { tokenizer, special }
    }

    /// 构建完整序列：
    /// `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`。
    fn build_sequence(
        &self,
        state: &Value,
        question: &Question,
        max_len: usize,
        head_max_len: usize,
    ) -> anyhow::Result<Sequence> {
        let Sequence {
            mut token_ids,
            markers,
            question_type,
        } = self.prefix_sequence(question, head_max_len)?;
        let room = max_len.saturating_sub(token_ids.len()).saturating_sub(1);
        let state_text = python::to_text(state).replace(&self.special.mask_token, " ");
        let mut state_token_ids = self.encode(&state_text)?;
        state_token_ids.truncate(room);
        token_ids.extend(state_token_ids);
        token_ids.push(self.special.sep_token_id as i32);
        token_ids.truncate(max_len);
        let markers = markers
            .into_iter()
            .filter(|&marker| marker < max_len as i32)
            .collect();
        Ok(Sequence {
            token_ids,
            markers,
            question_type,
        })
    }

    /// 构建仅含问题的前缀部分（在拼接 state token 与最终截断之前）。
    fn prefix_sequence(
        &self,
        question: &Question,
        head_max_len: usize,
    ) -> anyhow::Result<Sequence> {
        let mask_token = self.special.mask_token.clone();
        let strip_mask = |text: &str| text.replace(&mask_token, " ");

        let question_text = format!(
            "{} question: {}",
            question.question_type.as_str(),
            strip_mask(&question.instructions)
        );
        let mut head_token_ids = self.encode(&question_text)?;

        let options = question.render_options();
        let mut option_token_ids: Vec<Vec<i32>> = Vec::with_capacity(options.len());
        for option in options {
            let text = format!(" {}", strip_mask(&option));
            let mut token_ids = vec![self.special.mask_token_id as i32];
            token_ids.extend(self.encode(&text)?.into_iter().take(OPTION_TOKENS_MAX));
            option_token_ids.push(token_ids);
        }

        let head_budget = fit_options(&mut option_token_ids, head_max_len);
        head_token_ids.truncate(head_budget.max(HEAD_TOKENS_MIN));

        let mut token_ids = vec![self.special.cls_token_id as i32];
        token_ids.extend(head_token_ids);
        token_ids.push(self.special.sep_token_id as i32);

        let mut markers = Vec::with_capacity(option_token_ids.len());
        for option in &option_token_ids {
            markers.push(token_ids.len() as i32);
            token_ids.extend_from_slice(option);
        }
        token_ids.push(self.special.sep_token_id as i32);
        Ok(Sequence {
            token_ids,
            markers,
            question_type: question.question_type as i32,
        })
    }

    fn encode(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        let encoding = self
            .tokenizer
            .encode(text.to_string(), false)
            .map_err(|error| anyhow!("tokenizer error: {error}"))?;
        Ok(encoding
            .get_ids()
            .iter()
            .map(|&token_id| token_id as i32)
            .collect())
    }
}

/// 当问题头部超出 `head_max_len` 时收缩各选项的 token 列表，并返回留给问题文本
/// 的预算（不会小于零）。
fn fit_options(option_token_ids: &mut [Vec<i32>], head_max_len: usize) -> usize {
    let total_tokens: usize = option_token_ids.iter().map(Vec::len).sum();
    let mut token_budget = head_max_len as isize - total_tokens as isize;
    if token_budget < OPTION_BUDGET_MIN as isize {
        let tokens_per_option = OPTION_TOKENS_MIN
            .max(head_max_len.saturating_sub(OPTION_BUDGET_MIN) / option_token_ids.len().max(1));
        for token_ids in option_token_ids.iter_mut() {
            token_ids.truncate(tokens_per_option);
        }
        let total_tokens: usize = option_token_ids.iter().map(Vec::len).sum();
        token_budget = head_max_len as isize - total_tokens as isize;
    }
    token_budget.max(0) as usize
}

/// 一条分词后的问题序列，以及各选项 marker 的位置。
pub(crate) struct Sequence {
    pub(crate) token_ids: Vec<i32>,
    pub(crate) markers: Vec<i32>,
    pub(crate) question_type: i32,
}

/// 从 [`TokenizerConfig`] 解析出的特殊 token。
#[derive(Clone, Debug)]
pub(crate) struct SpecialTokens {
    pub(crate) cls_token_id: u32,
    pub(crate) sep_token_id: u32,
    pub(crate) pad_token_id: u32,
    pub(crate) mask_token: String,
    pub(crate) mask_token_id: u32,
}

impl SpecialTokens {
    pub(crate) fn resolve(tokenizer: &Tokenizer, config: &TokenizerConfig) -> anyhow::Result<Self> {
        let text_of = |entry: &Option<SpecialToken>, name: &str| -> anyhow::Result<String> {
            entry
                .as_ref()
                .map(|token| token.text().to_string())
                .ok_or_else(|| anyhow!("tokenizer_config.json is missing `{name}`"))
        };
        let id_of = |entry: &Option<SpecialToken>, name: &str| -> anyhow::Result<u32> {
            let text = text_of(entry, name)?;
            tokenizer
                .token_to_id(&text)
                .ok_or_else(|| anyhow!("tokenizer has no id for `{name}` = {text:?}"))
        };
        Ok(SpecialTokens {
            cls_token_id: id_of(&config.cls_token, "cls_token")?,
            sep_token_id: id_of(&config.sep_token, "sep_token")?,
            pad_token_id: id_of(&config.pad_token, "pad_token")?,
            mask_token: text_of(&config.mask_token, "mask_token")?,
            mask_token_id: id_of(&config.mask_token, "mask_token")?,
        })
    }
}

/// `tokenizer/tokenizer_config.json`：我们所需解析的特殊 token 条目。
///
/// 其余字段（如 `model_max_length`、`tokenizer_class`）用不到，会被忽略。
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct TokenizerConfig {
    #[serde(default)]
    cls_token: Option<SpecialToken>,
    #[serde(default)]
    sep_token: Option<SpecialToken>,
    #[serde(default)]
    pad_token: Option<SpecialToken>,
    #[serde(default)]
    mask_token: Option<SpecialToken>,
}

/// 特殊 token 既可能是普通字符串，也可能是带 `content` 字段的 added-token
/// 对象，取决于 tokenizer 的导出方式。
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum SpecialToken {
    Text(String),
    AddedToken { content: String },
}

impl SpecialToken {
    fn text(&self) -> &str {
        match self {
            SpecialToken::Text(text) => text,
            SpecialToken::AddedToken { content } => content,
        }
    }
}

/// 归一化后的问题表示。
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub question_type: QuestionType,
    pub instructions: String,
    pub criteria: Criteria,
}

impl Question {
    /// 按标签索引顺序返回选项文本（对应 `common.py::render_options`）。
    ///
    /// 与旧参考不同：只有 `None` / `null` / `""` 视为“无描述”，`0` 与 `False`
    /// 都是合法取值；结构化值渲染为紧凑 JSON 而非 Python `repr`。
    fn render_options(&self) -> Vec<String> {
        match &self.criteria {
            Criteria::Choice(items) => items
                .iter()
                .map(|(label, description)| match description {
                    Some(value) if !python::is_blank(value) => {
                        format!("{}: {}", label, python::to_text(value))
                    }
                    _ => label.clone(),
                })
                .collect(),
            Criteria::Score(items) => items
                .iter()
                .enumerate()
                .map(|(index, level)| format!("level {}: {}", index, python::to_text(level)))
                .collect(),
            Criteria::Noul {
                false_criterion,
                true_criterion,
            } => vec![
                format!(
                    "false: {}",
                    false_criterion
                        .as_ref()
                        .filter(|value| !python::is_blank(value))
                        .map(python::to_text)
                        .unwrap_or_else(|| "no, the statement does not hold".to_string())
                ),
                format!(
                    "true: {}",
                    true_criterion
                        .as_ref()
                        .filter(|value| !python::is_blank(value))
                        .map(python::to_text)
                        .unwrap_or_else(|| "yes, the statement holds".to_string())
                ),
            ],
        }
    }

    /// `score` legend 所用的原始分数等级；其他类型返回空。参考实现原样拷贝
    /// criteria（`{str(i): c for ...}`），因此这里返回未渲染的原始值而非字符串。
    pub(crate) fn legend(&self) -> Vec<Value> {
        match &self.criteria {
            Criteria::Score(items) => items.clone(),
            _ => Vec::new(),
        }
    }
}

/// 问题原语。其判别值同时用作类型 embedding 的索引。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum QuestionType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QuestionType {
    /// 原语数量；决策头为每种类型学一个 embedding。
    pub(crate) const COUNT: i32 = 3;

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Criteria {
    /// 标签 -> 可选描述（保序）。
    Choice(Vec<(String, Option<Value>)>),
    /// 有序的分数等级。
    Score(Vec<Value>),
    /// 布尔判定；`false` / `true` 为可选描述。
    Noul {
        false_criterion: Option<Value>,
        true_criterion: Option<Value>,
    },
}

/// 参考实现在把 criteria 与 state 嵌入 prompt 时所依赖的 Python 值格式化语义。
mod python {
    use serde_json::Value;

    /// `value is None or value == ""`（`render_options` 的空描述判定）。
    pub(super) fn is_blank(value: &Value) -> bool {
        value.is_null() || matches!(value, Value::String(text) if text.is_empty())
    }

    /// 把一个 JSON 取值转成 Python 会嵌入 prompt 的文本，对应
    /// `common.py::render_criterion`：字符串原样透传，其它值（含数字、布尔、
    /// 数组、对象）走 `json.dumps(value, ensure_ascii=False)`。
    pub(super) fn to_text(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            other => json_dumps(other),
        }
    }

    /// Python `json.dumps(value, ensure_ascii=False)`：双引号、`", "`/`": "`。
    pub(super) fn json_dumps(value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(boolean) => boolean.to_string(),
            Value::Number(number) => number.to_string(),
            Value::String(text) => {
                serde_json::to_string(text).unwrap_or_else(|_| format!("\"{text}\""))
            }
            Value::Array(items) => {
                let inner: Vec<String> = items.iter().map(json_dumps).collect();
                format!("[{}]", inner.join(", "))
            }
            Value::Object(map) => format!(
                "{{{}}}",
                map.iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\"")),
                        json_dumps(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
