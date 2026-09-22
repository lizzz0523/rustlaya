//! 基于 [candle] + Metal 后端的 Laya 类型化决策推理。
//!
//! 输入一个状态和一组类型化问题（`choice`、`score`、`noul`），一次前向传播
//! 即可给出经过校准的答案。
//!
//! crate 按单一依赖方向拆分：
//!
//! - [`encode`] —— 输入侧：类型化问题、Python 兼容渲染与分词，不含张量后端。
//! - [`model`] —— candle 网络与检查点加载。
//! - [`decode`] —— 输出侧：校准数学与类型化结果文档，不含张量后端。
//!
//! 此外，crate 根持有 agent 配置 `InferenceConfig`（`rl_agent_config.json`），
//! 供门面装配与 decode 校准共用。
//!
//! [`Laya`] 是唯一入口，负责把三者串联起来。
//!
//! [candle]: https://github.com/huggingface/candle

mod decode;
mod encode;
mod hub;
mod model;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, anyhow};
use candle_core::Device;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use candle_core::DType;
pub use decode::{ActionAnswer, Answer, Response, Usage};
pub use encode::{Criteria, Question, QuestionType};
pub use hub::DEFAULT_REPOSITORY;

use encode::{SpecialTokens, TokenizerConfig};
use model::{Batch, DecisionModel, EncoderConfig};

/// 已加载的分词器、配置与原生 candle 模型。
pub struct Laya {
    tokenizer: Tokenizer,
    special_tokens: SpecialTokens,
    config: InferenceConfig,
    model: DecisionModel,
    device: Device,
}

impl Laya {
    /// 把检查点加载到 Metal 设备，精度取自检查点声明的 `amp_dtype`。
    ///
    /// `model` 可以是本地目录，也可以是诸如 [`hub::DEFAULT_REPOSITORY`] 的
    /// Hugging Face 仓库 id。
    pub fn load(model: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::load_inner(model.as_ref(), None)
    }

    /// 同 [`Laya::load`]，但可显式覆盖参数精度。
    pub fn load_with_dtype(model: impl AsRef<str>, dtype: DType) -> anyhow::Result<Self> {
        Self::load_inner(model.as_ref(), Some(dtype))
    }

    fn load_inner(model: &str, dtype: Option<DType>) -> anyhow::Result<Self> {
        let device = Device::new_metal(0).context("initialising the Metal device")?;
        let model_paths = hub::resolve_model(model)?;

        let config: InferenceConfig = read_json(&model_paths.agent_config)?;
        let encoder_config: EncoderConfig = read_json(&model_paths.encoder_config)?;
        let tokenizer_config: TokenizerConfig = read_json(&model_paths.tokenizer_config)?;

        let mut tokenizer = Tokenizer::from_file(&model_paths.tokenizer)
            .map_err(|error| anyhow!("loading {}: {error}", model_paths.tokenizer.display()))?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(|error| anyhow!("disabling truncation: {error}"))?;
        let special_tokens = SpecialTokens::resolve(&tokenizer, &tokenizer_config)?;

        let dtype = dtype.unwrap_or_else(|| dtype_from(&config.amp_dtype));
        let var_builder = model::load_var_builder(&model_paths.weights, dtype, &device)?;
        let model = DecisionModel::load(
            var_builder,
            encoder_config,
            config.head_layers,
            config.num_actions(),
        )?;

        Ok(Self {
            tokenizer,
            special_tokens,
            config,
            model,
            device,
        })
    }

    /// 对同一个状态批量运行一组类型化问题。
    pub fn predict(&self, state: &Value, questions: &[Question]) -> anyhow::Result<Response> {
        let sequences = encode::build_sequences(
            &self.tokenizer,
            &self.special_tokens,
            state,
            questions,
            self.config.max_len,
            self.config.head_max_len,
        )?;

        let batch = Batch::collate(&sequences, self.special_tokens.pad_token_id, &self.device)?;
        let (logits, action_logits) = self.model.forward(&batch)?;
        let logits = to_host_f32(&logits)?;
        let action_logits = to_host_f32(&action_logits)?;

        Ok(decode::build_response(
            questions,
            &sequences,
            &logits,
            &action_logits,
            batch.num_markers,
            batch.input_token_count,
            self.config.num_actions(),
            &self.config.temperature,
            &self.config.temperature_by_options,
        ))
    }
}

/// `rl_agent_config.json`。
#[derive(Clone, Debug, Deserialize)]
struct InferenceConfig {
    head_layers: usize,
    max_len: usize,
    head_max_len: usize,
    #[serde(default = "default_precision")]
    amp_dtype: String,
    #[serde(default)]
    temperature: Vec<f32>,
    #[serde(default)]
    temperature_by_options: HashMap<String, f32>,
    /// 命名的 RL 动作；只需其数量决定动作头的输出维度（索引 0 是空操作的 answer）。
    #[serde(default)]
    act_costs: IndexMap<String, f32>,
}

impl InferenceConfig {
    /// 动作头输出的 logits 数量。
    fn num_actions(&self) -> usize {
        self.act_costs.len() + 1
    }
}

fn default_precision() -> String {
    "fp16".to_string()
}

/// 读取并反序列化一个 JSON 文件，出错时补上文件路径。
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let contents =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parsing {}", path.display()))
}

/// 把张量拷贝到主机端并转为 float32。
fn to_host_f32(tensor: &candle_core::Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

/// 选择 Metal 计算精度：除检查点显式声明 `fp32` 外一律用 F16。F16 在 Metal 上
/// 有原生加速，检查点本身也以 F16 存储，且（与 BF16 不同）无需逐算子模拟。
fn dtype_from(amp_dtype: &str) -> DType {
    match amp_dtype.to_ascii_lowercase().as_str() {
        "fp32" | "float32" | "f32" => DType::F32,
        _ => DType::F16,
    }
}
