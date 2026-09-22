//! Laya 决策模型的原生 candle 实现。
//!
//! 由 ModernBERT-large 编码器，加上 Laya 的决策 transformer、打分头与动作头
//! 组成。移植自 `laya_mlx/model.py`。本模块**只输出原始 logits**，校准逻辑在
//! [`crate::decode`]。
//!
//! 检查点的命名布局完全由 [`load_var_builder`] / [`checkpoint_name`] 处理，
//! 因此下面的网络代码使用干净的、与检查点无关的参数名。

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops::{sdpa, softmax_last_dim};
use candle_nn::rotary_emb::rope;
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, layer_norm_no_bias,
    linear, linear_no_bias,
};

use crate::encode::{QuestionType, Sequence};

/// 被屏蔽位置所用的加性掩码值。取有限值，使其在转换为 f16/bf16 后（会饱和为
/// -inf）仍能让 `exp` 精确得到零。
const BLOCKED: f64 = -1e9;
/// 为选项 marker 打分时用于屏蔽的占位 logit。
const MASKED_LOGIT: f32 = -1e4;

// ---------------------------------------------------------------------------
// 模型
// ---------------------------------------------------------------------------

pub(crate) struct DecisionModel {
    encoder: ModernBert,
    question_type_embedding: Embedding,
    decision_head: DecisionHead,
    scorer: Scorer,
    action_head: ActionHead,
}

impl DecisionModel {
    /// 用检查点支撑的 [`VarBuilder`] 构建网络。
    pub(crate) fn load(
        var_builder: VarBuilder,
        encoder_config: EncoderConfig,
        head_layers: usize,
        num_actions: usize,
    ) -> anyhow::Result<Self> {
        let encoder = ModernBert::load(var_builder.pp("encoder"), &encoder_config)?;

        let question_type_embedding = embedding(
            QuestionType::COUNT as usize,
            encoder_config.hidden_size,
            var_builder.pp("type_emb"),
        )?;

        let decision_head =
            DecisionHead::load(var_builder.pp("head"), &encoder_config, head_layers)?;

        let scorer = Scorer::load(var_builder.pp("scorer"), &encoder_config)?;

        let action_head =
            ActionHead::load(var_builder.pp("act_head"), &encoder_config, num_actions)?;

        Ok(Self {
            encoder,
            question_type_embedding,
            decision_head,
            scorer,
            action_head,
        })
    }

    /// 执行一次前向传播。
    ///
    /// 返回 `(option_logits [batch, options], action_logits [batch, actions])`，
    /// 均为 float32 且为行主序。
    pub(crate) fn forward(&self, batch: &Batch) -> anyhow::Result<(Tensor, Tensor)> {
        let hidden = self.encoder_forward(batch)?;
        let (logits, probabilities, option_count) = self.score_markers(&hidden, batch)?;
        let action_logits = self.action_logits(&hidden, &probabilities, &option_count)?;
        Ok((logits, action_logits))
    }

    /// 编码器、问题类型 embedding 与决策头：`[batch, seq, hidden]`。
    fn encoder_forward(&self, batch: &Batch) -> anyhow::Result<Tensor> {
        let hidden = self
            .encoder
            .forward(&batch.input_token_ids, &batch.attention)?;
        let (batch_size, sequence_length) = (hidden.dim(0)?, hidden.dim(1)?);

        let type_embedding = self
            .question_type_embedding
            .forward(&batch.question_type)?
            .unsqueeze(1)?;
        let hidden = hidden.broadcast_add(&type_embedding)?;

        // 决策头只屏蔽 padding 的 key。
        let real_key =
            batch
                .attention
                .to_dtype(DType::F32)?
                .reshape((batch_size, 1, 1, sequence_length))?;
        let mask = to_additive(&real_key, hidden.dtype())?;
        let hidden = self.decision_head.forward(&hidden, &mask)?.contiguous()?;

        Ok(hidden)
    }

    /// 为选项 marker 打分。
    ///
    /// 返回原始的 `[batch, options]` logits（f32）、其掩码 softmax 分布，以及
    /// `[batch, 1]` 的真实选项数（下限为二）。
    fn score_markers(
        &self,
        hidden: &Tensor,
        batch: &Batch,
    ) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, num_markers) = batch.marker_positions.dims2()?;
        let positions = batch
            .marker_positions
            .unsqueeze(2)?
            .broadcast_as((batch_size, num_markers, hidden.dim(2)?))?
            .contiguous()?;
        let markers = hidden.gather(&positions, 1)?;
        let logits = self
            .scorer
            .forward(&markers)?
            .squeeze(2)?
            .to_dtype(DType::F32)?;

        let mask = batch.marker_mask.to_dtype(DType::U8)?;
        let blocked = Tensor::full(MASKED_LOGIT, mask.shape(), logits.device())?;
        let probabilities = softmax_last_dim(&mask.where_cond(&logits, &blocked)?)?;

        let option_count = mask
            .to_dtype(DType::F32)?
            .sum_keepdim(D::Minus1)?
            .maximum(2.0f64)?;

        Ok((logits, probabilities, option_count))
    }

    /// 动作头的 logits，由 CLS 状态与选项分布得到。
    fn action_logits(
        &self,
        hidden: &Tensor,
        probabilities: &Tensor,
        option_count: &Tensor,
    ) -> anyhow::Result<Tensor> {
        let entropy = probabilities
            .clamp(1e-9f64, 1.0f64)?
            .log()?
            .mul(probabilities)?
            .sum_keepdim(D::Minus1)?
            .neg()?
            .broadcast_div(&option_count.log()?)?;

        // 通过升序排序取 top-2 概率：最大的在最后。
        let num_options = probabilities.dim(1)?;
        let (sorted, _) = probabilities.contiguous()?.sort_last_dim(true)?;
        let top_probability = sorted.narrow(D::Minus1, num_options - 1, 1)?;
        let second_probability = if num_options >= 2 {
            sorted.narrow(D::Minus1, num_options - 2, 1)?
        } else {
            top_probability.clone()
        };

        let features = Tensor::cat(
            &[
                &top_probability,
                &top_probability.sub(&second_probability)?,
                &entropy,
                &option_count.affine(1.0 / 255.0, 0.0)?,
            ],
            D::Minus1,
        )?;

        // 动作头与模型参数使用相同精度。
        let pooled = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let combined = Tensor::cat(&[&pooled, &features.to_dtype(hidden.dtype())?], D::Minus1)?;
        let action_logits = self.action_head.forward(&combined)?.to_dtype(DType::F32)?;

        Ok(action_logits)
    }
}

// ---------------------------------------------------------------------------
// 编码器
// ---------------------------------------------------------------------------

struct ModernBert {
    embeddings: Embeddings,
    local_attention: usize,
    layers: Vec<EncoderLayer>,
    final_norm: LayerNorm,
}

impl ModernBert {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        let embeddings = Embeddings::load(var_builder.pp("embeddings"), config)?;

        let global_rotary = Rotary::new(
            var_builder.dtype(),
            config,
            config.rope_base(true),
            var_builder.device(),
        )?;
        let local_rotary = Rotary::new(
            var_builder.dtype(),
            config,
            config.rope_base(false),
            var_builder.device(),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let is_global = config.layer_is_global(index);
            let rotary = if is_global {
                global_rotary.clone()
            } else {
                local_rotary.clone()
            };
            layers.push(EncoderLayer::load(
                var_builder.pp(format!("layers.{index}")),
                config,
                rotary,
                is_global,
            )?);
        }

        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            var_builder.pp("final_norm"),
        )?;

        Ok(Self {
            embeddings,
            local_attention: config.local_attention,
            layers,
            final_norm,
        })
    }

    fn forward(&self, token_ids: &Tensor, valid: &Tensor) -> anyhow::Result<Tensor> {
        let mut hidden = self.embeddings.forward(token_ids)?;
        let (global_mask, sliding_mask) =
            build_attention_masks(valid, self.local_attention, hidden.dtype())?;

        for layer in &self.layers {
            let mask = if layer.is_global {
                &global_mask
            } else {
                &sliding_mask
            };
            hidden = layer.forward(&hidden, mask)?;
        }

        Ok(self.final_norm.forward(&hidden)?)
    }
}

// ---------------------------------------------------------------------------
// 决策头、打分头与动作头
// ---------------------------------------------------------------------------

struct DecisionHead {
    layers: Vec<HeadLayer>,
}

impl DecisionHead {
    fn load(
        var_builder: VarBuilder,
        config: &EncoderConfig,
        head_layers: usize,
    ) -> anyhow::Result<Self> {
        let mut layers = Vec::with_capacity(head_layers);
        for index in 0..head_layers {
            layers.push(HeadLayer::load(
                var_builder.pp(format!("layers.{index}")),
                config,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, mask: &Tensor) -> anyhow::Result<Tensor> {
        let mut hidden = hidden_states.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask)?;
        }
        Ok(hidden)
    }
}

struct HeadLayer {
    self_attention: HeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl HeadLayer {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        let head_hidden_size = 4 * config.hidden_size;
        Ok(Self {
            self_attention: HeadAttention::load(var_builder.pp("self_attn"), config)?,
            norm1: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("norm1"),
            )?,
            norm2: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("norm2"),
            )?,
            linear1: linear(
                config.hidden_size,
                head_hidden_size,
                var_builder.pp("linear1"),
            )?,
            linear2: linear(
                head_hidden_size,
                config.hidden_size,
                var_builder.pp("linear2"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, mask: &Tensor) -> anyhow::Result<Tensor> {
        let normalized = self.norm1.forward(hidden_states)?;
        let attention_output = self.self_attention.forward(&normalized, mask)?;
        let hidden = hidden_states.add(&attention_output)?;
        let normalized = self.norm2.forward(&hidden)?;
        let feed_forward = self
            .linear2
            .forward(&self.linear1.forward(&normalized)?.relu()?)?;
        Ok(hidden.add(&feed_forward)?)
    }
}

struct HeadAttention {
    input_projection: Linear,
    output_projection: Linear,
    num_heads: usize,
    head_size: usize,
}

impl HeadAttention {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        Ok(Self {
            input_projection: linear(
                config.hidden_size,
                3 * config.hidden_size,
                var_builder.pp("in_proj"),
            )?,
            output_projection: linear(
                config.hidden_size,
                config.hidden_size,
                var_builder.pp("out_proj"),
            )?,
            num_heads: config.num_attention_heads,
            head_size: config.head_size(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, mask: &Tensor) -> anyhow::Result<Tensor> {
        let (query, key, value) = split_query_key_value(
            &self.input_projection.forward(hidden_states)?,
            self.num_heads,
            self.head_size,
        )?;
        let scale = (self.head_size as f64).powf(-0.5);
        let context = scaled_dot_product_attention(&query, &key, &value, mask, scale)?;
        Ok(self.output_projection.forward(&merge_heads(
            &context,
            self.num_heads,
            self.head_size,
        )?)?)
    }
}

struct Scorer {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl Scorer {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        Ok(Self {
            norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("norm"),
            )?,
            linear1: linear(
                config.hidden_size,
                config.hidden_size,
                var_builder.pp("linear1"),
            )?,
            linear2: linear(config.hidden_size, 1, var_builder.pp("linear2"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> anyhow::Result<Tensor> {
        let hidden = self.norm.forward(hidden_states)?;
        let hidden = self.linear1.forward(&hidden)?.gelu_erf()?;
        Ok(self.linear2.forward(&hidden)?)
    }
}

struct ActionHead {
    linear1: Linear,
    linear2: Linear,
}

impl ActionHead {
    fn load(
        var_builder: VarBuilder,
        config: &EncoderConfig,
        num_actions: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            linear1: linear(config.hidden_size + 4, 256, var_builder.pp("linear1"))?,
            linear2: linear(256, num_actions, var_builder.pp("linear2"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> anyhow::Result<Tensor> {
        let hidden = self.linear1.forward(hidden_states)?.gelu_erf()?;
        Ok(self.linear2.forward(&hidden)?)
    }
}

// ---------------------------------------------------------------------------
// 编码器层
// ---------------------------------------------------------------------------

struct EncoderLayer {
    attention_norm: Option<LayerNorm>,
    attention: EncoderAttention,
    mlp_norm: LayerNorm,
    mlp: EncoderMlp,
    /// 是否使用全（全局）注意力（而非滑动窗口）；决定 forward 时选用哪种掩码。
    is_global: bool,
}

impl EncoderLayer {
    fn load(
        var_builder: VarBuilder,
        config: &EncoderConfig,
        rotary: Rotary,
        is_global: bool,
    ) -> anyhow::Result<Self> {
        // 检查点中第 0 层没有 attention norm。
        let attention_norm = if var_builder.contains_tensor("attn_norm.weight") {
            Some(layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("attn_norm"),
            )?)
        } else {
            None
        };
        Ok(Self {
            attention_norm,
            attention: EncoderAttention::load(var_builder.pp("attn"), config, rotary)?,
            mlp_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("mlp_norm"),
            )?,
            mlp: EncoderMlp::load(var_builder.pp("mlp"), config)?,
            is_global,
        })
    }

    fn forward(&self, hidden_states: &Tensor, mask: &Tensor) -> anyhow::Result<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => norm.forward(hidden_states)?,
            None => hidden_states.clone(),
        };
        let attention_output = self.attention.forward(&normalized, mask)?;
        let hidden = hidden_states.add(&attention_output)?;
        let mlp_output = self.mlp.forward(&self.mlp_norm.forward(&hidden)?)?;
        Ok(hidden.add(&mlp_output)?)
    }
}

struct EncoderAttention {
    query_key_value: Linear,
    output: Linear,
    num_heads: usize,
    head_size: usize,
    rotary: Rotary,
}

impl EncoderAttention {
    fn load(
        var_builder: VarBuilder,
        config: &EncoderConfig,
        rotary: Rotary,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            query_key_value: linear_no_bias(
                config.hidden_size,
                3 * config.hidden_size,
                var_builder.pp("Wqkv"),
            )?,
            output: linear_no_bias(config.hidden_size, config.hidden_size, var_builder.pp("Wo"))?,
            num_heads: config.num_attention_heads,
            head_size: config.head_size(),
            rotary,
        })
    }

    fn forward(&self, hidden_states: &Tensor, mask: &Tensor) -> anyhow::Result<Tensor> {
        let (query, key, value) = split_query_key_value(
            &self.query_key_value.forward(hidden_states)?,
            self.num_heads,
            self.head_size,
        )?;
        let query = self.rotary.apply(&query)?;
        let key = self.rotary.apply(&key)?;

        let scale = (self.head_size as f64).powf(-0.5);
        let context = scaled_dot_product_attention(&query, &key, &value, mask, scale)?;
        Ok(self
            .output
            .forward(&merge_heads(&context, self.num_heads, self.head_size)?)?)
    }
}

/// 带门控的 MLP（`GeGLU`）：`Wo(gelu(Wi[..., :m]) * Wi[..., m:])`。
struct EncoderMlp {
    input: Linear,
    output: Linear,
}

impl EncoderMlp {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        Ok(Self {
            input: linear_no_bias(
                config.hidden_size,
                2 * config.intermediate_size,
                var_builder.pp("Wi"),
            )?,
            output: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                var_builder.pp("Wo"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> anyhow::Result<Tensor> {
        let projected = self.input.forward(hidden_states)?;
        let parts = projected.chunk(2, D::Minus1)?;
        let gate = parts[0].gelu_erf()?;
        Ok(self.output.forward(&gate.mul(&parts[1])?)?)
    }
}

struct Embeddings {
    tok_embeddings: Embedding,
    norm: LayerNorm,
}

impl Embeddings {
    fn load(var_builder: VarBuilder, config: &EncoderConfig) -> anyhow::Result<Self> {
        Ok(Self {
            tok_embeddings: embedding(
                config.vocab_size,
                config.hidden_size,
                var_builder.pp("tok_embeddings"),
            )?,
            norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                var_builder.pp("norm"),
            )?,
        })
    }

    fn forward(&self, token_ids: &Tensor) -> anyhow::Result<Tensor> {
        let embedded = self.tok_embeddings.forward(token_ids)?;
        Ok(self.norm.forward(&embedded)?)
    }
}

/// 预先计算好的旋转位置 embedding，供同一类（全局或滑动窗口）的每个注意力层共享。
#[derive(Clone)]
struct Rotary {
    sine: Tensor,
    cosine: Tensor,
}

impl Rotary {
    fn new(
        dtype: DType,
        config: &EncoderConfig,
        theta: f32,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let head_size = config.head_size();
        let inverse_frequencies: Vec<f32> = (0..head_size)
            .step_by(2)
            .map(|index| 1f32 / theta.powf(index as f32 / head_size as f32))
            .collect();
        let frequency_count = inverse_frequencies.len();
        let inverse_frequencies =
            Tensor::from_vec(inverse_frequencies, (1, frequency_count), device)?.to_dtype(dtype)?;
        let max_sequence_length = config.max_position_embeddings;
        let positions = Tensor::arange(0u32, max_sequence_length as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_sequence_length, 1))?;
        let frequencies = positions.matmul(&inverse_frequencies)?;
        Ok(Self {
            sine: frequencies.sin()?,
            cosine: frequencies.cos()?,
        })
    }

    /// 对 `[batch, heads, seq, head_size]` 张量施加旋转位置 embedding。
    fn apply(&self, hidden_states: &Tensor) -> anyhow::Result<Tensor> {
        Ok(rope(&hidden_states.contiguous()?, &self.cosine, &self.sine)?.contiguous()?)
    }
}

// ---------------------------------------------------------------------------
// 共享注意力原语
// ---------------------------------------------------------------------------

/// 把 `[batch, seq, 3 * hidden]` 的投影拆成 query/key/value，各自为连续的
/// `[batch, heads, seq, head_size]` 形状。
fn split_query_key_value(
    projection: &Tensor,
    num_heads: usize,
    head_size: usize,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let (batch_size, sequence_length) = (projection.dim(0)?, projection.dim(1)?);
    let parts = projection
        .reshape((batch_size, sequence_length, 3, num_heads, head_size))?
        .chunk(3, 2)?;
    let to_head_layout = |tensor: &Tensor| -> anyhow::Result<Tensor> {
        Ok(tensor.squeeze(2)?.permute((0, 2, 1, 3))?.contiguous()?)
    };
    Ok((
        to_head_layout(&parts[0])?,
        to_head_layout(&parts[1])?,
        to_head_layout(&parts[2])?,
    ))
}

/// [`split_query_key_value`] 的逆操作：把 `[batch, heads, seq, head_size]`
/// 合并回 `[batch, seq, hidden]`。
fn merge_heads(projected: &Tensor, num_heads: usize, head_size: usize) -> anyhow::Result<Tensor> {
    let (batch_size, _, sequence_length, _) = projected.dims4()?;
    Ok(projected.permute((0, 2, 1, 3))?.reshape((
        batch_size,
        sequence_length,
        num_heads * head_size,
    ))?)
}

/// 对 `[batch, heads, seq, head_size]` 张量计算
/// `softmax(query keyᵀ * scale + mask) value`，使用 candle 的融合 SDPA kernel。
/// `mask` 为加性掩码，可广播到 `[batch, heads, seq, seq]`。
fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: &Tensor,
    scale: f64,
) -> anyhow::Result<Tensor> {
    let (batch_size, num_heads, sequence_length, _) = query.dims4()?;
    let mask = mask.broadcast_as((batch_size, num_heads, sequence_length, sequence_length))?;
    Ok(sdpa(
        query,
        key,
        value,
        Some(&mask),
        false,
        scale as f32,
        1.0,
    )?)
}

/// 构建加性注意力掩码：
/// - `global`：真实 key 为 `0`，padding key 为 `BLOCKED`。
/// - `sliding`：滑动窗口（`|i - j| <= local_attention / 2`），与 `global` 合并，
///   并对 padding query 放宽，使没有整行被完全屏蔽。
fn build_attention_masks(
    valid: &Tensor,
    local_attention: usize,
    dtype: DType,
) -> anyhow::Result<(Tensor, Tensor)> {
    let (batch_size, sequence_length) = valid.dims2()?;
    let device = valid.device();

    // 真实 key 为 `1`，padding 为 `0`。
    let real_key = valid
        .to_dtype(DType::F32)?
        .reshape((batch_size, 1, 1, sequence_length))?;
    let global = to_additive(&real_key, dtype)?;

    // 当 `|i - j| <= local_attention / 2` 即位于滑动窗口内时为 `1`。
    let half_window = Tensor::new(local_attention as f32 / 2.0, device)?;
    let positions = Tensor::arange(0u32, sequence_length as u32, device)?.to_dtype(DType::F32)?;
    let distance = positions
        .reshape((sequence_length, 1))?
        .broadcast_sub(&positions.reshape((1, sequence_length))?)?
        .abs()?;
    let in_window = distance
        .broadcast_le(&half_window)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .unsqueeze(0)?;

    // padding query 可以关注任意真实 key，从而没有整行被完全屏蔽。
    // `[batch, 1, seq, 1]`：padding query 位置为 1，真实位置为 0。
    let padded_query = real_key
        .reshape((batch_size, 1, sequence_length, 1))?
        .affine(-1.0, 1.0)?;
    let allowed = in_window.broadcast_maximum(&padded_query)?;
    let keep_mask = allowed.broadcast_mul(&real_key)?;
    let sliding = to_additive(&keep_mask, dtype)?;

    Ok((global, sliding))
}

/// 把 `0`/`1` 的保留掩码转换为加性掩码：保留处为 `0`，屏蔽处为 `BLOCKED`。
fn to_additive(keep_mask: &Tensor, dtype: DType) -> anyhow::Result<Tensor> {
    Ok(keep_mask
        .affine(-1.0, 1.0)?
        .affine(BLOCKED, 0.0)?
        .to_dtype(dtype)?)
}

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

/// 构建编码器所需的 `encoder/config.json` 子集。
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct EncoderConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    local_attention: usize,
    #[serde(rename = "global_attn_every_n_layers")]
    global_attention_every_n_layers: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_layer_norm_epsilon")]
    layer_norm_eps: f64,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

impl EncoderConfig {
    /// 第 `index` 层是否使用全（全局）注意力，而非滑动窗口注意力。
    fn layer_is_global(&self, index: usize) -> bool {
        if let Some(layer_type) = self.layer_types.as_ref().and_then(|types| types.get(index)) {
            return layer_type == "full_attention";
        }
        index.is_multiple_of(self.global_attention_every_n_layers)
    }

    /// 全局或滑动窗口层的 RoPE base。
    fn rope_base(&self, is_global: bool) -> f32 {
        if let Some(parameters) = &self.rope_parameters {
            return if is_global {
                parameters.full_attention.rope_theta
            } else {
                parameters.sliding_attention.rope_theta
            };
        }
        if is_global { 160000.0 } else { 10000.0 }
    }

    fn head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RopeParameters {
    full_attention: RopeTheta,
    sliding_attention: RopeTheta,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RopeTheta {
    rope_theta: f32,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

fn default_max_position_embeddings() -> usize {
    8192
}

// ---------------------------------------------------------------------------
// 批处理
// ---------------------------------------------------------------------------

/// 一次前向传播所需的 padding 后张量。
pub(crate) struct Batch {
    pub(crate) input_token_ids: Tensor,
    pub(crate) attention: Tensor,
    pub(crate) marker_positions: Tensor,
    pub(crate) marker_mask: Tensor,
    pub(crate) question_type: Tensor,
    pub(crate) num_markers: usize,
    pub(crate) input_token_count: i32,
}

impl Batch {
    /// 把一批分词后的序列 padding 到最长序列 / 最多 marker 数。
    pub(crate) fn collate(
        sequences: &[Sequence],
        pad_token_id: u32,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let batch_size = sequences.len();
        let sequence_length = sequences
            .iter()
            .map(|sequence| sequence.token_ids.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let num_markers = sequences
            .iter()
            .map(|sequence| sequence.markers.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut input_token_ids = vec![pad_token_id; batch_size * sequence_length];
        let mut attention = vec![0u32; batch_size * sequence_length];
        let mut marker_positions = vec![0u32; batch_size * num_markers];
        let mut marker_mask = vec![0u8; batch_size * num_markers];
        let mut question_type = vec![0u32; batch_size];

        for (row, sequence) in sequences.iter().enumerate() {
            let row_offset = row * sequence_length;
            for (column, &token_id) in sequence.token_ids.iter().enumerate() {
                input_token_ids[row_offset + column] = token_id as u32;
                attention[row_offset + column] = 1;
            }
            let marker_offset = row * num_markers;
            for (column, &position) in sequence.markers.iter().enumerate() {
                marker_positions[marker_offset + column] = position as u32;
                marker_mask[marker_offset + column] = 1;
            }
            question_type[row] = sequence.question_type as u32;
        }

        let input_token_count = attention.iter().sum::<u32>() as i32;
        Ok(Self {
            input_token_ids: Tensor::from_vec(
                input_token_ids,
                (batch_size, sequence_length),
                device,
            )?,
            attention: Tensor::from_vec(attention, (batch_size, sequence_length), device)?,
            marker_positions: Tensor::from_vec(
                marker_positions,
                (batch_size, num_markers),
                device,
            )?,
            marker_mask: Tensor::from_vec(marker_mask, (batch_size, num_markers), device)?,
            question_type: Tensor::from_vec(question_type, (batch_size,), device)?,
            num_markers,
            input_token_count,
        })
    }
}

// ---------------------------------------------------------------------------
// 检查点加载
// ---------------------------------------------------------------------------

/// 以内存映射方式打开 `model.safetensors`，并通过一个以本模块参数路径命名的
/// [`VarBuilder`] 暴露它。
///
/// 只有真正被请求的张量才会被物化，检查点命名由 [`checkpoint_name`] 惰性解析，
/// 因此上面的网络保持与检查点无关。未使用的张量（如 `temperature`）永不加载。
pub(crate) fn load_var_builder(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<VarBuilder<'static>> {
    // SAFETY：在映射存活期间，检查点文件是只读的。
    let var_builder = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    Ok(var_builder.rename_f(checkpoint_name))
}

/// 把模块参数路径翻译为其在官方检查点中的名字。
///
/// 编码器已经使用检查点的名字；只有决策头、打分头与动作头使用 PyTorch 的顺序
/// 索引 / 合并 QKV 的布局。
fn checkpoint_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("head.layers.") {
        let mapped = rest
            .replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
            .replace(".self_attn.in_proj.bias", ".self_attn.in_proj_bias");
        return format!("head.layers.{mapped}");
    }
    for (module, checkpoint) in [
        ("scorer.norm.", "scorer.0."),
        ("scorer.linear1.", "scorer.1."),
        ("scorer.linear2.", "scorer.3."),
        ("act_head.linear1.", "act_head.0."),
        ("act_head.linear2.", "act_head.2."),
    ] {
        if let Some(rest) = name.strip_prefix(module) {
            return format!("{checkpoint}{rest}");
        }
    }
    name.to_string()
}
