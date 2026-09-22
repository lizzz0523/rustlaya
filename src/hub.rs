//! 从本地目录或 Hugging Face Hub 定位模型文件。

use std::env;
use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::{HFClientSync, split_id};

/// 英文 Laya 检查点的官方发布仓库。
pub const DEFAULT_REPOSITORY: &str = "convaiinnovations/laya";
/// 固定的检查点版本号。
const DEFAULT_REVISION: &str = "c5d78730f3493e4fe16d61507ef4b78eef7318cf";

/// 若 `model` 是已存在的目录则直接使用；否则按 Hugging Face 仓库 id 处理，
/// 下载（或复用缓存）所需文件。
///
/// `LAYA_REVISION` 可覆盖版本号；官方仓库缺省用固定 commit，其他仓库缺省用 `main`。
pub(crate) fn resolve_model(model: &str) -> anyhow::Result<ModelPaths> {
    if Path::new(model).is_dir() {
        return Ok(ModelPaths::from_directory(Path::new(model)));
    }

    let (owner, name) = split_id(model);
    let client = HFClientSync::new().context("creating Hugging Face client")?;
    let repos = client.model(owner, name);

    // 固定版本号只适用于官方英文仓库；其他仓库（如多语言仓库）跟随默认分支。
    let default_revision = if model == DEFAULT_REPOSITORY {
        DEFAULT_REVISION
    } else {
        "main"
    };
    let revision = env::var("LAYA_REVISION").unwrap_or_else(|_| default_revision.to_string());
    // 只取仓库根目录的检查点文件：官方仓库根目录是英文版，跳过 multilingual/、
    // typed-decisions/、assets/；独立的多语言镜像仓库根目录即为所需文件。
    let snapshot_directory = repos
        .snapshot_download()
        .revision(revision.clone())
        .allow_patterns(
            [
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/*",
                "tokenizer/*",
            ]
            .iter()
            .map(|pattern| pattern.to_string())
            .collect(),
        )
        .send()
        .with_context(|| format!("downloading {model}@{revision}"))?;

    Ok(ModelPaths::from_directory(&snapshot_directory))
}

/// 模型所需文件的绝对路径，与其实际存放位置无关。
pub(crate) struct ModelPaths {
    pub(crate) weights: PathBuf,
    pub(crate) agent_config: PathBuf,
    pub(crate) encoder_config: PathBuf,
    pub(crate) tokenizer: PathBuf,
    pub(crate) tokenizer_config: PathBuf,
}

impl ModelPaths {
    fn from_directory(directory: &Path) -> Self {
        Self {
            weights: directory.join("model.safetensors"),
            agent_config: directory.join("rl_agent_config.json"),
            encoder_config: directory.join("encoder/config.json"),
            tokenizer: directory.join("tokenizer/tokenizer.json"),
            tokenizer_config: directory.join("tokenizer/tokenizer_config.json"),
        }
    }
}
