//! HN 关键字检索示例：抓取 HackerNews 新帖，用 BM25 粗筛后交给 Laya 判定相关性。

mod hn;
mod search;

use std::env;

use rustlaya::{DEFAULT_REPOSITORY, Laya};

use crate::hn::fetch_stories;
use crate::search::search;

/// 未提供关键字参数时的默认查询词。
const DEFAULT_KEYWORD: &str = "jev model";
/// 相关性分数阈值（0-3 量纲），低于该值的 story 会被丢弃。
const MIN_SCORE: f64 = 1.6;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = env::args().skip(1);
    let mut compact = false;
    let mut keyword: Option<String> = None;
    for arg in args {
        if arg == "--compact" {
            compact = true;
        } else {
            keyword = Some(arg);
        }
    }
    let keyword = keyword.unwrap_or_else(|| DEFAULT_KEYWORD.to_string());

    eprintln!("fetching HackerNews new stories");
    let stories = fetch_stories().await?;
    eprintln!("fetched {} stories", stories.len());

    let model = env::var("LAYA_MODEL").unwrap_or_else(|_| DEFAULT_REPOSITORY.to_string());
    eprintln!("loading Laya model {model}");
    let laya = Laya::load(&model)?;
    eprintln!("Laya model is loaded");

    let hits = search(&laya, &stories, &keyword, MIN_SCORE)?;
    for (rank, (index, score)) in hits.into_iter().enumerate() {
        let content = if compact {
            stories[index].title.as_deref().unwrap_or_default()
        } else {
            &serde_json::to_string_pretty(&stories[index])?
        };
        println!("{:02}: [{:.3}] {}", rank + 1, score, content);
    }

    Ok(())
}
