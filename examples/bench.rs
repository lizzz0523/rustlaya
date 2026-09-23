//! 批量收益基准：量化同一状态下多个问题在一次前向中的耗时缩放。
//!
//! 覆盖两类用例：
//! - 等长问题（batch = 1/2/4/8/16/32），看总耗时是否随问题数线性增长；
//! - 长短混合问题（16 短 + 16 长），量化 `Batch::collate` 按批内最长序列
//!   padding 造成的浪费。
//!
//! 运行：
//! ```bash
//! cargo run --release --example bench
//! BENCH_REPS=10 cargo run --release --example bench
//! ```

use std::env;
use std::time::{Duration, Instant};

use rustlaya::{Criteria, DEFAULT_REPOSITORY, Laya, Question, QuestionType};
use serde_json::json;

/// 等长用例每个问题指令填充的词数。
const EQUAL_FILLER: usize = 40;
/// 长短混合用例中「短」问题填充的词数。
const SHORT_FILLER: usize = 8;
/// 长短混合用例中「长」问题填充的词数。
const LONG_FILLER: usize = 120;
/// 长短混合用例的短/长问题数量。
const MIXED_HALF: usize = 16;

/// 用固定词填充构造一个长度可控的 two-option `choice` 问题。
fn question(id: usize, filler_words: usize) -> Question {
    let mut instructions = String::from("Is the following item relevant to the keyword? Item:");
    for _ in 0..filler_words {
        instructions.push_str(" lorem");
    }
    Question {
        id: id.to_string(),
        question_type: QuestionType::Choice,
        instructions,
        criteria: Criteria::Choice(vec![
            ("A".to_string(), Some(json!("yes, it is relevant"))),
            ("B".to_string(), Some(json!("no, it is not relevant"))),
        ]),
    }
}

/// `reps` 次采样的中位耗时（毫秒）。
fn median_ms(samples: &[Duration]) -> f64 {
    let mut ordered: Vec<Duration> = samples.to_vec();
    ordered.sort();
    ordered[ordered.len() / 2].as_secs_f64() * 1000.0
}

/// 对一组问题跑 `reps` 次，返回中位耗时（毫秒）。
fn measure(laya: &Laya, state: &serde_json::Value, questions: &[Question], reps: usize) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let _ = laya.predict(state, questions).expect("predict");
        samples.push(start.elapsed());
    }
    median_ms(&samples)
}

fn main() -> anyhow::Result<()> {
    let model = env::var("LAYA_MODEL").unwrap_or_else(|_| DEFAULT_REPOSITORY.to_string());
    let reps = env::var("BENCH_REPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(5);

    let laya = Laya::load(&model)?;
    let state = json!({ "keyword": "benchmark" });
    let sizes = [1usize, 2, 4, 8, 16, 32];

    // 预热：每个 batch size 先跑一次，触发 kernel/缓冲初始化。
    for &size in &sizes {
        let questions: Vec<Question> = (0..size).map(|id| question(id, EQUAL_FILLER)).collect();
        let _ = laya.predict(&state, &questions)?;
    }

    println!("== equal-length questions (filler={EQUAL_FILLER} words, reps={reps}) ==");
    println!(
        "{:>6}  {:>12}  {:>16}  {:>10}",
        "batch", "median_ms", "ms_per_question", "scale"
    );

    let mut single_ms = 0.0;
    for &size in &sizes {
        let questions: Vec<Question> = (0..size).map(|id| question(id, EQUAL_FILLER)).collect();
        let ms = measure(&laya, &state, &questions, reps);
        if size == 1 {
            single_ms = ms;
        }
        let scale = ms / (single_ms * size as f64);
        println!(
            "{size:>6}  {ms:>12.3}  {:>16.3}  {scale:>10.2}",
            ms / size as f64
        );
    }

    println!();
    println!(
        "== mixed-length questions ({} short + {} long) ==",
        MIXED_HALF, MIXED_HALF
    );

    let shorts: Vec<Question> = (0..MIXED_HALF)
        .map(|id| question(id, SHORT_FILLER))
        .collect();
    let longs: Vec<Question> = (MIXED_HALF..MIXED_HALF * 2)
        .map(|id| question(id, LONG_FILLER))
        .collect();
    let mixed: Vec<Question> = shorts.iter().chain(&longs).cloned().collect();

    // 分别预热短批与长批。
    let _ = laya.predict(&state, &shorts)?;
    let _ = laya.predict(&state, &longs)?;

    let short_ms = measure(&laya, &state, &shorts, reps);
    let long_ms = measure(&laya, &state, &longs, reps);
    let mixed_ms = measure(&laya, &state, &mixed, reps);
    let separated_ms = short_ms + long_ms;

    println!(
        "{:>28}  {:>12.3}  {:>16.3}",
        "mixed batch (32)",
        mixed_ms,
        mixed_ms / (MIXED_HALF * 2) as f64
    );
    println!(
        "{:>28}  {:>12.3}  {:>16.3}",
        "separate short batch (16)",
        short_ms,
        short_ms / MIXED_HALF as f64
    );
    println!(
        "{:>28}  {:>12.3}  {:>16.3}",
        "separate long batch (16)",
        long_ms,
        long_ms / MIXED_HALF as f64
    );
    println!(
        "{:>28}  {:>12.3}  {:>16.3}",
        "short + long (sum)",
        separated_ms,
        separated_ms / (MIXED_HALF * 2) as f64
    );
    println!(
        "mixed / (short+long) = {:.2}  (>1 表示混合批被 pad 到最长，存在浪费)",
        mixed_ms / separated_ms
    );

    Ok(())
}
