use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
use rustlaya::{Answer, Criteria, Laya, Question, QuestionType};
use serde_json::json;

use crate::hn::Story;

/// 每个请求 state.stories 里放入的候选数。本地检查点 `max_len=512`，问题头约
/// 占 100 token，整份 state（含本批候选）只能留约 400 token，故批必须很小。
const BATCH_SIZE: usize = 2;
/// 粗筛后最多送给 Laya 的候选数。
const TOP_K_CANDIDATES: usize = 100;
/// 每篇 story 文本写入 state.stories 前的字符上限（约 100 token，保证本批能放进窗口）。
const STORY_TEXT_MAX_CHARS: usize = 400;
/// BM25 TF 饱和系数。
const BM25_K1: f64 = 1.2;
/// BM25 长度归一系数。
const BM25_B: f64 = 0.75;

/// 全局只做一次 Metal kernel 预热。
static WARMED: OnceLock<()> = OnceLock::new();

/// 基于 rustlaya 对 stories 做关键字相关性检索。
///
/// 先用 BM25 词法打分粗筛出 `TOP_K_CANDIDATES` 个候选，再按 `BATCH_SIZE` 分批：
/// 每批候选写入 state.stories，并把每个候选包成一个 `score` 问题（`Evaluate ONLY
/// stories[i] ...`，0-3 四级）送 Laya 打分。
/// 返回 `(story 下标, 相关性分数 0-3)`，仅保留分数 >= `min_score` 的项，按分数降序。
pub fn search(
    laya: &Laya,
    stories: &[Story],
    keyword: &str,
    min_score: f64,
) -> Result<Vec<(usize, f64)>> {
    let candidates = candidates(stories, keyword, TOP_K_CANDIDATES);

    let mut hits = Vec::new();

    // 每个请求的 state 只装本批候选，问题按 `stories[position]` 引用；这与 JeV
    // `rankingPayload` 的形态一致（候选进 state、question 按下标引用）。
    for (batch_index, batch) in candidates.chunks(BATCH_SIZE).enumerate() {
        let state = ranking_state(keyword, batch, stories);
        let questions: Vec<Question> = batch
            .iter()
            .enumerate()
            .map(|(position, &index)| ranking_question(position, index))
            .collect();

        WARMED.get_or_init(|| {
            if let Err(error) = laya.predict(&state, std::slice::from_ref(&questions[0])) {
                eprintln!("warmup predict failed: {error:#}");
            }
        });

        let started = Instant::now();
        let response = laya.predict(&state, &questions).with_context(|| {
            format!(
                "predict batch {batch_index} ({} candidates)",
                questions.len()
            )
        })?;
        eprintln!(
            "predict batch {batch_index} ({} candidates): {:.1} ms",
            questions.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );

        for &index in batch {
            if let Some(Answer::Score { score, .. }) = response.answers.get(&index.to_string())
                && *score >= min_score
            {
                hits.push((index, *score));
            }
        }
    }

    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    Ok(hits)
}

/// BM25 粗筛：对每篇 story 按关键字词打 Okapi BM25 分，降序取前 `top_k`；
/// 全不命中时退回前 `top_k` 篇。
fn candidates(stories: &[Story], keyword: &str, top_k: usize) -> Vec<usize> {
    let mut query = tokens_of(keyword);
    query.sort();
    query.dedup();

    if query.is_empty() || stories.is_empty() {
        return fallback_candidates(stories, top_k);
    }

    let documents: Vec<Vec<String>> = stories
        .iter()
        .map(|story| tokens_of(&story_haystack(story)))
        .collect();
    let total_count = documents.len();
    let average_length = documents.iter().map(Vec::len).sum::<usize>() as f64 / total_count as f64;

    let query_terms: HashSet<&str> = query.iter().map(String::as_str).collect();

    // 每个查询词的文档频率 df。
    let mut document_frequency: HashMap<&str, usize> = HashMap::new();
    for document in &documents {
        let present: HashSet<&str> = document
            .iter()
            .map(String::as_str)
            .filter(|term| query_terms.contains(term))
            .collect();
        for term in present {
            *document_frequency.entry(term).or_insert(0) += 1;
        }
    }

    let mut scored: Vec<(usize, f64)> = Vec::new();
    for (index, document) in documents.iter().enumerate() {
        let length = document.len() as f64;

        let mut term_frequency: HashMap<&str, usize> = HashMap::new();
        for token in document {
            let token = token.as_str();
            if query_terms.contains(token) {
                *term_frequency.entry(token).or_insert(0) += 1;
            }
        }
        if term_frequency.is_empty() {
            continue;
        }

        let mut score = 0.0;
        for term in &query {
            let tf = *term_frequency.get(term.as_str()).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }
            let df = *document_frequency.get(term.as_str()).unwrap_or(&0) as f64;
            let idf = (1.0 + (total_count as f64 - df + 0.5) / (df + 0.5)).ln();
            let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length);
            score += idf * tf * (BM25_K1 + 1.0) / denominator;
        }

        if score > 0.0 {
            scored.push((index, score));
        }
    }

    if scored.is_empty() {
        return fallback_candidates(stories, top_k);
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect()
}

/// BM25 无有效信号时退回前 `top_k` 篇的下标。
fn fallback_candidates(stories: &[Story], top_k: usize) -> Vec<usize> {
    (0..stories.len().min(top_k)).collect()
}

/// story 的小写文本，用于粗筛。
fn story_haystack(story: &Story) -> String {
    let mut haystack = String::new();
    if let Some(title) = &story.title {
        haystack.push_str(title);
        haystack.push(' ');
    }
    if let Some(text) = &story.text {
        haystack.push_str(text);
        haystack.push(' ');
    }
    if let Some(url) = &story.url {
        haystack.push_str(url);
        haystack.push(' ');
    }
    haystack.to_lowercase()
}

/// 按非字母数字切词、去空、小写。
///
/// 注意：`is_alphanumeric` 会把连续中文视作单个 token，因此中文关键字无法按词
/// 匹配；示例默认关键字为英文。
fn tokens_of(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

/// 把一批 story 组装成请求 state：`request` 为关键字，`stories` 为本批候选。
///
/// 形态对齐 JeV `rankingPayload` 的 state（候选放进 state，问题按下标引用）。
fn ranking_state(keyword: &str, batch: &[usize], stories: &[Story]) -> serde_json::Value {
    let candidates: Vec<serde_json::Value> = batch
        .iter()
        .map(|&index| {
            let story = &stories[index];
            json!({
                "title": story.title,
                "text": story.text.as_deref().map(|text| {
                    text.chars().take(STORY_TEXT_MAX_CHARS).collect::<String>()
                }),
                "url": story.url,
            })
        })
        .collect();

    json!({ "request": keyword, "stories": candidates })
}

/// 把本批中处于 `position` 的候选包成一个 0-3 四级 `score` 问题。
///
/// 只引用 `stories[position]`，候选正文留在 state 中，与 JeV 的 ranking 问题一致。
fn ranking_question(position: usize, index: usize) -> Question {
    Question {
        id: index.to_string(),
        question_type: QuestionType::Score,
        instructions: format!(
            "Evaluate ONLY stories[{position}] against request. \
             Use the supplied description as evidence; do not invent plot details. \
             How well does this story fit the requested keyword?"
        ),
        criteria: Criteria::Score(vec![
            json!("Contradicts the request OR insufficient evidence of any meaningful match."),
            json!("Only broadly related; most specific requested qualities are unsupported."),
            json!("Good match to the main preference; some details are unverified."),
            json!("Strong evidence for the main requested qualities without a known conflict."),
        ]),
    }
}
