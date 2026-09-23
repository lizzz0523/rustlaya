use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
use rustlaya::{Answer, Criteria, Laya, Question, QuestionType};
use serde_json::json;

use crate::hn::Story;

/// 每次前向批量打分的 story 数。批内会 pad 到最长序列，实测该 workload 下
/// 小批量（约 4~8）总耗时更优，故取 8。
const BATCH_SIZE: usize = 8;
/// 粗筛后最多送给 Laya 的候选数。
const TOP_K_CANDIDATES: usize = 100;
/// story 文本送入问题头前的字符上限（head_max_len=192 token，约等于此量级）。
const STORY_TEXT_MAX_CHARS: usize = 512;
/// BM25 TF 饱和系数。
const BM25_K1: f64 = 1.2;
/// BM25 长度归一系数。
const BM25_B: f64 = 0.75;

/// 全局只做一次 Metal kernel 预热。
static WARMED: OnceLock<()> = OnceLock::new();

/// 基于 rustlaya 对 stories 做关键字相关性检索。
///
/// 先用 BM25 词法打分粗筛出 `TOP_K_CANDIDATES` 个候选，再把每篇候选作为
/// 一个中立键 `choice` 问题（`A` = 相关）批量送 Laya 打分。
/// 返回 `(story 下标, 相关性概率 0-1)`，仅保留概率 >= `min_score` 的项，按概率降序。
pub fn search(
    laya: &Laya,
    stories: &[Story],
    keyword: &str,
    min_score: f64,
) -> Result<Vec<(usize, f64)>> {
    let state = json!({ "keyword": keyword });

    let candidates = candidates(stories, keyword, TOP_K_CANDIDATES);

    // 每个候选只构造一次问题，并按问题指令长度排序后再切块：使每个 batch 的
    // 序列长度尽量同质，减少 `Batch::collate` 按批内最长序列 padding 造成的
    // 计算浪费。最终命中按概率排序输出，因此这里重排候选顺序不影响结果。
    let mut ranked: Vec<(usize, Question)> = candidates
        .into_iter()
        .map(|index| (index, relevant_question(index, &stories[index], keyword)))
        .collect();
    ranked.sort_by_key(|(_, question)| question.instructions.len());

    // 拆成一一对应的「story 下标」与「问题」，问题作为连续切片直接用于 predict。
    let (indices, questions): (Vec<usize>, Vec<Question>) = ranked.into_iter().unzip();

    if let Some(first) = questions.first() {
        WARMED.get_or_init(|| {
            if let Err(error) = laya.predict(&state, std::slice::from_ref(first)) {
                eprintln!("warmup predict failed: {error:#}");
            }
        });
    }

    let mut hits = Vec::new();

    for (chunk_index, (question_chunk, index_chunk)) in questions
        .chunks(BATCH_SIZE)
        .zip(indices.chunks(BATCH_SIZE))
        .enumerate()
    {
        let started = Instant::now();
        let response = laya.predict(&state, question_chunk).with_context(|| {
            format!(
                "predict batch {chunk_index} ({} candidates)",
                question_chunk.len()
            )
        })?;
        eprintln!(
            "predict batch {chunk_index} ({} candidates): {:.1} ms",
            question_chunk.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );

        for &index in index_chunk {
            if let Some(Answer::Choice { probabilities, .. }) =
                response.answers.get(&index.to_string())
                && let Some(&score) = probabilities.get("A")
                && score >= min_score
            {
                hits.push((index, score));
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
            let frequency = *term_frequency.get(term.as_str()).unwrap_or(&0) as f64;
            if frequency == 0.0 {
                continue;
            }
            let frequency_in_documents =
                *document_frequency.get(term.as_str()).unwrap_or(&0) as f64;
            let inverse_document_frequency = (1.0
                + (total_count as f64 - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5))
                .ln();
            let denominator =
                frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length);
            score += inverse_document_frequency * frequency * (BM25_K1 + 1.0) / denominator;
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

/// 把一篇 story 包成中立键的 two-option `choice` 问题，`A` 表示相关。
fn relevant_question(index: usize, story: &Story, keyword: &str) -> Question {
    let mut instructions =
        format!("Is the following Hacker News story relevant to the keyword \"{keyword}\"? Story:");
    if let Some(title) = &story.title {
        instructions.push_str(&format!(" title: {title}"));
    }
    if let Some(text) = &story.text {
        instructions.push_str(" text: ");
        instructions.extend(text.chars().take(STORY_TEXT_MAX_CHARS));
    }
    if let Some(url) = &story.url {
        instructions.push_str(&format!(" url: {url}"));
    }

    Question {
        id: index.to_string(),
        question_type: QuestionType::Choice,
        instructions,
        criteria: Criteria::Choice(vec![
            ("A".to_string(), Some(json!("yes, the story is relevant"))),
            (
                "B".to_string(),
                Some(json!("no, the story is not relevant")),
            ),
        ]),
    }
}
