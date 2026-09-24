//! Laya 驱动的 composition —— 本示例里的 “Jev 决策模型 + json-render composer”。
//!
//! 关键点：Laya 不生成自由文本，只在候选中做离散决策。composer 把这些决策
//! 翻译成一棵 flat [`Spec`]：
//!
//! 1. **选择（step 1，一次前向传播）**：`choice` 选根布局，`noul` 逐个判断候选
//!    元素是否纳入（`noul >= INCLUDE_THRESHOLD`）。
//! 2. **布局（step 2）**：`choice` 决定命名槽位，`choice` 分段决定顺序。
//! 3. **编辑**：先 `choice` 选操作（add/remove/move），再 `choice` 选目标并应用。
//!
//! 这正是 `experimental_composeSpec` 的分工：模型选题、平台定能力、composer 组装。

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Result, bail};
use rustlaya::{Answer, Criteria, Laya, Question, QuestionType, Response};
use serde_json::{Value, json};

use crate::catalog::{CATALOG, Candidate, Preset};
use crate::spec::Spec;

/// 一条决策记录，供 TUI trace 面板展示。
pub struct Step {
    pub id: String,
    pub choice: String,
    pub detail: Vec<(String, f64)>,
    pub confidence: f64,
    pub ms: f64,
}

/// 一次 composition 的结果。
pub struct Composition {
    pub spec: Spec,
    pub trace: Vec<Step>,
    pub elapsed_ms: f64,
}

/// 纳入候选的 `noul` 阈值。Laya 对“是否包含”类问题整体偏高，取 0.75 才有区分度。
const INCLUDE_THRESHOLD: f64 = 0.75;

/// 排序使用的语义分段：模型把每个元素归到一个区间，再按区间 + catalog 顺序排列。
const BANDS: [(&str, &str); 4] = [
    ("top", "The very first section near the title"),
    ("upper", "An early section"),
    ("lower", "A later section"),
    ("bottom", "The final section at the bottom"),
];

/// 从零组装一棵新树。
pub fn compose(laya: &Laya, preset: &Preset, prompt: &str) -> Result<Composition> {
    let started = Instant::now();
    let mut trace = Vec::new();
    let state = state_for(preset, prompt, None);

    let roots: Vec<(usize, &Candidate)> = preset
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.root)
        .collect();
    let members: Vec<(usize, &Candidate)> = preset
        .candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !candidate.root)
        .collect();
    if roots.is_empty() {
        bail!("preset {} has no root candidate", preset.name);
    }

    // ---- step 1: root + membership --------------------------------------
    let mut select_questions = vec![Question {
        id: "root".to_string(),
        question_type: QuestionType::Choice,
        instructions: format!(
            "Choose the page layout that best fits this request: \"{prompt}\". \
             Prefer the layout that presents the content most clearly."
        ),
        criteria: Criteria::Choice(
            roots
                .iter()
                .map(|(_, candidate)| {
                    (candidate.id.to_string(), Some(json!(candidate.description)))
                })
                .collect(),
        ),
    }];
    for (_, candidate) in &members {
        select_questions.push(Question {
            id: format!("include_{}", candidate.id),
            question_type: QuestionType::Noul,
            instructions: format!(
                "Should this page include the following element? {}",
                candidate.description
            ),
            criteria: Criteria::Noul {
                false_criterion: Some(json!("omit this element")),
                true_criterion: Some(json!("include this element")),
            },
        });
    }
    predict_and_trace(laya, &state, &select_questions, &mut trace)?;

    let root_id = choice_answer(&trace, "root").unwrap_or_else(|| roots[0].1.id.to_string());
    let root_candidate = roots
        .iter()
        .find(|(_, candidate)| candidate.id == root_id)
        .map(|(_, candidate)| *candidate)
        .unwrap_or(roots[0].1);

    let included: Vec<(usize, &Candidate)> = members
        .iter()
        .filter(|(_, candidate)| {
            noul_answer(&trace, &format!("include_{}", candidate.id)).unwrap_or(0.0)
                >= INCLUDE_THRESHOLD
        })
        .copied()
        .collect();

    // ---- step 2: slots + order ------------------------------------------
    let root_slots: Vec<String> = root_candidate.element.slots.keys().cloned().collect();
    let mut layout_questions = Vec::new();
    for (_, candidate) in &included {
        if root_slots.len() > 1 {
            layout_questions.push(Question {
                id: format!("slot_{}", candidate.id),
                question_type: QuestionType::Choice,
                instructions: format!(
                    "Which column should hold this element? {}",
                    candidate.description
                ),
                criteria: Criteria::Choice(
                    root_slots
                        .iter()
                        .map(|slot| (slot.clone(), Some(json!(slot_describe(slot)))))
                        .collect(),
                ),
            });
        }
        layout_questions.push(Question {
            id: format!("order_{}", candidate.id),
            question_type: QuestionType::Choice,
            instructions: format!(
                "Where in the page order should this section appear? {}",
                candidate.description
            ),
            criteria: Criteria::Choice(
                BANDS
                    .iter()
                    .map(|(label, description)| (label.to_string(), Some(json!(description))))
                    .collect(),
            ),
        });
    }
    if !layout_questions.is_empty() {
        predict_and_trace(laya, &state, &layout_questions, &mut trace)?;
    }

    let mut order_for: HashMap<&str, usize> = HashMap::new();
    let mut slot_for: HashMap<&str, String> = HashMap::new();
    for (_, candidate) in &included {
        let band = choice_answer(&trace, &format!("order_{}", candidate.id)).unwrap_or_default();
        order_for.insert(candidate.id, band_rank(&band));
        if root_slots.len() > 1 {
            slot_for.insert(
                candidate.id,
                choice_answer(&trace, &format!("slot_{}", candidate.id))
                    .filter(|slot| root_slots.contains(slot))
                    .unwrap_or_else(|| root_slots[0].clone()),
            );
        }
    }

    // ---- assemble a flat Spec -------------------------------------------
    let root_key = "root";
    let mut spec = Spec::new();
    spec.insert(root_key, root_candidate.element.as_container());
    spec.root = Some(root_key.to_string());

    let mut ordered = included.clone();
    ordered.sort_by(|a, b| {
        let left = order_for.get(a.1.id).copied().unwrap_or(1);
        let right = order_for.get(b.1.id).copied().unwrap_or(1);
        left.cmp(&right).then(a.0.cmp(&b.0))
    });

    for (_, candidate) in ordered {
        spec.insert(candidate.id.to_string(), candidate.element.as_leaf());
        if root_slots.is_empty() {
            spec.attach(candidate.id, root_key, None);
        } else {
            let slot = slot_for
                .get(candidate.id)
                .cloned()
                .unwrap_or_else(|| root_slots[0].clone());
            spec.attach(candidate.id, root_key, Some(&slot));
        }
    }

    Ok(Composition {
        spec,
        trace,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// 在已有页面上应用一步编辑：step 1 用 `choice` 选操作（add/remove/move），
/// step 2 用 `choice` 在可行目标里选元素，最后应用到 `current` 的克隆上。
pub fn edit(laya: &Laya, preset: &Preset, prompt: &str, current: &Spec) -> Result<Composition> {
    let started = Instant::now();
    let mut trace = Vec::new();
    let state = state_for(preset, prompt, Some(current));

    let present: Vec<String> = current
        .elements
        .keys()
        .filter(|id| current.root.as_deref() != Some(id.as_str()))
        .cloned()
        .collect();

    let addable: Vec<&Candidate> = preset
        .candidates
        .iter()
        .filter(|candidate| !candidate.root && !present.contains(&candidate.id.to_string()))
        .collect();

    // ---- step 1: 选择操作类型（只在可行的操作里选）-----------------------
    let mut action_criteria: Vec<(String, Option<Value>)> = Vec::new();
    if !addable.is_empty() {
        action_criteria.push((
            "add".to_string(),
            Some(json!("Add a new element that is not currently on the page")),
        ));
    }
    if !present.is_empty() {
        action_criteria.push((
            "remove".to_string(),
            Some(json!("Remove an existing element from the page")),
        ));
        action_criteria.push((
            "move".to_string(),
            Some(json!("Move an existing element to the top of the page")),
        ));
    }
    if action_criteria.is_empty() {
        bail!("this page has nothing to edit");
    }

    let first_action = action_criteria[0].0.clone();
    let action_question = Question {
        id: "action".to_string(),
        question_type: QuestionType::Choice,
        instructions: format!(
            "Which single operation should be applied to the page in response to: \"{prompt}\"?"
        ),
        criteria: Criteria::Choice(action_criteria),
    };
    let action_questions = vec![action_question];
    predict_and_trace(laya, &state, &action_questions, &mut trace)?;
    let action = choice_answer(&trace, "action").unwrap_or(first_action);

    // ---- step 2: 选择目标元素 -------------------------------------------
    // 与 Jev 的 “先选操作、再选目标” 一致：step 1 已确定动词，这里只在可行目标里选。
    let (question_text, targets): (&str, Vec<String>) = match action.as_str() {
        "remove" if !present.is_empty() => ("Which element should be removed?", present.clone()),
        "move" if !present.is_empty() => {
            ("Which element should be moved to the top?", present.clone())
        }
        "add" if !addable.is_empty() => (
            "Which element should be added?",
            addable
                .iter()
                .map(|candidate| candidate.id.to_string())
                .collect(),
        ),
        _ => bail!("no applicable edit for: {prompt}"),
    };

    let first_target = targets.first().cloned().unwrap_or_default();
    let target_question = Question {
        id: "target".to_string(),
        question_type: QuestionType::Choice,
        instructions: format!("{question_text} The instruction was: \"{prompt}\"."),
        criteria: Criteria::Choice(
            targets
                .iter()
                .map(|id| {
                    (
                        id.clone(),
                        Some(json!(format!("{} — {}", id, describe(preset, current, id)))),
                    )
                })
                .collect(),
        ),
    };
    let target_questions = vec![target_question];
    predict_and_trace(laya, &state, &target_questions, &mut trace)?;
    let target = choice_answer(&trace, "target").unwrap_or(first_target);

    let mut spec = current.clone();
    match action.as_str() {
        "remove" => spec.detach(&target),
        "move" => spec.move_to_front(&target),
        _ => {
            // add：处理互斥变体后挂入。
            if let Some(candidate) = addable.iter().find(|candidate| candidate.id == target)
                && let Some(resource) = candidate.resource
            {
                let conflicts: Vec<String> = present
                    .iter()
                    .filter(|id| {
                        preset.candidates.iter().any(|other| {
                            other.id == id.as_str() && other.resource == Some(resource)
                        })
                    })
                    .cloned()
                    .collect();
                for conflict in conflicts {
                    spec.detach(&conflict);
                }
            }
            add_candidate(laya, preset, &mut spec, &target, &state, &mut trace)?;
        }
    }

    Ok(Composition {
        spec,
        trace,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// 把某个候选元素挂进现有树（必要时先问一次槽位）。
fn add_candidate(
    laya: &Laya,
    preset: &Preset,
    spec: &mut Spec,
    id: &str,
    state: &Value,
    trace: &mut Vec<Step>,
) -> Result<()> {
    let Some(candidate) = preset
        .candidates
        .iter()
        .find(|candidate| candidate.id == id)
    else {
        return Ok(());
    };
    let root_key = spec.root.clone().unwrap_or_default();
    let root_slots: Vec<String> = spec
        .get(&root_key)
        .map(|element| element.slots.keys().cloned().collect())
        .unwrap_or_default();

    let slot = if root_slots.len() > 1 {
        let question = Question {
            id: format!("slot_{id}"),
            question_type: QuestionType::Choice,
            instructions: format!(
                "Which column should hold this element? {}",
                candidate.description
            ),
            criteria: Criteria::Choice(
                root_slots
                    .iter()
                    .map(|slot| (slot.clone(), Some(json!(slot_describe(slot)))))
                    .collect(),
            ),
        };
        let questions = vec![question];
        predict_and_trace(laya, state, &questions, trace)?;
        choice_answer(trace, &format!("slot_{id}"))
            .filter(|slot| root_slots.contains(slot))
            .unwrap_or_else(|| root_slots[0].clone())
    } else {
        root_slots.first().cloned().unwrap_or_default()
    };

    spec.insert(candidate.id.to_string(), candidate.element.as_leaf());
    if root_slots.is_empty() {
        spec.attach(candidate.id, &root_key, None);
    } else {
        spec.attach(candidate.id, &root_key, Some(&slot));
    }
    Ok(())
}

/// 构建发送给 Laya 的 state。`initial` 非空时表示这是一次编辑，附带当前页面摘要。
fn state_for(preset: &Preset, request: &str, initial: Option<&Spec>) -> Value {
    let components: Vec<Value> = CATALOG
        .iter()
        .map(|definition| json!({"name": definition.name, "description": definition.description}))
        .collect();
    let mut state = json!({
        "request": request,
        "preset": preset.name,
        "available_components": components,
    });
    if let Some(spec) = initial {
        state["current_page"] = json!(spec.summary());
    }
    state
}

/// 跑一次前向传播，并把该批每个问题的答案记入 trace。
fn predict_and_trace(
    laya: &Laya,
    state: &Value,
    questions: &[Question],
    trace: &mut Vec<Step>,
) -> Result<()> {
    let started = Instant::now();
    let response = laya.predict(state, questions)?;
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    for question in questions {
        trace.push(step_from(&response, question, ms));
    }
    Ok(())
}

/// 把某个问题的答案整理成一条 trace。
fn step_from(response: &Response, question: &Question, ms: f64) -> Step {
    let base = |choice: String, detail: Vec<(String, f64)>, confidence: f64| Step {
        id: question.id.clone(),
        choice,
        detail,
        confidence,
        ms,
    };
    match response.answers.get(&question.id) {
        Some(Answer::Choice {
            choice,
            probabilities,
            confidence,
            ..
        }) => base(
            choice.clone(),
            probabilities
                .iter()
                .map(|(label, value)| (label.clone(), *value))
                .collect(),
            *confidence,
        ),
        Some(Answer::Noul {
            noul, confidence, ..
        }) => base(
            format!("{} ({:.2})", yes_or_no(*noul >= INCLUDE_THRESHOLD), noul),
            vec![("yes".to_string(), *noul), ("no".to_string(), 1.0 - *noul)],
            *confidence,
        ),
        _ => base("—".to_string(), Vec::new(), 0.0),
    }
}

fn yes_or_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn choice_answer(source: &[Step], id: &str) -> Option<String> {
    source.iter().find(|step| step.id == id).and_then(|step| {
        let choice = step.choice.as_str();
        if step.detail.is_empty() || step.detail.iter().any(|(label, _)| label == choice) {
            Some(choice.to_string())
        } else {
            step.detail
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(label, _)| label.clone())
        }
    })
}

fn noul_answer(source: &[Step], id: &str) -> Option<f64> {
    let step = source.iter().rev().find(|step| step.id == id)?;
    step.detail.first().map(|(_, value)| *value)
}

/// 把模型选中的分段名映射为排序权重（越小越靠前）。
fn band_rank(label: &str) -> usize {
    BANDS
        .iter()
        .position(|(name, _)| *name == label)
        .unwrap_or(1)
}

fn describe(preset: &Preset, spec: &Spec, id: &str) -> String {
    if let Some(candidate) = preset
        .candidates
        .iter()
        .find(|candidate| candidate.id == id)
    {
        return candidate.description.to_string();
    }
    spec.get(id)
        .map(|element| element.kind.clone())
        .unwrap_or_else(|| id.to_string())
}

fn slot_describe(slot: &str) -> &'static str {
    match slot {
        "main" | "default" => "Primary column for the main content",
        "aside" => "Secondary side column for supporting content",
        _ => "Content column",
    }
}
