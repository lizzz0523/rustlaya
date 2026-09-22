//! 用 Laya 对 move/risk/food 三类问题的回答驱动贪吃蛇，并叠加确定性的安全护盾。

use std::time::Instant;

use anyhow::bail;
use indexmap::IndexMap;
use rustlaya::{Answer, Criteria, Laya, Question, QuestionType};
use serde_json::Value;

use crate::game::{DIRECTIONS, MoveInfo, SnakeGame};

/// 提示词风格。
#[derive(Clone, Copy, Default)]
pub enum Prompt {
    #[default]
    Compact,
    Detailed,
}

impl Prompt {
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "compact" => Ok(Prompt::Compact),
            "detailed" => Ok(Prompt::Detailed),
            other => bail!("--prompt must be compact or detailed, got {other}"),
        }
    }
}

/// 单个 tick 的决策结果，供渲染与统计使用。
#[derive(Clone, Debug)]
pub struct Decision {
    pub probabilities: IndexMap<String, f64>,
    pub proposed: Option<String>,
    pub executed: String,
    pub intervened: bool,
    pub dead_end_risk: f64,
    pub food_reachable: f64,
    pub inference_ms: f64,
    pub output_tokens: i32,
}

/// 包裹一次 Laya 推理的策略。
pub struct Policy {
    pub guarded: bool,
    prompt: Prompt,
}

impl Policy {
    pub fn new(guarded: bool, prompt: Prompt) -> Self {
        Self { guarded, prompt }
    }

    pub fn decide(&self, game: &SnakeGame, laya: &Laya) -> anyhow::Result<Decision> {
        let moves = game.moves();
        let safe: Vec<&MoveInfo> = moves.iter().filter(|mov| mov.safe).collect();
        if safe.is_empty() && self.guarded {
            bail!("Cycle safety invariant violated: no safe action");
        }
        let mut planner_best = "NONE".to_string();
        let mut best_advance = -1i64;
        for mov in &safe {
            if mov.advance as i64 > best_advance {
                best_advance = mov.advance as i64;
                planner_best = mov.direction.clone();
            }
        }
        let (reachable, space) = game.food_reachability();
        let compact = matches!(self.prompt, Prompt::Compact);

        let state = if compact {
            format!(
                "Safe route: {}. Food reachable through empty cells: {}.",
                yes_or_no(!safe.is_empty()),
                yes_or_no(reachable)
            )
        } else {
            format!(
                "Snake game. {} safe directions available. Food reachable through empty cells: {}. \
                 Open cells: {}. Snake length: {}. {}",
                safe.len(),
                yes_or_no(reachable),
                space,
                game.length(),
                if safe.is_empty() {
                    "The snake is trapped."
                } else {
                    "There is a safe route forward."
                }
            )
        };

        let criteria = moves
            .iter()
            .map(|mov| {
                let description = if !mov.legal {
                    if compact {
                        "Blocked. Collision.".to_string()
                    } else {
                        format!("Collision: {}. Unsafe.", mov.reason)
                    }
                } else if !mov.safe {
                    if compact {
                        "Unsafe. Traps the snake.".to_string()
                    } else {
                        "Unsafe route. Risk of trapping the snake.".to_string()
                    }
                } else if mov.eats {
                    if compact {
                        "Safe. Eat food now. Best.".to_string()
                    } else {
                        "Safe. Eat the food immediately. Best move.".to_string()
                    }
                } else if mov.direction == planner_best {
                    if compact {
                        "Safe. Best route to food.".to_string()
                    } else {
                        "Safe. Best progress toward food.".to_string()
                    }
                } else if compact {
                    "Safe. Slower route.".to_string()
                } else {
                    "Safe but less progress toward food.".to_string()
                };
                (mov.direction.clone(), Some(Value::String(description)))
            })
            .collect::<Vec<_>>();

        let questions = vec![
            Question {
                id: "move".to_string(),
                question_type: QuestionType::Choice,
                instructions: if compact {
                    "Choose the best safe move toward food.".to_string()
                } else {
                    "Select the safest move with best progress toward food. Avoid collisions."
                        .to_string()
                },
                criteria: Criteria::Choice(criteria),
            },
            Question {
                id: "risk".to_string(),
                question_type: QuestionType::Noul,
                instructions: if compact {
                    "Is a safe route available?".to_string()
                } else {
                    "Is there a safe route forward for the snake?".to_string()
                },
                criteria: Criteria::Noul {
                    false_criterion: None,
                    true_criterion: None,
                },
            },
            Question {
                id: "food".to_string(),
                question_type: QuestionType::Noul,
                instructions: if compact {
                    "Is food reachable through empty cells?".to_string()
                } else {
                    "Is food reachable through the currently empty cells?".to_string()
                },
                criteria: Criteria::Noul {
                    false_criterion: None,
                    true_criterion: None,
                },
            },
        ];

        let state_value = Value::String(state);
        let inference_start = Instant::now();
        let output = laya.predict(&state_value, &questions)?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        let answers = &output.answers;
        let probabilities = match answers.get("move") {
            Some(Answer::Choice { probabilities, .. }) => probabilities.clone(),
            _ => bail!("model did not return a choice answer for 'move'"),
        };
        let risk = match answers.get("risk") {
            Some(Answer::Noul { noul, .. }) => *noul,
            _ => bail!("model did not return a noul answer for 'risk'"),
        };
        let food = match answers.get("food") {
            Some(Answer::Noul { noul, .. }) => *noul,
            _ => bail!("model did not return a noul answer for 'food'"),
        };

        for score in probabilities.values().chain([&risk, &food]) {
            if !score.is_finite() || !(0.0..=1.0).contains(score) {
                bail!("Model returned an invalid probability; no move executed");
            }
        }

        let proposed = argmax_direction(&probabilities);
        let allowed: Vec<String> = safe.iter().map(|mov| mov.direction.clone()).collect();
        let executed = if self.guarded && !allowed.contains(&proposed) {
            argmax_direction(
                &allowed
                    .iter()
                    .map(|direction| (direction.clone(), probabilities[direction]))
                    .collect(),
            )
        } else {
            proposed.clone()
        };

        Ok(Decision {
            probabilities,
            proposed: Some(proposed.clone()),
            executed: executed.clone(),
            intervened: proposed != executed,
            dead_end_risk: 1.0 - risk,
            food_reachable: food,
            inference_ms,
            output_tokens: output.usage.output_tokens,
        })
    }
}

fn yes_or_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn argmax_direction(probabilities: &IndexMap<String, f64>) -> String {
    let mut best = DIRECTIONS[0].to_string();
    let mut best_value = f64::NEG_INFINITY;
    for direction in DIRECTIONS {
        let value = probabilities
            .get(direction)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        if value > best_value {
            best_value = value;
            best = direction.to_string();
        }
    }
    best
}
