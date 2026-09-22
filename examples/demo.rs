//! 极简 CLI 示例：对同一个状态运行一组固定的类型化问题，并打印结果文档。

use std::env;
use std::time;

use anyhow::Context;
use serde_json::json;
use rustlaya::{Criteria, DEFAULT_REPOSITORY, Laya, Question, QuestionType};

fn main() -> anyhow::Result<()> {
    // 本地目录或 Hugging Face 仓库 id；由 `LAYA_MODEL` 指定，缺省用官方仓库。
    let model = env::var("LAYA_MODEL").unwrap_or_else(|_| DEFAULT_REPOSITORY.to_string());

    let laya = Laya::load(&model)?;

    let state = json!("I was billed twice. Please refund the duplicate today.");
    // 示例输入：针对同一个状态的 choice、score、noul 三个问题。
    let questions = {
        let choice = Question {
            id: "department".to_string(),
            question_type: QuestionType::Choice,
            instructions: "Which department should handle this request?".to_string(),
            criteria: Criteria::Choice(vec![
                (
                    "billing".to_string(),
                    Some(json!("invoices, payments, refunds")),
                ),
                ("technical".to_string(), Some(json!("bugs and outages"))),
                ("sales".to_string(), Some(json!("new purchases"))),
            ]),
        };
        let score = Question {
            id: "urgency".to_string(),
            question_type: QuestionType::Score,
            instructions: "How urgent is this request?".to_string(),
            criteria: Criteria::Score(vec![json!("not urgent"), json!("soon"), json!("critical")]),
        };
        let noul = Question {
            id: "refund".to_string(),
            question_type: QuestionType::Noul,
            instructions: "Does the customer ask for money back?".to_string(),
            criteria: Criteria::Noul {
                false_criterion: None,
                true_criterion: None,
            },
        };

        vec![choice, score, noul]
    };

    let start = time::Instant::now();
    let result = laya.predict(&state, &questions)?;
    let elapsed = start.elapsed();
    eprintln!(
        "predict: {:.3} ms ({} question(s))",
        elapsed.as_secs_f64() * 1000.0,
        questions.len()
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&result).context("serialising result")?
    );
    Ok(())
}
