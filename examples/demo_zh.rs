//! 中文 CLI 示例：使用多语言检查点对一段中文状态运行一组类型化问题。

use std::env;
use std::time;

use anyhow::Context;
use rustlaya::{Criteria, Laya, Question, QuestionType};
use serde_json::json;

/// 官方多语言检查点（`multilingual/` 子目录的独立镜像仓库）。
const MULTILINGUAL_REPOSITORY: &str = "convaiinnovations/laya-multilingual";

fn main() -> anyhow::Result<()> {
    // 本地目录或 Hugging Face 仓库 id；由 `LAYA_MODEL` 指定，缺省用多语言仓库。
    let model = env::var("LAYA_MODEL").unwrap_or_else(|_| MULTILINGUAL_REPOSITORY.to_string());

    let laya = Laya::load(&model)?;

    let state = json!("我的订单被重复扣款了两次，请今天就帮我退款。");
    // 示例输入：针对同一个状态的 choice、score、noul 三个中文问题。
    let questions = {
        let choice = Question {
            id: "department".to_string(),
            question_type: QuestionType::Choice,
            instructions: "这个请求应该由哪个部门处理？".to_string(),
            criteria: Criteria::Choice(vec![
                ("billing".to_string(), Some(json!("账单、支付与退款"))),
                ("technical".to_string(), Some(json!("故障与线上问题"))),
                ("sales".to_string(), Some(json!("购买与咨询"))),
            ]),
        };
        let score = Question {
            id: "urgency".to_string(),
            question_type: QuestionType::Score,
            instructions: "这个请求有多紧急？".to_string(),
            criteria: Criteria::Score(vec![json!("不急"), json!("尽快"), json!("非常紧急")]),
        };
        let noul = Question {
            id: "refund".to_string(),
            question_type: QuestionType::Noul,
            instructions: "客户是否要求退款？".to_string(),
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
