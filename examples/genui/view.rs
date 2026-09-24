//! 终端 TUI 框架：左栏渲染 Spec，右栏展示决策 trace / 原始 JSON，底部是输入行。
//!
//! 这一层对应 json-render 的 `@json-render/ink` 之类的宿主界面：它把渲染器输出
//! 和“模型做了什么决策”并排展示，让 Jev 的离散选择过程可见。

use std::cmp::Ordering;

use crate::composer::Composition;
use crate::render::{self, Canvas};
use crate::spec::Spec;
use crate::text_layout::truncate;

/// 右侧 trace/JSON 面板宽度，以及显示该面板所需的最小列数。
const TRACE_WIDTH: usize = 46;
const TRACE_MIN_COLUMNS: usize = 112;
/// 顶部（空行、标题、分隔线、空行）与底部（分隔线、输入行、状态行、空行）占用的行数。
const HEADER_ROWS: usize = 4;
const FOOTER_ROWS: usize = 4;

/// 右侧面板一页可滚动的行数（与面板内可见行数一致）。
pub fn page_rows(rows: usize) -> usize {
    rows.saturating_sub(HEADER_ROWS + FOOTER_ROWS + 4).max(1)
}

/// 底部输入行当前用于哪个动作。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    New,
    Edit,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Input(Purpose),
    Browse,
}

pub struct UiState {
    pub preset: &'static str,
    pub title: &'static str,
    pub input: String,
    pub mode: Mode,
    pub show_json: bool,
    pub status: String,
    pub composition: Option<Composition>,
    /// 右侧面板顶部显示的起始行号（0 表示顶部）。
    pub scroll: usize,
}

impl UiState {
    pub fn new(preset: &'static str, title: &'static str) -> Self {
        Self {
            preset,
            title,
            input: String::new(),
            mode: Mode::Input(Purpose::New),
            show_json: false,
            status: "type a request and press Enter".to_string(),
            composition: None,
            scroll: 0,
        }
    }

    /// 切换到新内容时回到顶部。
    pub fn reset_scroll(&mut self) {
        self.scroll = 0;
    }

    /// 按 `delta` 行滚动右侧面板，并把 `scroll` 夹在有效范围内。
    ///
    /// `columns` 决定是否显示右侧面板；`rows` 与 `compose` 一样用于推算可见行数。
    pub fn scroll_by(&mut self, delta: i32, columns: usize, rows: usize) {
        let Some((content, visible)) = self.panel_metrics(columns, rows) else {
            return;
        };
        let max = content.len().saturating_sub(visible);
        let next = if delta.is_negative() {
            self.scroll.saturating_sub(delta.unsigned_abs() as usize)
        } else {
            self.scroll.saturating_add(delta as usize)
        };
        self.scroll = next.min(max);
    }

    /// 构建完整的一帧。
    pub fn compose(&self, columns: usize, rows: usize) -> Canvas {
        let mut canvas = Canvas::new(columns, rows);

        canvas.put(1, 2, "LAYA  /  GENERATIVE UI", render::FG);
        canvas.put(1, 26, self.title, render::ACCENT);
        canvas.put(
            1,
            26 + self.title.chars().count() as i32 + 2,
            "(json-render + jev, terminal edition)",
            render::MUTED,
        );

        let right = format!(
            "PRESET {}   │   {}   │   {}",
            self.preset.to_uppercase(),
            match self.mode {
                Mode::Input(Purpose::New) => "NEW",
                Mode::Input(Purpose::Edit) => "EDIT",
                Mode::Browse => "BROWSE",
            },
            if self.show_json { "JSON" } else { "TRACE" },
        );
        canvas.put(
            1,
            columns as i32 - right.chars().count() as i32 - 2,
            &right,
            render::MUTED,
        );

        canvas.put(2, 2, &"─".repeat(columns.saturating_sub(4)), render::DIM);

        let body_top = HEADER_ROWS;
        let body_height = rows.saturating_sub(HEADER_ROWS + FOOTER_ROWS);
        if body_height < 4 {
            return canvas;
        }

        let gap = 2usize;
        let trace_width = if columns >= TRACE_MIN_COLUMNS {
            TRACE_WIDTH
        } else {
            0
        };
        let page_left = 2usize;
        let page_width = columns
            .saturating_sub(
                page_left
                    + 2
                    + if trace_width > 0 {
                        trace_width + gap
                    } else {
                        0
                    },
            )
            .max(20);

        if let Some(composition) = &self.composition {
            render::paint_page(
                &mut canvas,
                &composition.spec,
                body_top as i32,
                page_left as i32,
                page_width,
                body_height,
            );
        } else {
            canvas.put(
                body_top as i32 + 1,
                page_left as i32 + 2,
                "No page yet — send a request to compose one.",
                render::MUTED,
            );
        }

        if trace_width > 0 {
            let trace_left = page_left + page_width + gap;
            let content = self.trace_lines(trace_width.saturating_sub(4));
            let visible = body_height.saturating_sub(4);
            let scroll = self.scroll.min(content.len().saturating_sub(visible));
            let label = if self.show_json {
                "SPEC JSON"
            } else {
                "COMPOSITION TRACE"
            };
            let title = if content.len() > visible {
                format!("{label} ▲ {}/{}", scroll + 1, content.len())
            } else {
                label.to_string()
            };
            canvas.frame(
                body_top as i32,
                trace_left as i32,
                trace_width,
                body_height,
                &title,
            );
            for (offset, line) in content.iter().skip(scroll).take(visible).enumerate() {
                canvas.put(
                    body_top as i32 + 2 + offset as i32,
                    trace_left as i32 + 2,
                    line,
                    render::FG,
                );
            }
        }

        let divider_row = rows.saturating_sub(FOOTER_ROWS) as i32;
        canvas.put(
            divider_row,
            2,
            &"─".repeat(columns.saturating_sub(4)),
            render::DIM,
        );

        let (prompt, color) = match self.mode {
            Mode::Input(Purpose::New) => ("new › ", render::GREEN),
            Mode::Input(Purpose::Edit) => ("edit ›", render::AMBER),
            Mode::Browse => ("›", render::MUTED),
        };
        canvas.put(divider_row + 1, 2, prompt, color);
        let input_column = 2 + prompt.chars().count() as i32 + 1;
        let shown = if self.input.is_empty() && matches!(self.mode, Mode::Browse) {
            "press c compose, e edit, t trace/json, ↑/↓ scroll, q quit".to_string()
        } else {
            self.input.clone()
        };
        canvas.put(divider_row + 1, input_column, &shown, render::FG);
        if matches!(self.mode, Mode::Input(_)) {
            let cursor = input_column + self.input.chars().count() as i32;
            canvas.put(divider_row + 1, cursor, "▏", color);
        }
        canvas.put(divider_row + 2, 2, &self.status, render::MUTED);
        let help = "↑/↓ PgUp/PgDn scroll   1/2 preset   t trace/json   q quit";
        canvas.put(
            divider_row + 2,
            columns as i32 - help.chars().count() as i32 - 2,
            help,
            render::MUTED,
        );

        canvas
    }

    /// 面板内容行数与可见行数；不显示面板时返回 `None`。
    fn panel_metrics(&self, columns: usize, rows: usize) -> Option<(Vec<String>, usize)> {
        if columns < TRACE_MIN_COLUMNS {
            return None;
        }
        let body_height = rows.saturating_sub(HEADER_ROWS + FOOTER_ROWS);
        let visible = body_height.saturating_sub(4);
        if visible == 0 {
            return None;
        }
        Some((self.trace_lines(TRACE_WIDTH.saturating_sub(4)), visible))
    }

    fn trace_lines(&self, width: usize) -> Vec<String> {
        let Some(composition) = &self.composition else {
            return Vec::new();
        };
        if self.show_json {
            return pretty(&composition.spec, width);
        }
        let mut lines = Vec::new();
        for step in &composition.trace {
            let short = shorten(&step.id);
            let bar_width = 10usize;
            let filled = ((step.confidence.clamp(0.0, 1.0)) * bar_width as f64).round() as usize;
            let bar = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(bar_width.saturating_sub(filled))
            );
            lines.push(format!(
                "{short:<16} {bar} {:.2} {:>4.0}ms",
                step.confidence, step.ms
            ));
            if let Some(choice) = step
                .detail
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            {
                lines.push(format!("   › {}", choice.0));
            } else {
                lines.push(format!("   › {}", step.choice));
            }
            let probabilities: Vec<String> = step
                .detail
                .iter()
                .take(3)
                .map(|(label, value)| format!("{} {:.2}", shorten(label), value))
                .collect();
            if !probabilities.is_empty() {
                lines.push(format!("     {}", probabilities.join("  ")));
            }
            lines.push(String::new());
        }
        lines.push(format!(
            "{} decision step(s) · {:.0} ms total",
            composition.trace.len(),
            composition.elapsed_ms
        ));
        lines
            .into_iter()
            .map(|line| truncate(&line, width))
            .collect()
    }
}

fn pretty(spec: &Spec, width: usize) -> Vec<String> {
    let text = serde_json::to_string_pretty(spec).unwrap_or_else(|_| "{}".to_string());
    text.lines().map(|line| truncate(line, width)).collect()
}

fn shorten(label: &str) -> &str {
    label
        .strip_prefix("include_")
        .or_else(|| label.strip_prefix("order_"))
        .or_else(|| label.strip_prefix("slot_"))
        .or_else(|| label.strip_prefix("remove:"))
        .or_else(|| label.strip_prefix("add:"))
        .or_else(|| label.strip_prefix("top:"))
        .unwrap_or(label)
}
