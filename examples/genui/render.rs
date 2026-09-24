//! Spec -> 终端画布的渲染器，对应 json-render 各端 renderer 的终端版本。
//!
//! 渲染器只认识 catalog 中的组件 `kind`，并按 props 绘制。它是纯“读 Spec”的一方，
//! 不关心 Spec 是怎么被 Laya 决策出来的。

use std::io::{self, Write};

use serde_json::Value;

use crate::spec::{Element, Spec};
use crate::text_layout::{truncate, wrap};

pub const BG: &str = "#0b0f14";
pub const FG: &str = "#dbe7ef";
pub const MUTED: &str = "#6b7f8f";
pub const DIM: &str = "#1b2b36";
pub const ACCENT: &str = "#62d5f5";
pub const GREEN: &str = "#62f5b5";
pub const AMBER: &str = "#ffce73";
pub const PURPLE: &str = "#b98cff";

/// 固定单元格的 truecolor 终端画布。
pub struct Canvas {
    width: usize,
    height: usize,
    chars: Vec<Vec<char>>,
    styles: Vec<Vec<String>>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            chars: vec![vec![' '; width]; height],
            styles: vec![vec![FG.to_string(); width]; height],
        }
    }

    pub fn put(&mut self, row: i32, column: i32, text: &str, color: &str) {
        if row < 0 || row as usize >= self.height {
            return;
        }
        for (offset, character) in text.chars().enumerate() {
            let x = column + offset as i32;
            if x >= 0 && (x as usize) < self.width {
                self.chars[row as usize][x as usize] = character;
                self.styles[row as usize][x as usize] = color.to_string();
            }
        }
    }

    fn hline(&mut self, row: i32, column: i32, length: usize, color: &str) {
        self.put(row, column, &"─".repeat(length), color);
    }

    fn vline(&mut self, row: i32, column: i32, length: usize, color: &str) {
        for offset in 0..length {
            self.put(row + offset as i32, column, "│", color);
        }
    }

    pub fn frame(&mut self, top: i32, left: i32, width: usize, height: usize, title: &str) {
        if width < 4 || height < 2 {
            return;
        }
        let inner = width - 2;
        let mut header = format!(" {title} ");
        if header.chars().count() > inner {
            header = format!(" {} ", truncate(title, inner.saturating_sub(4)));
        }
        let padding = inner.saturating_sub(header.chars().count());
        self.put(top, left, "┌", DIM);
        self.put(top, left + 1, &header, ACCENT);
        self.put(
            top,
            left + 1 + header.chars().count() as i32,
            &"─".repeat(padding),
            DIM,
        );
        self.put(top, left + width as i32 - 1, "┐", DIM);
        let right = left + width as i32 - 1;
        for offset in 1..height as i32 - 1 {
            self.put(top + offset, left, "│", DIM);
            self.put(top + offset, right, "│", DIM);
        }
        self.put(top + height as i32 - 1, left, "└", DIM);
        self.hline(top + height as i32 - 1, left + 1, inner, DIM);
        self.put(top + height as i32 - 1, right, "┘", DIM);
    }

    /// 输出带 ANSI 的全屏帧。
    pub fn draw(&self) -> io::Result<()> {
        let (br, bg, bb) = rgb(BG);
        let mut out = String::with_capacity(self.width * self.height * 8);
        out.push_str("\x1b[H");
        for row in 0..self.height {
            out.push_str(&format!("\x1b[0m\x1b[48;2;{br};{bg};{bb}m"));
            let mut current: Option<&str> = None;
            for column in 0..self.width {
                let color = self.styles[row][column].as_str();
                if current != Some(color) {
                    let (r, g, b) = rgb(color);
                    out.push_str(&format!("\x1b[38;2;{r};{g};{b}m"));
                    current = Some(color);
                }
                out.push(self.chars[row][column]);
            }
            out.push_str("\x1b[K");
            if row + 1 < self.height {
                out.push_str("\r\n");
            }
        }
        out.push_str("\x1b[0m");
        let mut stdout = io::stdout();
        stdout.write_all(out.as_bytes())?;
        stdout.flush()
    }

    /// 无 ANSI 的纯文本快照（用于 `--print`）。
    pub fn to_plain(&self) -> String {
        self.chars
            .iter()
            .map(|row| {
                let line: String = row.iter().collect();
                line.trim_end().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// 把一个 Spec 渲染进画布的矩形区域。
pub fn paint_page(
    canvas: &mut Canvas,
    spec: &Spec,
    top: i32,
    left: i32,
    width: usize,
    height: usize,
) {
    let Some(root_id) = spec.root.as_deref() else {
        return;
    };
    let Some(root) = spec.get(root_id) else {
        return;
    };
    let title = str_prop(root, "title").unwrap_or("Generated page");
    canvas.frame(top, left, width, height, title);

    if width < 6 || height < 6 {
        return;
    }
    let inner_top = top + 2;
    let inner_left = left + 2;
    let inner_width = width - 4;
    let inner_height = height - 4;

    let main = root
        .slots
        .get("main")
        .or_else(|| root.slots.get("default"))
        .cloned()
        .unwrap_or_else(|| root.children.clone());
    let aside = root.slots.get("aside").cloned().unwrap_or_default();

    if !main.is_empty() && !aside.is_empty() {
        let main_width = (inner_width * 3) / 5;
        let aside_width = inner_width.saturating_sub(main_width + 3);
        paint_flow(
            canvas,
            spec,
            &main,
            inner_top,
            inner_left,
            main_width,
            inner_height,
        );
        canvas.vline(
            inner_top,
            inner_left + main_width as i32 + 1,
            inner_height,
            DIM,
        );
        paint_flow(
            canvas,
            spec,
            &aside,
            inner_top,
            inner_left + main_width as i32 + 3,
            aside_width,
            inner_height,
        );
    } else {
        paint_flow(
            canvas,
            spec,
            &main,
            inner_top,
            inner_left,
            inner_width,
            inner_height,
        );
    }
}

fn paint_flow(
    canvas: &mut Canvas,
    spec: &Spec,
    children: &[String],
    top: i32,
    left: i32,
    width: usize,
    max_height: usize,
) {
    let mut row = top;
    for id in children {
        let Some(element) = spec.get(id) else {
            continue;
        };
        let consumed = (row - top) as usize;
        if consumed >= max_height {
            break;
        }
        let remaining = max_height - consumed;
        let used = paint_element(canvas, element, row, left, width, remaining);
        row += used.max(0);
    }
}

/// 绘制单个元素，返回占用的行数。
fn paint_element(
    canvas: &mut Canvas,
    element: &Element,
    row: i32,
    left: i32,
    width: usize,
    max_height: usize,
) -> i32 {
    if width == 0 {
        return 0;
    }
    match element.kind.as_str() {
        "Heading" => {
            let text = str_prop(element, "text").unwrap_or("");
            canvas.put(row, left, &truncate(text, width), ACCENT);
            canvas.hline(row + 1, left, width.min(60), DIM);
            3
        }
        "Text" => {
            let text = str_prop(element, "text").unwrap_or("");
            let lines = wrap(text, width);
            for (offset, line) in lines.iter().enumerate() {
                canvas.put(row + offset as i32, left, line, MUTED);
            }
            lines.len().min(max_height) as i32 + 1
        }
        "Metric" => {
            let label = str_prop(element, "label").unwrap_or("Metric");
            let value = str_prop(element, "value").unwrap_or("—");
            let delta = str_prop(element, "delta");
            canvas.put(row, left, &truncate(label, width), MUTED);
            canvas.put(
                row + 1,
                left,
                &truncate(&value.to_uppercase(), width),
                AMBER,
            );
            if let Some(delta) = delta {
                let offset = value.chars().count().min(width) as i32 + 1;
                canvas.put(row + 1, left + offset, delta, GREEN);
            }
            3
        }
        "Field" => {
            let label = str_prop(element, "label").unwrap_or("");
            let value = str_prop(element, "value").unwrap_or("");
            canvas.put(row, left, &format!("{label:<16}"), MUTED);
            canvas.put(
                row,
                left + 16,
                &truncate(value, width.saturating_sub(16)),
                FG,
            );
            1
        }
        "Toggle" => {
            let label = str_prop(element, "label").unwrap_or("");
            let on = element
                .props
                .get("on")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            canvas.put(row, left, &format!("{label:<24}"), FG);
            let (marker, color) = if on {
                ("[ ON  ]", GREEN)
            } else {
                ("[ off ]", MUTED)
            };
            canvas.put(row, left + 24, marker, color);
            1
        }
        "BarChart" => {
            let title = str_prop(element, "title").unwrap_or("Chart");
            canvas.put(row, left, &truncate(title, width), MUTED);
            let data = element
                .props
                .get("data")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let max = data
                .iter()
                .filter_map(|entry| entry.get(1).and_then(Value::as_f64))
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let bar_width = width.saturating_sub(18).max(6);
            for (index, entry) in data.iter().enumerate() {
                let label = entry
                    .get(0)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let value = entry.get(1).and_then(Value::as_f64).unwrap_or(0.0);
                let y = row + 1 + index as i32;
                canvas.put(y, left, &format!("{label:<8}"), FG);
                let filled = ((value / max) * bar_width as f64).round() as usize;
                canvas.put(y, left + 9, &"█".repeat(filled.min(bar_width)), AMBER);
                canvas.put(
                    y,
                    left + 9 + bar_width as i32 + 1,
                    &format!("{value:.0}"),
                    MUTED,
                );
            }
            data.len() as i32 + 2
        }
        "Table" => {
            let title = str_prop(element, "title").unwrap_or("Table");
            canvas.put(row, left, &truncate(title, width), MUTED);
            let columns: Vec<String> = element
                .props
                .get("columns")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .map(|item| item.as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default();
            let rows: Vec<Vec<String>> = element
                .props
                .get("rows")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .map(|row| {
                            row.as_array()
                                .map(|cells| {
                                    cells
                                        .iter()
                                        .map(|cell| cell.as_str().unwrap_or("").to_string())
                                        .collect()
                                })
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default();
            if columns.is_empty() {
                return 1;
            }
            let column_width = width / columns.len();
            let mut header = String::new();
            for column in &columns {
                header.push_str(&format!(
                    "{:<column_width$}",
                    truncate(column, column_width)
                ));
            }
            canvas.put(row + 1, left, &truncate(&header, width), FG);
            canvas.hline(row + 2, left, width, DIM);
            for (index, cells) in rows.iter().enumerate() {
                let mut line = String::new();
                for cell in cells {
                    line.push_str(&format!("{:<column_width$}", truncate(cell, column_width)));
                }
                canvas.put(row + 3 + index as i32, left, &truncate(&line, width), MUTED);
            }
            (rows.len() + 4) as i32
        }
        "Badge" => {
            let text = str_prop(element, "text").unwrap_or("");
            canvas.put(row, left, &format!("▐ {text} ▌"), PURPLE);
            2
        }
        "Button" => {
            let label = str_prop(element, "label").unwrap_or("Button");
            canvas.put(row, left, &format!("[ {label} ]"), GREEN);
            2
        }
        "Divider" => {
            canvas.hline(row, left, width, DIM);
            2
        }
        other => {
            canvas.put(row, left, &truncate(other, width), MUTED);
            1
        }
    }
}

fn str_prop<'a>(element: &'a Element, key: &str) -> Option<&'a str> {
    element.props.get(key).and_then(Value::as_str)
}

fn rgb(hex: &str) -> (u8, u8, u8) {
    let hex = hex.trim_start_matches('#');
    let value = u32::from_str_radix(hex, 16).unwrap_or(0);
    (
        ((value >> 16) & 0xFF) as u8,
        ((value >> 8) & 0xFF) as u8,
        (value & 0xFF) as u8,
    )
}
