//! 固定单元格的 truecolor 终端画布，以及 compose / draw。

use std::io::{self, Write};

use crossterm::queue;
use crossterm::terminal::{self, ClearType};

use crate::game::{DIRECTIONS, Snapshot};
use crate::policy::Decision;

const BG: &str = "#090f13";
const FG: &str = "#e3f3ef";
const MUTED: &str = "#68868c";
const DIM: &str = "#20353c";
const GREEN: &str = "#62f5b5";
const AMBER: &str = "#ffce73";
const RED: &str = "#ff7c8c";
const CYAN: &str = "#8ad8e9";

const DIGITS: [(&str, &str, &str); 10] = [
    ("█▀█", "█ █", "▀▀▀"),
    ("▄█ ", " █ ", "▀▀▀"),
    ("▀▀█", "█▀▀", "▀▀▀"),
    ("▀▀█", "▀▀█", "▀▀▀"),
    ("█ █", "▀▀█", "  ▀"),
    ("█▀▀", "▀▀█", "▀▀▀"),
    ("█▀▀", "█▀█", "▀▀▀"),
    ("▀▀█", "  █", "  ▀"),
    ("█▀█", "█▀█", "▀▀▀"),
    ("█▀█", "▀▀█", "▀▀▀"),
];

/// 界面上与决策无关的运行统计。
pub struct Stats {
    pub hardware: String,
    pub guarded: bool,
    pub interventions: usize,
    pub best: i64,
    pub round: usize,
    pub paused: bool,
    pub elapsed: f64,
    pub steps_per_second: f64,
}

pub struct Canvas {
    width: usize,
    height: usize,
    chars: Vec<Vec<char>>,
    styles: Vec<Vec<String>>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            chars: vec![vec![' '; width]; height],
            styles: vec![vec![FG.to_string(); width]; height],
        }
    }

    fn put(&mut self, row: i32, column: i32, text: &str, color: &str) {
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

    fn bar(&mut self, row: i32, column: i32, value: f64, length: usize, color: &str) {
        let count = (value.clamp(0.0, 1.0) * length as f64).round() as usize;
        self.put(row, column, &"━".repeat(length), DIM);
        self.put(row, column, &"━".repeat(count), color);
    }

    fn number(&mut self, row: i32, column: i32, value: i64, color: &str) {
        for (index, digit) in format!("{value:03}").chars().enumerate() {
            let glyphs = DIGITS[digit.to_digit(10).unwrap_or(0) as usize];
            let x = column + index as i32 * 4;
            self.put(row, x, glyphs.0, color);
            self.put(row + 1, x, glyphs.1, color);
            self.put(row + 2, x, glyphs.2, color);
        }
    }

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
}

pub fn compose(game: &Snapshot, decision: Option<&Decision>, stats: &Stats) -> Canvas {
    let (width, height) = layout_size(game.width, game.height);
    let mut c = Canvas::new(width, height);
    let left = 3i32;
    let right = (game.width * 2 + 10).max(58);
    let side = width as i32 - right - 4;
    let top = 6i32;
    let bottom = top + game.height + 1;

    let state = if stats.paused {
        "PAUSED"
    } else if game.won {
        "BOARD CLEAR"
    } else if !game.alive {
        "GAME OVER"
    } else {
        "LIVE"
    };
    let state_color = if game.alive { GREEN } else { RED };
    c.put(1, left, "LAYA  /  LOCAL INTELLIGENCE", MUTED);
    c.put(
        1,
        width as i32 - state.chars().count() as i32 - 3,
        state,
        state_color,
    );
    c.put(2, left, &"─".repeat(width - 6), DIM);
    c.put(4, left, "S N A K E", FG);
    c.put(4, left + 31, &format!("ROUND {:02}", stats.round), MUTED);
    c.put(
        top,
        left,
        &format!("┌{}┐", "─".repeat(game.width as usize * 2)),
        DIM,
    );
    c.put(
        bottom,
        left,
        &format!("└{}┘", "─".repeat(game.width as usize * 2)),
        DIM,
    );
    for y in 0..game.height {
        c.put(top + 1 + y, left, "│", DIM);
        c.put(top + 1 + y, left + game.width * 2 + 1, "│", DIM);
        c.put(
            top + 1 + y,
            left + 1,
            &"· ".repeat(game.width as usize),
            "#13272e",
        );
    }
    let length = game.body.len();
    for (index, cell) in game.body.iter().enumerate().rev() {
        let fraction = 1.0 - index as f64 / length.max(1) as f64;
        let color = if index == 0 {
            "#dcfff0".to_string()
        } else {
            let r = (18.0 + 64.0 * fraction) as u8;
            let g = (73.0 + 150.0 * fraction) as u8;
            let b = (57.0 + 102.0 * fraction) as u8;
            format!("#{r:02x}{g:02x}{b:02x}")
        };
        c.put(top + cell[1] + 1, left + 1 + 2 * cell[0], "██", &color);
    }
    if let Some(food) = game.food {
        c.put(top + food[1] + 1, left + 1 + 2 * food[0], "● ", AMBER);
    }
    for (offset, label, value, color) in [
        (0, "SCORE", game.score, GREEN),
        (18, "LENGTH", game.length as i64, FG),
        (36, "BEST", stats.best, MUTED),
    ] {
        c.put(bottom + 2, left + offset, label, MUTED);
        c.number(bottom + 3, left + offset, value, color);
    }
    let fill = game.length as f64 / (game.width * game.height) as f64;
    c.bar(bottom + 7, left, fill, 41, GREEN);
    c.put(
        bottom + 7,
        left + 43,
        &format!("{:4.1}%", 100.0 * fill),
        MUTED,
    );

    c.put(4, right, "Laya Candle", GREEN);
    c.put(5, right, &format!("{} · Local", stats.hardware), MUTED);
    c.put(7, right, "NEXT MOVE", FG);
    c.put(7, right + 15, "MODEL PROBABILITIES", MUTED);
    let proposed = decision.and_then(|d| d.proposed.as_deref());
    for (index, direction) in DIRECTIONS.iter().enumerate() {
        let row = 9 + index as i32;
        let probability = decision
            .and_then(|d| d.probabilities.get(*direction).copied())
            .unwrap_or(0.0);
        let selected = proposed == Some(*direction);
        let color = if selected { GREEN } else { MUTED };
        c.put(
            row,
            right,
            &format!("{} {:<5}", if selected { '›' } else { ' ' }, direction),
            color,
        );
        let count = (probability * 18.0).round().clamp(0.0, 18.0) as usize;
        c.put(row, right + 9, &"░".repeat(18), DIM);
        c.put(row, right + 9, &"█".repeat(count), color);
        c.put(row, right + 29, &format!("{probability:.2}"), color);
    }
    c.put(14, right, "EXECUTING", MUTED);
    c.put(
        14,
        right + 12,
        decision.map_or("—", |d| d.executed.as_str()),
        GREEN,
    );
    if decision.is_some_and(|d| d.intervened) {
        c.put(14, right + 20, "SHIELD", AMBER);
    }
    c.put(16, right, "DEAD-END RISK", MUTED);
    let risk = decision.map_or(0.0, |d| d.dead_end_risk);
    let risk_color = if risk < 0.5 { AMBER } else { RED };
    c.bar(17, right, risk, (side - 9).min(24) as usize, risk_color);
    c.put(17, right + 29, &format!("{risk:.2}"), risk_color);
    c.put(19, right, "FOOD REACHABLE", MUTED);
    let reachable = decision.map_or(0.0, |d| d.food_reachable);
    c.bar(20, right, reachable, (side - 9).min(24) as usize, CYAN);
    c.put(20, right + 29, &format!("{reachable:.2}"), CYAN);
    c.put(22, right, "INFERENCE", MUTED);
    c.put(
        22,
        right + 18,
        &format!("{:5.1} ms", decision.map_or(0.0, |d| d.inference_ms)),
        FG,
    );
    c.put(23, right, "DECISIONS", MUTED);
    c.put(
        23,
        right + 18,
        &format!("{:5.1} /s", stats.steps_per_second),
        FG,
    );
    c.put(24, right, "OUTPUT TOKENS", MUTED);
    c.put(
        24,
        right + 18,
        &decision.map_or(0, |d| d.output_tokens).to_string(),
        FG,
    );
    c.put(25, right, "NETWORK", MUTED);
    c.put(25, right + 18, "OFFLINE", GREEN);
    c.put(26, right, "ENGINE", MUTED);
    c.put(26, right + 18, "candle · FP16", MUTED);
    c.put(
        28,
        right,
        if stats.guarded {
            "Laya + cycle safety"
        } else {
            "Laya · shield OFF"
        },
        MUTED,
    );
    c.put(
        29,
        right,
        &format!("Shield interventions  {:04}", stats.interventions),
        AMBER,
    );
    c.put(height as i32 - 3, left, &"─".repeat(width - 6), DIM);
    c.put(
        height as i32 - 2,
        left,
        "SPACE pause   ↑/↓ speed   R reset   Q quit",
        MUTED,
    );
    let clock = format!(
        "{:02}:{:02}",
        stats.elapsed as i64 / 60,
        stats.elapsed as i64 % 60
    );
    c.put(
        height as i32 - 2,
        right,
        &format!("ESTIMATES BY LAYA            {clock}"),
        MUTED,
    );
    c
}

pub fn layout_size(width: i32, height: i32) -> (usize, usize) {
    (
        104usize.max(width as usize * 2 + 50),
        35usize.max(height as usize + 19),
    )
}

pub fn show_message(message: &str) -> io::Result<()> {
    let mut stdout = io::stdout();
    queue!(stdout, terminal::Clear(ClearType::All))?;
    stdout.write_all(message.as_bytes())?;
    stdout.flush()
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
