//! 终端版 Generative UI demo：用 Laya（类型化决策模型）扮演 Jev，把预置候选组装成
//! json-render 风格的 flat Spec，再由终端渲染器画出来。
//!
//! ```text
//! catalog（允许的组件） + candidates（预置实例）
//!        │  choice / noul 问题
//!        ▼
//!  Laya（= Jev）：只做离散选择，不生成自由文本
//!        │  composer 翻译
//!        ▼
//!  flat Spec { root, elements }  ──renderer──▶  终端 UI
//! ```
//!
//! 运行：
//!   cargo run --release --example genui -- --model convaiinnovations/laya-multilingual
//!   cargo run --release --example genui -- --prompt "sales dashboard" --print

mod catalog;
mod composer;
mod render;
mod spec;
mod text_layout;
mod view;

use std::env;
use std::io::{self, IsTerminal};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, bail};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use rustlaya::Laya;

use crate::catalog::{Preset, names, preset as preset_for};
use crate::composer::{Composition, compose, edit};
use crate::render::Canvas;
use crate::view::{Mode, Purpose, UiState, page_rows};

/// 缺省检查点：多语言版，中英文 prompt 都可用。
const DEFAULT_REPOSITORY: &str = "convaiinnovations/laya-multilingual";
const MIN_WIDTH: usize = 104;
const MIN_HEIGHT: usize = 30;
/// `--print` 时画布尺寸。
const PRINT_WIDTH: usize = 92;
const PRINT_HEIGHT: usize = 42;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = Args::parse()?;
    let model = args
        .model
        .clone()
        .or_else(|| env::var("LAYA_MODEL").ok())
        .unwrap_or_else(|| DEFAULT_REPOSITORY.to_string());
    let mut preset = preset_for(&args.preset)
        .with_context(|| format!("unknown preset {}; try one of {:?}", args.preset, names()))?;

    eprintln!("Loading Laya weights for {model}...");
    let laya = Laya::load(&model)?;
    eprintln!("Model ready ({}).", hardware_name());

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| preset.request_hint.to_string());

    if args.print {
        return run_print(&laya, &preset, &prompt, args.edit.as_deref());
    }

    if !io::stdout().is_terminal() {
        bail!("interactive display needs a TTY; use --print for a one-shot render");
    }

    warmup(&laya, &preset)?;
    run_interactive(&laya, &mut preset, args.no_alt_screen)
}

/// 用一次组合预热 Metal kernel，避免第一帧卡顿。
fn warmup(laya: &Laya, preset: &Preset) -> anyhow::Result<()> {
    if let Err(error) = compose(laya, preset, preset.request_hint) {
        eprintln!("warmup composition failed: {error:#}");
    }
    Ok(())
}

fn run_print(
    laya: &Laya,
    preset: &Preset,
    prompt: &str,
    edit_prompt: Option<&str>,
) -> anyhow::Result<()> {
    let composed = compose(laya, preset, prompt)?;
    report(preset, prompt, "compose", &composed);

    let composition = match edit_prompt {
        Some(prompt) => {
            let edited = edit(laya, preset, prompt, &composed.spec)?;
            report(preset, prompt, "edit", &edited);
            edited
        }
        None => composed,
    };

    let mut canvas = Canvas::new(PRINT_WIDTH, PRINT_HEIGHT);
    render::paint_page(
        &mut canvas,
        &composition.spec,
        0,
        0,
        PRINT_WIDTH,
        PRINT_HEIGHT,
    );
    println!("{}", canvas.to_plain());
    println!("\n--- spec.json ---");
    println!("{}", serde_json::to_string_pretty(&composition.spec)?);
    Ok(())
}

fn report(preset: &Preset, prompt: &str, phase: &str, composition: &Composition) {
    eprintln!(
        "[{phase}] preset={} prompt={:?} decisions={} elapsed={:.0} ms",
        preset.name,
        prompt,
        composition.trace.len(),
        composition.elapsed_ms
    );
    for step in &composition.trace {
        eprintln!(
            "  {:<28} => {:<24} conf {:.2}",
            step.id, step.choice, step.confidence
        );
    }
}

fn run_interactive(laya: &Laya, preset: &mut Preset, no_alt_screen: bool) -> anyhow::Result<()> {
    let terminal = Terminal::enter(!no_alt_screen)?;
    let mut ui = UiState::new(preset.name, preset.title);
    ui.input = preset.request_hint.to_string();

    loop {
        let (columns, rows) = terminal::size()?;
        let (columns, rows) = (usize::from(columns), usize::from(rows));

        if columns < MIN_WIDTH || rows < MIN_HEIGHT {
            let mut canvas = Canvas::new(columns.max(1), rows.max(1));
            canvas.put(1, 1, "Terminal too small.", render::FG);
            canvas.put(
                2,
                1,
                &format!("Need at least {MIN_WIDTH}x{MIN_HEIGHT}. Q quits."),
                render::MUTED,
            );
            canvas.draw()?;
            if event::poll(Duration::from_millis(120))?
                && let Event::Key(key) = event::read()?
                && is_quit(key.code)
            {
                break;
            }
            continue;
        }

        ui.compose(columns, rows).draw()?;

        if !event::poll(Duration::from_millis(60))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if is_ctrl_quit(key.code, key.modifiers) {
            break;
        }

        match ui.mode {
            Mode::Input(purpose) => match key.code {
                KeyCode::Enter => {
                    let request = ui.input.trim().to_string();
                    if request.is_empty() {
                        ui.status = "empty request — type something first".to_string();
                    } else {
                        ui.status = "composing with Laya…".to_string();
                        // 先画出状态，再阻塞推理。
                        let (columns, rows) = terminal::size()?;
                        ui.compose(usize::from(columns), usize::from(rows)).draw()?;
                        let result = match purpose {
                            Purpose::New => compose(laya, preset, &request),
                            Purpose::Edit => match ui.composition.as_ref() {
                                Some(existing) => edit(laya, preset, &request, &existing.spec),
                                None => compose(laya, preset, &request),
                            },
                        };
                        match result {
                            Ok(composition) => {
                                ui.status = status_line(&composition);
                                ui.composition = Some(composition);
                            }
                            Err(error) => ui.status = format!("error: {error:#}"),
                        }
                        ui.reset_scroll();
                        ui.mode = Mode::Browse;
                    }
                }
                KeyCode::Esc => {
                    ui.mode = Mode::Browse;
                    ui.status = "cancelled".to_string();
                }
                KeyCode::Backspace => {
                    ui.input.pop();
                }
                KeyCode::Char(character) if !character.is_control() => ui.input.push(character),
                _ => {}
            },
            Mode::Browse => match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => break,
                KeyCode::Char('c') | KeyCode::Char('C') => {
                    ui.mode = Mode::Input(Purpose::New);
                    ui.input = preset.request_hint.to_string();
                    ui.status = "describe a new page".to_string();
                }
                KeyCode::Char('e') | KeyCode::Char('E') => {
                    if ui.composition.is_some() {
                        ui.mode = Mode::Input(Purpose::Edit);
                        ui.input.clear();
                        ui.status = "describe exactly one edit".to_string();
                    } else {
                        ui.status = "compose a page before editing".to_string();
                    }
                }
                KeyCode::Char('t') | KeyCode::Char('T') => {
                    ui.show_json = !ui.show_json;
                    ui.reset_scroll();
                }
                KeyCode::Char('1') => switch_preset(preset, &mut ui, "dashboard"),
                KeyCode::Char('2') => switch_preset(preset, &mut ui, "settings"),
                KeyCode::Up => ui.scroll_by(-1, columns, rows),
                KeyCode::Down => ui.scroll_by(1, columns, rows),
                KeyCode::PageUp => {
                    ui.scroll_by(-(page_rows(rows) as i32), columns, rows);
                }
                KeyCode::PageDown => ui.scroll_by(page_rows(rows) as i32, columns, rows),
                KeyCode::Home => ui.scroll = 0,
                KeyCode::End => ui.scroll_by(i32::MAX, columns, rows),
                _ => {}
            },
        }
    }

    drop(terminal);
    Ok(())
}

fn switch_preset(preset: &mut Preset, ui: &mut UiState, name: &str) {
    let Some(next) = preset_for(name) else {
        return;
    };
    ui.preset = next.name;
    ui.title = next.title;
    ui.input = next.request_hint.to_string();
    ui.mode = Mode::Input(Purpose::New);
    ui.composition = None;
    ui.reset_scroll();
    ui.status = format!("switched to preset {name}");
    *preset = next;
}

fn status_line(composition: &Composition) -> String {
    format!(
        "{} decision step(s) · {:.0} ms · {} element(s)",
        composition.trace.len(),
        composition.elapsed_ms,
        composition.spec.elements.len()
    )
}

fn is_quit(code: KeyCode) -> bool {
    matches!(code, KeyCode::Char('q') | KeyCode::Char('Q'))
}

fn is_ctrl_quit(code: KeyCode, modifiers: crossterm::event::KeyModifiers) -> bool {
    code == KeyCode::Char('c') && modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
}

fn hardware_name() -> String {
    if cfg!(target_os = "macos")
        && let Ok(output) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        && output.status.success()
        && let Ok(text) = String::from_utf8(output.stdout)
    {
        return text.trim().trim_start_matches("Apple ").to_string();
    }
    env::consts::ARCH.to_string()
}

#[derive(Default)]
struct Args {
    model: Option<String>,
    preset: String,
    prompt: Option<String>,
    edit: Option<String>,
    print: bool,
    no_alt_screen: bool,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            preset: "dashboard".to_string(),
            ..Self::default()
        };
        let mut argv = env::args().skip(1);

        macro_rules! value {
            ($flag:expr) => {
                argv.next()
                    .with_context(|| format!("{} expects a value", $flag))?
            };
        }

        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--model" => args.model = Some(value!(&flag)),
                "--preset" => args.preset = value!(&flag),
                "--prompt" => args.prompt = Some(value!(&flag)),
                "--edit" => args.edit = Some(value!(&flag)),
                "--print" => args.print = true,
                "--no-alt-screen" => args.no_alt_screen = true,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }
        Ok(args)
    }
}

fn print_help() {
    println!(
        "laya-genui — terminal Generative UI demo (Laya as Jev + json-render style Spec)\n\n\
         Options:\n\
         \x20 --model <id|dir>      Local model directory or Hugging Face id\n\
         \x20 --preset <name>       dashboard | settings (default: dashboard)\n\
         \x20 --prompt <text>       Compose once from this request\n\
         \x20 --edit <text>         After composing, apply one edit (used with --print)\n\
         \x20 --print               Compose once, print the rendered page + JSON, exit\n\
         \x20 --no-alt-screen       Do not switch to the alternate screen\n\n\
         Interactive keys:\n\
         \x20 Enter submit   Esc cancel   c new   e edit   t trace/json   1/2 preset\n\
         \x20 ↑/↓ PgUp/PgDn Home/End scroll the trace/JSON panel   q quit"
    );
}

struct Terminal {
    alt_screen: bool,
}

impl Terminal {
    fn enter(alt_screen: bool) -> anyhow::Result<Self> {
        terminal::enable_raw_mode().context("enabling raw mode")?;
        let mut stdout = io::stdout();
        if alt_screen {
            execute!(stdout, EnterAlternateScreen, Hide).context("entering alternate screen")?;
        } else {
            execute!(stdout, Hide).context("hiding cursor")?;
        }
        Ok(Self { alt_screen })
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut stdout = io::stdout();
        if self.alt_screen {
            let _ = execute!(stdout, Show, LeaveAlternateScreen);
        } else {
            let _ = execute!(stdout, Show);
        }
    }
}
