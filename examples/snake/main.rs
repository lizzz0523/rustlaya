//! 本地交互式贪吃蛇：由 Laya 在 move/risk/food 三类问题中决策，带哈密顿环安全护盾。

mod game;
mod policy;
mod ui;

use std::collections::VecDeque;
use std::env;
use std::io::{self, IsTerminal};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use rustlaya::Laya;
use serde_json::{Value, json};

use crate::game::SnakeGame;
use crate::policy::{Decision, Policy, Prompt};
use crate::ui::{Stats, compose, layout_size, show_message};

const DEFAULT_REPOSITORY: &str = "convaiinnovations/laya-multilingual";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let mut args = Args::parse()?;
    let model = args
        .model
        .clone()
        .or_else(|| env::var("LAYA_MODEL").ok())
        .unwrap_or_else(|| DEFAULT_REPOSITORY.to_string());

    let mut game = SnakeGame::new(args.width, args.height, args.seed, args.initial_length)?;
    eprintln!("Loading Laya weights for {model}...");
    let laya = Laya::load(&model)?;
    let policy = Policy::new(!args.unassisted, args.prompt);
    eprintln!("Model ready on {}.", hardware_name());

    let mut warm = SnakeGame::new(
        args.width,
        args.height,
        args.seed + 10_000,
        args.initial_length,
    )?;
    for _ in 0..6 {
        let decision = policy.decide(&warm, &laya)?;
        warm.step(&decision.executed)?;
        if !warm.alive() {
            break;
        }
    }

    if !io::stdout().is_terminal() {
        bail!("Interactive display needs a TTY.");
    }
    let terminal = Terminal::enter(!args.no_alt_screen)?;

    let started = Instant::now();
    let mut timestamps: VecDeque<Instant> = VecDeque::new();
    let mut inference: Vec<f64> = Vec::new();
    let mut calls = 0usize;
    let mut total_steps = 0usize;
    let mut deaths = 0usize;
    let mut stats = Stats {
        hardware: hardware_name(),
        guarded: policy.guarded,
        interventions: 0,
        best: 0,
        round: 1,
        paused: false,
        elapsed: 0.0,
        steps_per_second: 0.0,
    };
    let mut last_board = game.snapshot();
    let mut last_decision: Option<Decision> = None;

    loop {
        let now = Instant::now();
        if let Some(duration) = args.duration
            && now.duration_since(started).as_secs_f64() >= duration
        {
            break;
        }
        if let Some(steps) = args.steps
            && total_steps >= steps
        {
            break;
        }

        let actions = Actions::poll()?;
        if actions.quit {
            break;
        }
        if actions.pause {
            stats.paused = !stats.paused;
        }
        if actions.fps_delta != 0 {
            args.fps = (args.fps + f64::from(actions.fps_delta)).clamp(1.0, 240.0);
        }
        if actions.reset {
            stats.round += 1;
            game = SnakeGame::new(
                args.width,
                args.height,
                args.seed + stats.round as i64 - 1,
                args.initial_length,
            )?;
            last_board = game.snapshot();
            last_decision = None;
        }

        if stats.paused {
            stats.elapsed = now.duration_since(started).as_secs_f64();
            compose(&last_board, last_decision.as_ref(), &stats).draw()?;
            std::thread::sleep(Duration::from_millis(30));
            continue;
        }

        let (columns, rows) = terminal::size()?;
        let (minimum_width, minimum_height) = layout_size(game.width(), game.height());
        if usize::from(columns) < minimum_width || usize::from(rows) < minimum_height {
            show_message(&format!(
                "Resize terminal to at least {minimum_width} columns × {minimum_height} rows.\n\
                 The game is waiting. Q quits."
            ))?;
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }

        let tick_start = Instant::now();
        let decision = policy.decide(&game, &laya)?;
        calls += 1;
        inference.push(decision.inference_ms);
        stats.interventions += usize::from(decision.intervened);
        let shown = Instant::now();
        timestamps.push_back(shown);
        while timestamps.len() > 60 {
            timestamps.pop_front();
        }
        stats.elapsed = shown.duration_since(started).as_secs_f64();
        stats.steps_per_second = if timestamps.len() > 1 {
            let first = *timestamps.front().unwrap();
            let last = *timestamps.back().unwrap();
            (timestamps.len() - 1) as f64 / last.duration_since(first).as_secs_f64()
        } else {
            0.0
        };
        stats.best = stats.best.max(game.score());

        last_board = game.snapshot();
        last_decision = Some(decision.clone());
        compose(&last_board, last_decision.as_ref(), &stats).draw()?;

        if !args.max_speed {
            let remaining = 1.0 / args.fps - tick_start.elapsed().as_secs_f64();
            if remaining > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(remaining));
            }
        }
        game.step(&decision.executed)?;
        total_steps += 1;
        stats.best = stats.best.max(game.score());
        if policy.guarded && game.alive() && !game.cycle_order_valid() {
            bail!("Guarded game broke its cycle-order invariant");
        }

        if !game.alive() || game.won() {
            deaths += usize::from(!game.alive());
            if args.unassisted {
                break;
            }
            compose(&game.snapshot(), None, &stats).draw()?;
            std::thread::sleep(Duration::from_secs(1));
            stats.round += 1;
            game = SnakeGame::new(
                args.width,
                args.height,
                args.seed + stats.round as i64 - 1,
                args.initial_length,
            )?;
        }
    }

    drop(terminal);

    let seconds = started.elapsed().as_secs_f64();
    let summary = json!({
        "steps": total_steps,
        "inference_calls": calls,
        "seconds": seconds,
        "steps_per_second": if seconds > 0.0 { total_steps as f64 / seconds } else { 0.0 },
        "score": game.score(),
        "length": game.length(),
        "best_score": stats.best,
        "interventions": stats.interventions,
        "deaths": deaths,
        "guarded": policy.guarded,
        "network": "offline",
        "mean_inference_ms": if inference.is_empty() {
            Value::Null
        } else {
            json!(inference.iter().sum::<f64>() / inference.len() as f64)
        },
    });
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
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
    prompt: Prompt,
    width: i32,
    height: i32,
    seed: i64,
    initial_length: usize,
    fps: f64,
    max_speed: bool,
    duration: Option<f64>,
    steps: Option<usize>,
    unassisted: bool,
    no_alt_screen: bool,
}

impl Args {
    fn parse() -> anyhow::Result<Self> {
        let mut args = Self {
            prompt: Prompt::Compact,
            width: 24,
            height: 16,
            seed: 7,
            initial_length: 6,
            fps: 12.0,
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
                "--prompt" => {
                    let prompt = value!(&flag);
                    args.prompt = Prompt::parse(&prompt)?;
                }
                "--width" => args.width = value!(&flag).parse()?,
                "--height" => args.height = value!(&flag).parse()?,
                "--seed" => args.seed = value!(&flag).parse()?,
                "--initial-length" => args.initial_length = value!(&flag).parse()?,
                "--fps" => args.fps = value!(&flag).parse()?,
                "--steps" => args.steps = Some(value!(&flag).parse()?),
                "--duration" => args.duration = Some(value!(&flag).parse()?),
                "--max-speed" => args.max_speed = true,
                "--unassisted" => args.unassisted = true,
                "--no-alt-screen" => args.no_alt_screen = true,
                "--optimize" => {}
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }
        if !args.fps.is_finite() || args.fps <= 0.0 {
            bail!("--fps must be a positive finite number");
        }
        if args.steps == Some(0) {
            bail!("--steps must be positive");
        }
        if let Some(duration) = args.duration
            && (!duration.is_finite() || duration <= 0.0)
        {
            bail!("--duration must be a positive finite number");
        }
        Ok(args)
    }
}

fn print_help() {
    println!(
        "laya-snake (rust)\n\n\
         Options:\n\
         \x20 --model <id|dir>        Local model directory or Hugging Face id\n\
         \x20 --prompt compact|detailed  Prompt style (default: compact)\n\
         \x20 --width <n>             Board width (default: 24)\n\
         \x20 --height <n>            Board height (default: 16)\n\
         \x20 --seed <n>              Deterministic seed (default: 7)\n\
         \x20 --initial-length <n>    Starting snake length (default: 6)\n\
         \x20 --fps <n>               Game decisions per second (default: 12)\n\
         \x20 --max-speed             One move per completed inference\n\
         \x20 --duration <seconds>    Stop after this many seconds\n\
         \x20 --steps <n>             Stop after this many game steps\n\
         \x20 --unassisted            Execute raw Laya top-1; disable the safety shield"
    );
}

#[derive(Default)]
struct Actions {
    pause: bool,
    fps_delta: i32,
    reset: bool,
    quit: bool,
}

impl Actions {
    fn poll() -> io::Result<Self> {
        let mut actions = Self::default();
        while event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => actions.quit = true,
                    KeyCode::Char(' ') => actions.pause = true,
                    KeyCode::Up | KeyCode::Char('+') => actions.fps_delta += 2,
                    KeyCode::Down | KeyCode::Char('-') => actions.fps_delta -= 2,
                    KeyCode::Char('r') | KeyCode::Char('R') => actions.reset = true,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        actions.quit = true;
                    }
                    _ => {}
                }
            }
        }
        Ok(actions)
    }
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
