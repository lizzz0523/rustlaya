//! 确定性贪吃蛇规则、哈密顿环安全规划与棋盘快照。

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::bail;

/// 四个方向，顺序同时用于模型选项与界面展示。
pub const DIRECTIONS: [&str; 4] = ["UP", "DOWN", "LEFT", "RIGHT"];

/// 单个方向的可走性、安全性评估。
#[derive(Clone, Debug)]
pub struct MoveInfo {
    pub direction: String,
    pub legal: bool,
    pub safe: bool,
    pub advance: usize,
    pub reason: String,
    pub eats: bool,
}

/// 供渲染使用的棋盘快照。
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub width: i32,
    pub height: i32,
    pub body: Vec<[i32; 2]>,
    pub food: Option<[i32; 2]>,
    pub score: i64,
    pub length: usize,
    pub alive: bool,
    pub won: bool,
}

/// 确定性对局；蛇身沿哈密顿环初始化并随食物推进。
#[derive(Clone)]
pub struct SnakeGame {
    width: i32,
    height: i32,
    cycle: Vec<(i32, i32)>,
    indices: HashMap<(i32, i32), usize>,
    capacity: usize,
    body: VecDeque<(i32, i32)>,
    score: i64,
    alive: bool,
    won: bool,
    food: Option<(i32, i32)>,
    random: Random,
}

impl SnakeGame {
    pub fn new(width: i32, height: i32, seed: i64, initial_length: usize) -> anyhow::Result<Self> {
        let cycle = hamiltonian_cycle(width, height)?;
        let capacity = (width as usize) * (height as usize);
        if initial_length < 2 || initial_length >= capacity {
            bail!("Initial length must be >= 2 and smaller than the board");
        }

        let indices = cycle
            .iter()
            .enumerate()
            .map(|(index, cell)| (*cell, index))
            .collect::<HashMap<_, _>>();
        let start = indices[&(width / 2, height / 2)];

        let body = (0..initial_length)
            .map(|i| cycle[(start + capacity - i) % capacity])
            .collect::<VecDeque<_>>();

        let mut game = Self {
            width,
            height,
            cycle,
            indices,
            capacity,
            body,
            score: 0,
            alive: true,
            won: false,
            food: None,
            random: Random::new(seed as u64),
        };
        game.food = game.spawn_food();

        Ok(game)
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub fn score(&self) -> i64 {
        self.score
    }

    pub fn length(&self) -> usize {
        self.body.len()
    }

    pub fn alive(&self) -> bool {
        self.alive
    }

    pub fn won(&self) -> bool {
        self.won
    }

    pub fn moves(&self) -> Vec<MoveInfo> {
        if !self.alive || self.won {
            return Vec::new();
        }

        let head_index = self.indices[&self.head()];
        let tail_distance =
            (self.indices[self.body.back().unwrap()] + self.capacity - head_index) % self.capacity;
        let food_distance = match self.food {
            Some(food) => (self.indices[&food] + self.capacity - head_index) % self.capacity,
            None => self.capacity,
        };

        let mut moves = Vec::with_capacity(DIRECTIONS.len());
        for direction in DIRECTIONS {
            let reason = self.legal_reason(direction);
            let legal = reason == "legal";

            let target = self.target(direction);
            let advance = (self.indices.get(&target).copied().unwrap_or(head_index)
                + self.capacity
                - head_index)
                % self.capacity;
            let eats = self.food == Some(target);

            let mut safe = legal;
            let mut reason = reason;
            if safe && (advance > tail_distance || (advance == tail_distance && eats)) {
                safe = false;
                reason = "would cross the tail".to_string();
            }
            if safe && (advance == 0 || advance > food_distance) {
                safe = false;
                reason = "would skip the food on the safe route".to_string();
            }

            moves.push(MoveInfo {
                direction: direction.to_string(),
                legal,
                safe,
                advance,
                reason,
                eats,
            });
        }

        moves
    }

    pub fn food_reachability(&self) -> (bool, usize) {
        let head = self.head();
        let blocked: HashSet<(i32, i32)> = self
            .body
            .iter()
            .copied()
            .filter(|cell| *cell != head)
            .collect();

        let mut visited: HashSet<(i32, i32)> = HashSet::new();
        visited.insert(head);

        let mut queue = VecDeque::new();
        queue.push_back(head);

        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                let cell = (x + dx, y + dy);
                if (0..self.width).contains(&cell.0)
                    && (0..self.height).contains(&cell.1)
                    && !blocked.contains(&cell)
                    && !visited.contains(&cell)
                {
                    visited.insert(cell);
                    queue.push_back(cell);
                }
            }
        }

        (
            self.food.is_some_and(|food| visited.contains(&food)),
            visited.len(),
        )
    }

    pub fn step(&mut self, direction: &str) -> anyhow::Result<bool> {
        if !self.alive || self.won {
            bail!("Cannot step a finished game");
        }
        if !DIRECTIONS.contains(&direction) {
            bail!("Unknown direction: {direction}");
        }

        let reason = self.legal_reason(direction);
        if reason != "legal" {
            self.alive = false;
            return Ok(false);
        }

        let target = self.target(direction);
        self.body.push_front(target);

        if self.food == Some(target) {
            self.score += 1;
            if self.body.len() == self.capacity {
                self.won = true;
                self.food = None;
            } else {
                self.food = self.spawn_food();
            }
            return Ok(true);
        }

        self.body.pop_back();

        Ok(false)
    }

    pub fn cycle_order_valid(&self) -> bool {
        let indices: Vec<usize> = self
            .body
            .iter()
            .rev()
            .map(|cell| self.indices[cell])
            .collect();

        let mut sum = 0usize;
        for pair in indices.windows(2) {
            let distance = (pair[1] + self.capacity - pair[0]) % self.capacity;
            if distance == 0 {
                return false;
            }
            sum += distance;
        }

        sum < self.capacity
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            width: self.width,
            height: self.height,
            body: self.body.iter().map(|cell| [cell.0, cell.1]).collect(),
            food: self.food.map(|cell| [cell.0, cell.1]),
            score: self.score,
            length: self.body.len(),
            alive: self.alive,
            won: self.won,
        }
    }

    fn head(&self) -> (i32, i32) {
        self.body[0]
    }

    fn spawn_food(&mut self) -> Option<(i32, i32)> {
        let occupied: HashSet<(i32, i32)> = self.body.iter().copied().collect();
        let empty: Vec<(i32, i32)> = self
            .cycle
            .iter()
            .copied()
            .filter(|cell| !occupied.contains(cell))
            .collect();
        if empty.is_empty() {
            None
        } else {
            Some(empty[self.random.below(empty.len())])
        }
    }

    fn target(&self, direction: &str) -> (i32, i32) {
        let (dx, dy) = direction_delta(direction);
        let (x, y) = self.head();
        (x + dx, y + dy)
    }

    fn legal_reason(&self, direction: &str) -> String {
        let (x, y) = self.target(direction);
        if !(0..self.width).contains(&x) || !(0..self.height).contains(&y) {
            return "wall".to_string();
        }

        let cell = (x, y);
        if self.body.get(1) == Some(&cell) {
            return "reverse".to_string();
        }

        let mut occupied: HashSet<(i32, i32)> = self.body.iter().copied().collect();
        if self.food != Some(cell)
            && let Some(tail) = self.body.back()
        {
            occupied.remove(tail);
        }
        if occupied.contains(&cell) {
            "body".to_string()
        } else {
            "legal".to_string()
        }
    }
}

fn direction_delta(direction: &str) -> (i32, i32) {
    match direction {
        "UP" => (0, -1),
        "DOWN" => (0, 1),
        "LEFT" => (-1, 0),
        "RIGHT" => (1, 0),
        _ => unreachable!("unknown direction"),
    }
}

fn hamiltonian_cycle(width: i32, height: i32) -> anyhow::Result<Vec<(i32, i32)>> {
    if width.min(height) < 4 || (width % 2 == 1 && height % 2 == 1) {
        bail!("Board dimensions must be >= 4, with at least one even dimension");
    }
    if height % 2 == 1 {
        let swapped = hamiltonian_cycle(height, width)?;
        return Ok(swapped.into_iter().map(|(x, y)| (y, x)).collect());
    }

    let mut path = vec![(0, 0)];
    for y in 0..height {
        if y % 2 == 0 {
            for x in 1..width {
                path.push((x, y));
            }
        } else {
            let mut x = width - 1;
            while x > 0 {
                path.push((x, y));
                x -= 1;
            }
        }
    }

    let mut y = height - 1;
    while y > 0 {
        path.push((0, y));
        y -= 1;
    }

    Ok(path)
}

#[derive(Clone)]
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}
