//! HackerNews API 抓取：取新帖 id 列表，并受限并发拉取每篇 story 详情。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

#[derive(Debug, Serialize, Deserialize)]
pub struct Story {
    pub id: u64,
    pub by: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub descendants: Option<u64>,
    pub score: Option<u64>,
    pub time: Option<u64>,
    pub title: Option<String>,
    pub text: Option<String>,
    pub url: Option<String>,
    pub kids: Option<Vec<u64>>,
    pub parent: Option<u64>,
}

/// 单篇 story 的本地缓存目录（仓库根下，与启动时的 CWD 无关）。
fn cache_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/hn/items")
}

/// 读取本地缓存的 story，文件缺失或损坏时返回 `None`。
async fn load_cached(path: &Path) -> Option<Story> {
    let bytes = tokio::fs::read(path).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// 将 story 写入本地缓存，先写临时文件再原子重命名，失败只记录不中断。
async fn store_cached(path: &Path, story: &Story) {
    let Ok(bytes) = serde_json::to_vec_pretty(story) else {
        return;
    };
    let tmp = path.with_extension("json.tmp");
    if let Err(err) = tokio::fs::write(&tmp, &bytes).await {
        eprintln!("cache write failed for {}: {err}", path.display());
        return;
    }
    if let Err(err) = tokio::fs::rename(&tmp, path).await {
        eprintln!("cache rename failed for {}: {err}", path.display());
    }
}

/// 抓取 HackerNews 最新 story 列表，单篇失败只记录不中断（并发上限 10）。
///
/// 每篇 story 按 id 缓存在本地，重复运行时命中缓存则不再发起请求。
pub async fn fetch_stories() -> anyhow::Result<Vec<Story>> {
    let client = reqwest::Client::new();

    let ids: Vec<u64> = client
        .get("https://hacker-news.firebaseio.com/v0/newstories.json")
        .send()
        .await?
        .json()
        .await?;

    let semaphore = Arc::new(Semaphore::new(10));
    let client = Arc::new(client);

    let dir = cache_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let mut handles = Vec::new();
    for id in ids.into_iter() {
        let semaphore = semaphore.clone();
        let client = client.clone();
        let path = dir.join(format!("{id}.json"));
        handles.push(tokio::spawn(async move {
            if let Some(story) = load_cached(&path).await {
                return Ok::<_, reqwest::Error>(story);
            }

            let _permit = semaphore.acquire().await.unwrap();
            let story: Story = client
                .get(format!(
                    "https://hacker-news.firebaseio.com/v0/item/{id}/.json"
                ))
                .send()
                .await?
                .json()
                .await?;
            store_cached(&path, &story).await;

            Ok(story)
        }));
    }

    let mut stories = Vec::new();
    for handle in handles {
        match handle.await? {
            Ok(story) => {
                stories.push(story);
            }
            Err(err) => {
                eprintln!("{err}");
            }
        }
    }

    Ok(stories)
}
