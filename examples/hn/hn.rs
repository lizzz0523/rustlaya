//! HackerNews API 抓取：取新帖 id 列表，并受限并发拉取每篇 story 详情。

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

/// 抓取 HackerNews 最新 story 列表，单篇失败只记录不中断（并发上限 10）。
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

    let mut handles = Vec::new();
    for id in ids.into_iter() {
        let semaphore = semaphore.clone();
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            let story: Story = client
                .get(format!(
                    "https://hacker-news.firebaseio.com/v0/item/{id}/.json"
                ))
                .send()
                .await?
                .json()
                .await?;
            Ok::<_, reqwest::Error>(story)
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
