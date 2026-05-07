use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;

use crate::actor_critic::BgEvent;
use crate::mcp::OllamaTool;

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMsg>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallMsg {
    #[serde(rename = "function")]
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: Value,
}

/// Decoded chunk from the streaming /api/chat endpoint.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    message: Option<ChunkMessage>,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallMsg>>,
}

/// Result of a single chat invocation.
#[derive(Debug, Default)]
pub struct ChatResult {
    pub content: String,
    pub tool_calls: Vec<ToolCallMsg>,
}

// ── Constructors for OllamaMessage ────────────────────────────────────────────

impl OllamaMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into(), tool_calls: None, tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), tool_calls: None, tool_call_id: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into(), tool_calls: None, tool_call_id: None }
    }
    pub fn assistant_tool_call(calls: Vec<ToolCallMsg>) -> Self {
        Self { role: "assistant".into(), content: String::new(), tool_calls: Some(calls), tool_call_id: None }
    }
    /// MCP tool result returned back to the model.
    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_name.into()),
        }
    }
}

// ── OllamaClient ─────────────────────────────────────────────────────────────

pub struct OllamaClient {
    pub base_url: String,
    client: Client,
}

impl OllamaClient {
    pub fn new(base_url: impl Into<String>, timeout_secs: u64) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        Self { base_url: base_url.into(), client }
    }

    /// Returns available model names.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let body: Value = resp.json().await?;
        let models = body["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    /// Run a streaming chat completion.
    ///
    /// * Sends actor/critic text chunks to `tx` as `BgEvent::ActorChunk` /
    ///   `BgEvent::CriticChunk` depending on `is_critic`.
    /// * Returns the complete `ChatResult` (accumulated text + any tool calls).
    pub async fn chat(
        &self,
        model: &str,
        messages: &[OllamaMessage],
        tools: &[OllamaTool],
        temperature: f32,
        is_critic: bool,
        tx: &UnboundedSender<BgEvent>,
    ) -> Result<ChatResult> {
        let url = format!("{}/api/chat", self.base_url);

        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "options": { "temperature": temperature, "num_predict": 2048 }
        });

        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools)?;
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Ollama request failed: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("Ollama {}: {}", status, text));
        }

        let mut result = ChatResult::default();
        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(|e| anyhow!("Stream error: {}", e))?;
            buf.push_str(&String::from_utf8_lossy(&bytes));

            // Process all complete newline-delimited JSON objects in the buffer
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf.drain(..=pos);

                if line.is_empty() {
                    continue;
                }

                let chunk: StreamChunk = match serde_json::from_str(&line) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("Failed to parse stream chunk: {} | line: {}", e, line);
                        continue;
                    }
                };

                if let Some(msg) = chunk.message {
                    // Text delta
                    if !msg.content.is_empty() {
                        result.content.push_str(&msg.content);
                        let ev = if is_critic {
                            BgEvent::CriticChunk(msg.content.clone())
                        } else {
                            BgEvent::ActorChunk(msg.content.clone())
                        };
                        let _ = tx.send(ev);
                    }
                    // Tool calls (usually arrive in the `done=true` chunk)
                    if let Some(calls) = msg.tool_calls {
                        result.tool_calls.extend(calls);
                    }
                }

                if chunk.done {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Convenience: check connectivity by hitting /api/tags.
    #[allow(dead_code)]
    pub async fn ping(&self) -> bool {
        self.list_models().await.is_ok()
    }
}
