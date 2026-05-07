use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc::UnboundedSender, Mutex};

use crate::config::Config;
use crate::mcp::{client as mcp_client, McpTool};
use crate::ollama::{OllamaClient, OllamaMessage};

// ── Background event bus ─────────────────────────────────────────────────────

/// Events produced by the actor/critic background task and consumed by the UI.
#[derive(Debug, Clone)]
pub enum BgEvent {
    // ── Actor ──
    ActorStart { model: String, round: usize },
    ActorChunk(String),
    ActorToolCall { server: String, tool: String, args: Value },
    ActorToolResult { tool: String, result: String, is_error: bool },
    ActorDone { content: String },

    // ── Critic ──
    CriticStart { model: String },
    CriticChunk(String),
    CriticDone { score: u8, feedback: String, should_revise: bool },

    // ── Control ──
    RoundStart(usize),
    #[allow(dead_code)]
    ProcessingDone { final_response: String },
    Error(String),
    Status(String),

    // ── MCP lifecycle ──
    McpConnected { server: String, tool_count: usize },
    #[allow(dead_code)]
    McpDisconnected(String),
    McpError { server: String, error: String },
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Run the full actor→critic loop for a single user prompt.
/// All progress is broadcast via `tx`; the final event is `ProcessingDone`.
pub async fn run(
    prompt: String,
    prior_messages: Vec<OllamaMessage>,
    config: Arc<Config>,
    mcp_clients: Arc<Mutex<HashMap<String, crate::mcp::McpClient>>>,
    tx: UnboundedSender<BgEvent>,
) -> Result<()> {
    let ollama = OllamaClient::new(&config.ollama.base_url, config.ollama.timeout_secs);

    // Collect MCP tools from all connected servers
    let tools: Vec<McpTool> = {
        let clients = mcp_clients.lock().await;
        mcp_client::all_tools(&clients)
    };
    let ollama_tools: Vec<_> = tools.iter().map(|t| t.to_ollama_tool()).collect();

    // Build initial message list
    let mut messages: Vec<OllamaMessage> = {
        let mut m = vec![OllamaMessage::system(&config.actor.system_prompt)];
        m.extend(prior_messages);
        m.push(OllamaMessage::user(&prompt));
        m
    };

    let mut final_response = String::new();

    // ── Revision loop ──────────────────────────────────────────────────────
    for round in 0..=config.orchestration.max_revision_rounds {
        let _ = tx.send(BgEvent::RoundStart(round + 1));

        // ── Actor phase ────────────────────────────────────────────────────
        let _ = tx.send(BgEvent::ActorStart {
            model: config.actor.model.clone(),
            round: round + 1,
        });

        let actor_content = run_actor(
            &prompt,
            &mut messages,
            &config,
            &ollama_tools,
            &mcp_clients,
            &ollama,
            &tx,
        )
        .await?;

        final_response = actor_content.clone();

        // ── Critic phase ───────────────────────────────────────────────────
        let _ = tx.send(BgEvent::CriticStart {
            model: config.critic.model.clone(),
        });

        let critic = run_critic(&prompt, &actor_content, &config, &ollama, &tx).await?;

        let _ = tx.send(BgEvent::CriticDone {
            score: critic.score,
            feedback: critic.feedback.clone(),
            should_revise: critic.should_revise,
        });

        // Decide whether to continue
        let will_revise = critic.should_revise
            && critic.score < config.orchestration.critique_threshold
            && round < config.orchestration.max_revision_rounds;

        if will_revise {
            // Feed critic feedback back to the actor as a system note
            messages.push(OllamaMessage {
                role: "user".to_string(),
                content: format!(
                    "[Revision request — Critic score {}/10]\n{}",
                    critic.score, critic.feedback
                ),
                tool_calls: None,
                tool_call_id: None,
            });
        } else {
            break;
        }
    }

    let _ = tx.send(BgEvent::ProcessingDone { final_response });
    Ok(())
}

// ── Actor ─────────────────────────────────────────────────────────────────────

async fn run_actor(
    _original_prompt: &str,
    messages: &mut Vec<OllamaMessage>,
    config: &Config,
    ollama_tools: &[crate::mcp::OllamaTool],
    mcp_clients: &Arc<Mutex<HashMap<String, crate::mcp::McpClient>>>,
    ollama: &OllamaClient,
    tx: &UnboundedSender<BgEvent>,
) -> Result<String> {
    let mut tool_loops = 0;

    loop {
        let result = ollama
            .chat(
                &config.actor.model,
                messages,
                ollama_tools,
                config.actor.temperature,
                false,
                tx,
            )
            .await?;

        // No tool calls → final text response
        if result.tool_calls.is_empty() || tool_loops >= config.actor.max_tool_rounds {
            let content = result.content.trim().to_string();
            let _ = tx.send(BgEvent::ActorDone { content: content.clone() });
            messages.push(OllamaMessage::assistant(&content));
            return Ok(content);
        }

        // Execute each tool call via MCP
        let mut tool_results: Vec<OllamaMessage> = vec![];
        tool_loops += 1;

        // Append the assistant message that contains the tool calls
        messages.push(OllamaMessage::assistant_tool_call(result.tool_calls.clone()));

        for tc in &result.tool_calls {
            let tool_name = &tc.function.name;
            let args = &tc.function.arguments;

            // Find which server owns this tool
            let server_name = {
                let clients = mcp_clients.lock().await;
                mcp_client::server_for_tool(&clients, tool_name)
                    .map(|s| s.to_string())
            };

            let server_name = match server_name {
                Some(s) => s,
                None => {
                    let msg = format!("No MCP server found for tool '{}'", tool_name);
                    tracing::warn!("{}", msg);
                    let _ = tx.send(BgEvent::ActorToolResult {
                        tool: tool_name.clone(),
                        result: msg.clone(),
                        is_error: true,
                    });
                    tool_results.push(OllamaMessage::tool_result(tool_name, msg));
                    continue;
                }
            };

            let _ = tx.send(BgEvent::ActorToolCall {
                server: server_name.clone(),
                tool: tool_name.clone(),
                args: args.clone(),
            });

            let (tool_output, is_error) = {
                let mut clients = mcp_clients.lock().await;
                match clients.get_mut(&server_name) {
                    Some(client) => client.call_tool(tool_name, args).await.unwrap_or_else(|e| {
                        (format!("Tool error: {}", e), true)
                    }),
                    None => (format!("Server '{}' not connected", server_name), true),
                }
            };

            let _ = tx.send(BgEvent::ActorToolResult {
                tool: tool_name.clone(),
                result: tool_output.clone(),
                is_error,
            });

            tool_results.push(OllamaMessage::tool_result(tool_name, tool_output));
        }

        messages.extend(tool_results);
    }
}

// ── Critic ────────────────────────────────────────────────────────────────────

struct CriticVerdict {
    score: u8,
    feedback: String,
    should_revise: bool,
}

async fn run_critic(
    user_prompt: &str,
    actor_response: &str,
    config: &Config,
    ollama: &OllamaClient,
    tx: &UnboundedSender<BgEvent>,
) -> Result<CriticVerdict> {
    let critic_user_msg = format!(
        "User request:\n{}\n\nAssistant response to evaluate:\n{}",
        user_prompt, actor_response
    );

    let messages = vec![
        OllamaMessage::system(&config.critic.system_prompt),
        OllamaMessage::user(critic_user_msg),
    ];

    let result = ollama
        .chat(
            &config.critic.model,
            &messages,
            &[], // critic never calls tools
            config.critic.temperature,
            true,
            tx,
        )
        .await?;

    Ok(parse_critic_response(&result.content))
}

/// Robustly parse the critic's JSON response, with graceful fallbacks.
fn parse_critic_response(text: &str) -> CriticVerdict {
    // Strip common markdown fences
    let cleaned = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Try strict JSON parse
    if let Ok(v) = serde_json::from_str::<Value>(cleaned) {
        let score = v["score"].as_u64().unwrap_or(5).min(10) as u8;
        let feedback = v["feedback"].as_str().unwrap_or("No feedback.").to_string();
        let should_revise = v["should_revise"]
            .as_bool()
            .unwrap_or(score < 7);
        return CriticVerdict { score, feedback, should_revise };
    }

    // Fallback: scan for a JSON object anywhere in the text
    if let Some(start) = cleaned.find('{') {
        if let Some(end) = cleaned.rfind('}') {
            if let Ok(v) = serde_json::from_str::<Value>(&cleaned[start..=end]) {
                let score = v["score"].as_u64().unwrap_or(5).min(10) as u8;
                let feedback = v["feedback"].as_str().unwrap_or(cleaned).to_string();
                let should_revise = v["should_revise"].as_bool().unwrap_or(score < 7);
                return CriticVerdict { score, feedback, should_revise };
            }
        }
    }

    // Last resort: heuristic score extraction
    let score = extract_score_heuristic(cleaned).unwrap_or(5);
    CriticVerdict {
        score,
        feedback: cleaned.chars().take(300).collect(),
        should_revise: score < 7,
    }
}

fn extract_score_heuristic(text: &str) -> Option<u8> {
    // Look for patterns: "score: 8", "8/10", "Score:7"
    for part in text.split(|c: char| !c.is_ascii_digit() && c != '/') {
        if let Some((left, right)) = part.split_once('/') {
            if right.trim() == "10" || right.trim() == "10." {
                if let Ok(n) = left.trim().parse::<u8>() {
                    if n <= 10 { return Some(n); }
                }
            }
        }
        if let Ok(n) = part.trim().parse::<u8>() {
            if n <= 10 { return Some(n); }
        }
    }
    None
}
