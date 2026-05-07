use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Top-level ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub actor: ActorConfig,
    #[serde(default)]
    pub critic: CriticConfig,
    #[serde(default)]
    pub orchestration: OrchestrationConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

// ── Actor ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ActorConfig {
    #[serde(default = "default_actor_model")]
    pub model: String,
    #[serde(default = "default_actor_system")]
    pub system_prompt: String,
    /// Max tool-call loops before the actor must produce a text response.
    #[serde(default = "default_max_tool_rounds")]
    pub max_tool_rounds: usize,
    #[serde(default = "default_actor_temp")]
    pub temperature: f32,
}

// ── Critic ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CriticConfig {
    #[serde(default = "default_critic_model")]
    pub model: String,
    #[serde(default = "default_critic_system")]
    pub system_prompt: String,
    #[serde(default = "default_critic_temp")]
    pub temperature: f32,
}

// ── Orchestration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrchestrationConfig {
    /// How many actor→critic revision cycles to allow.
    #[serde(default = "default_max_revisions")]
    pub max_revision_rounds: usize,
    /// If critic score is below this, request a revision (when rounds remain).
    #[serde(default = "default_critique_threshold")]
    pub critique_threshold: u8,
}

// ── Ollama ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_url")]
    pub base_url: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

// ── MCP Server ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

// ── Default value functions ───────────────────────────────────────────────────

fn bool_true() -> bool { true }
fn default_actor_model() -> String { "qwen2.5:3b".to_string() }
fn default_critic_model() -> String { "qwen2.5:3b".to_string() }
fn default_actor_temp() -> f32 { 0.7 }
fn default_critic_temp() -> f32 { 0.15 }
fn default_max_tool_rounds() -> usize { 6 }
fn default_max_revisions() -> usize { 2 }
fn default_critique_threshold() -> u8 { 7 }
fn default_ollama_url() -> String { "http://localhost:11434".to_string() }
fn default_timeout() -> u64 { 180 }

fn default_actor_system() -> String {
    "You are a capable AI assistant with access to tools via MCP (Model Context Protocol). \
     Use available tools when they help you give a more accurate or complete answer. \
     Think step-by-step. Be concise but thorough.".to_string()
}

fn default_critic_system() -> String {
    r#"You are a rigorous AI quality critic. Given a user request and an assistant response, evaluate the response quality.

Respond ONLY with valid compact JSON on a single line (no markdown, no explanation):
{"score":<0-10>,"feedback":"<concise feedback>","should_revise":<true|false>}

Scoring rubric:
  9-10 : Excellent – accurate, complete, well-structured. should_revise=false
  7-8  : Good – minor gaps or style issues. should_revise=false
  5-6  : Adequate – missing important aspects. should_revise=true
  0-4  : Poor – inaccurate, off-topic, or dangerously incomplete. should_revise=true

Only set should_revise=true when a revision would meaningfully improve the answer."#.to_string()
}

// ── Default impls ─────────────────────────────────────────────────────────────

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            model: default_actor_model(),
            system_prompt: default_actor_system(),
            max_tool_rounds: default_max_tool_rounds(),
            temperature: default_actor_temp(),
        }
    }
}

impl Default for CriticConfig {
    fn default() -> Self {
        Self {
            model: default_critic_model(),
            system_prompt: default_critic_system(),
            temperature: default_critic_temp(),
        }
    }
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            max_revision_rounds: default_max_revisions(),
            critique_threshold: default_critique_threshold(),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_url(),
            timeout_secs: default_timeout(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            actor: ActorConfig::default(),
            critic: CriticConfig::default(),
            orchestration: OrchestrationConfig::default(),
            ollama: OllamaConfig::default(),
            mcp_servers: vec![],
        }
    }
}

// ── Loaders ───────────────────────────────────────────────────────────────────

/// Load config from default locations:
///   1. ~/.config/ollama-duel/config.toml
///   2. ./config.toml
///   3. built-in defaults
pub fn load_config() -> Result<Config> {
    if let Some(base) = dirs::config_dir() {
        let p = base.join("ollama-duel").join("config.toml");
        if p.exists() {
            tracing::info!("Loading config from {}", p.display());
            return load_config_from(&p.to_string_lossy());
        }
    }
    if std::path::Path::new("config.toml").exists() {
        tracing::info!("Loading config from ./config.toml");
        return load_config_from("config.toml");
    }
    tracing::info!("Using built-in defaults");
    Ok(Config::default())
}

pub fn load_config_from(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read config {}: {}", path, e))?;
    toml::from_str(&content).map_err(|e| anyhow::anyhow!("Config parse error: {}", e))
}
