pub mod client;

pub use client::McpClient;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── JSON-RPC 2.0 ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

// ── MCP Tool (as returned by tools/list) ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
    /// Which server this tool belongs to (filled in by our client, not the protocol).
    #[serde(skip)]
    pub server_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolsListResult {
    pub tools: Vec<McpTool>,
    #[serde(rename = "nextCursor", default)]
    pub next_cursor: Option<String>,
}

// ── Ollama tool format (for the /api/chat request) ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OllamaFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl McpTool {
    pub fn to_ollama_tool(&self) -> OllamaTool {
        OllamaTool {
            tool_type: "function".to_string(),
            function: OllamaFunction {
                name: self.name.clone(),
                description: self.description.clone().unwrap_or_default(),
                parameters: self.input_schema.clone(),
            },
        }
    }
}
