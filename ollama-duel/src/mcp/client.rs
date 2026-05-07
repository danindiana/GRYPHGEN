use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};

use crate::config::McpServerConfig;
use super::{
    JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpTool,
    McpToolsListResult,
};

// ── McpClient ─────────────────────────────────────────────────────────────────

/// A client for a single MCP stdio server.
pub struct McpClient {
    /// Keep the child alive for its lifetime.
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    pub name: String,
    pub tools: Vec<McpTool>,
    pub server_info: Option<ServerInfo>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

impl McpClient {
    /// Spawn the MCP server process and perform the MCP initialisation handshake.
    pub async fn connect(cfg: &McpServerConfig) -> Result<Self> {
        tracing::info!("Spawning MCP server '{}': {} {:?}", cfg.name, cfg.command, cfg.args);

        let mut cmd = tokio::process::Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());

        // Merge environment variables, resolving ${VAR} patterns
        for (k, v) in &cfg.env {
            let resolved = resolve_env(v);
            cmd.env(k, resolved);
        }

        let mut child = cmd.spawn().with_context(|| {
            format!("Failed to spawn MCP server '{}' (command: {})", cfg.name, cfg.command)
        })?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("No stdin"))?;
        let stdout_raw = child.stdout.take().ok_or_else(|| anyhow!("No stdout"))?;
        let stdout = BufReader::new(stdout_raw);

        let mut client = McpClient {
            _child: child,
            stdin,
            stdout,
            next_id: 1,
            name: cfg.name.clone(),
            tools: vec![],
            server_info: None,
        };

        client.initialize().await?;
        client.load_tools().await?;

        tracing::info!(
            "MCP server '{}' ready with {} tools",
            cfg.name,
            client.tools.len()
        );
        Ok(client)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    async fn send_notification(&mut self, method: &str, params: Option<Value>) -> Result<()> {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        };
        let line = serde_json::to_string(&notif)? + "\n";
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn send_request(&mut self, method: &str, params: Option<Value>) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };
        let line = serde_json::to_string(&req)? + "\n";
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;

        // Read response lines until we get one matching our id
        loop {
            let mut raw = String::new();
            let n = self.stdout.read_line(&mut raw).await?;
            if n == 0 {
                return Err(anyhow!("MCP server '{}' closed stdout", self.name));
            }
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }

            let resp: JsonRpcResponse = serde_json::from_str(raw)
                .with_context(|| format!("Bad JSON from '{}': {}", self.name, raw))?;

            // Match by id (id can be number or string in JSON-RPC)
            let matches = match &resp.id {
                Some(Value::Number(n)) => n.as_u64() == Some(id),
                Some(Value::String(s)) => s.parse::<u64>().ok() == Some(id),
                _ => false,
            };

            if matches {
                if let Some(err) = resp.error {
                    return Err(anyhow!("MCP error {} from '{}': {}", err.code, self.name, err.message));
                }
                return Ok(resp.result.unwrap_or(Value::Null));
            }
            // Drop unmatched messages (e.g. server-sent notifications)
            tracing::debug!("Dropping unmatched message from '{}': {}", self.name, raw);
        }
    }

    async fn initialize(&mut self) -> Result<()> {
        let result = self
            .send_request(
                "initialize",
                Some(json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "roots": { "listChanged": false } },
                    "clientInfo": { "name": "ollama-duel", "version": env!("CARGO_PKG_VERSION") }
                })),
            )
            .await?;

        // Parse server info
        if let (Some(name), Some(version)) = (
            result["serverInfo"]["name"].as_str(),
            result["serverInfo"]["version"].as_str(),
        ) {
            self.server_info = Some(ServerInfo {
                name: name.to_string(),
                version: version.to_string(),
            });
        }

        // Send initialized notification (required by MCP spec)
        self.send_notification("notifications/initialized", None).await?;
        Ok(())
    }

    async fn load_tools(&mut self) -> Result<()> {
        let result = self.send_request("tools/list", Some(json!({}))).await?;
        let list: McpToolsListResult =
            serde_json::from_value(result).unwrap_or(McpToolsListResult {
                tools: vec![],
                next_cursor: None,
            });

        self.tools = list
            .tools
            .into_iter()
            .map(|mut t| {
                t.server_name = self.name.clone();
                t
            })
            .collect();

        // TODO: handle pagination via next_cursor
        Ok(())
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Call a tool and return (text_content, is_error).
    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<(String, bool)> {
        let result = self
            .send_request(
                "tools/call",
                Some(json!({ "name": tool_name, "arguments": arguments })),
            )
            .await?;

        let is_error = result["isError"].as_bool().unwrap_or(false);
        let content = extract_content_text(&result);
        Ok((content, is_error))
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Collect all text/content blocks from a tool result into a single string.
fn extract_content_text(result: &Value) -> String {
    if let Some(arr) = result["content"].as_array() {
        arr.iter()
            .map(|item| match item["type"].as_str() {
                Some("text") => item["text"].as_str().unwrap_or("").to_string(),
                Some("resource") => {
                    let uri = item["resource"]["uri"].as_str().unwrap_or("?");
                    let text = item["resource"]["text"].as_str().unwrap_or("");
                    format!("[resource: {}]\n{}", uri, text)
                }
                _ => serde_json::to_string(item).unwrap_or_default(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else if let Some(s) = result.as_str() {
        s.to_string()
    } else {
        serde_json::to_string_pretty(result).unwrap_or_default()
    }
}

/// Resolve ${VAR} patterns in strings from the environment.
fn resolve_env(s: &str) -> String {
    let mut out = s.to_string();
    for (k, v) in std::env::vars() {
        out = out.replace(&format!("${{{}}}", k), &v);
        out = out.replace(&format!("${}", k), &v);
    }
    out
}

// ── Pool helper ───────────────────────────────────────────────────────────────

/// Collect all tools from all connected MCP clients.
pub fn all_tools(clients: &HashMap<String, McpClient>) -> Vec<super::McpTool> {
    clients.values().flat_map(|c| c.tools.clone()).collect()
}

/// Find which client owns a tool by name.
pub fn server_for_tool<'a>(
    clients: &'a HashMap<String, McpClient>,
    tool_name: &str,
) -> Option<&'a str> {
    clients
        .values()
        .find(|c| c.tools.iter().any(|t| t.name == tool_name))
        .map(|c| c.name.as_str())
}
