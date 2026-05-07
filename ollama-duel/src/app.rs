use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::Backend;
use ratatui::widgets::ListState;
use ratatui::Terminal;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;

use crate::actor_critic::BgEvent;
use crate::config::{Config, McpServerConfig};
use crate::mcp;
use crate::ollama::OllamaMessage;
use crate::ui;

// ── Conversation model ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ConvRole {
    User,
    Actor { model: String, round: usize },
    ActorStreaming,
    Critic { model: String },
    CriticStreaming,
    ToolCall { server: String, tool: String },
    ToolResult { tool: String, is_error: bool },
    SystemNote,
}

#[derive(Debug, Clone)]
pub struct ConvItem {
    pub role: ConvRole,
    pub content: String,
    pub score: Option<u8>,
    pub should_revise: Option<bool>,
}

impl ConvItem {
    fn user(content: impl Into<String>) -> Self {
        Self { role: ConvRole::User, content: content.into(), score: None, should_revise: None }
    }
    fn system(content: impl Into<String>) -> Self {
        Self { role: ConvRole::SystemNote, content: content.into(), score: None, should_revise: None }
    }
}

// ── MCP server state ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ServerStatus {
    Pending,
    Connecting,
    Connected,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct McpServerState {
    pub name: String,
    pub status: ServerStatus,
    pub tools: Vec<String>,
}

// ── Input mode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum InputMode {
    Normal,
    Editing,
}

// ── Spinner frames ────────────────────────────────────────────────────────────

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── App ───────────────────────────────────────────────────────────────────────

pub struct App {
    // Config
    pub config: Arc<Config>,

    // UI state
    pub mode: InputMode,
    pub input: String,
    pub cursor_pos: usize,
    pub should_quit: bool,

    // Conversation
    pub conv: Vec<ConvItem>,
    pub conv_scroll: u16,

    // Streaming buffers (replaced when ActorDone/CriticDone arrives)
    pub actor_buf: String,
    pub critic_buf: String,

    // MCP
    pub mcp_servers: Vec<McpServerState>,
    pub mcp_clients: Arc<Mutex<HashMap<String, mcp::McpClient>>>,
    #[allow(dead_code)]
    pub mcp_list_state: ListState,

    // Processing
    pub is_processing: bool,
    pub current_round: usize,
    pub max_rounds: usize,
    pub spinner_tick: usize,
    pub last_tick: Instant,
    pub status: String,

    // Available Ollama models
    #[allow(dead_code)]
    pub available_models: Vec<String>,

    // Background event channel
    event_tx: UnboundedSender<BgEvent>,
    event_rx: UnboundedReceiver<BgEvent>,
}

impl App {
    pub fn new(config: Arc<Config>, mcp_clients: Arc<Mutex<HashMap<String, mcp::McpClient>>>) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        // Pre-populate MCP server states from config
        let mcp_servers: Vec<McpServerState> = config
            .mcp_servers
            .iter()
            .filter(|s| s.enabled)
            .map(|s| McpServerState {
                name: s.name.clone(),
                status: ServerStatus::Pending,
                tools: vec![],
            })
            .collect();

        let max_rounds = config.orchestration.max_revision_rounds;

        Self {
            config,
            mode: InputMode::Normal,
            input: String::new(),
            cursor_pos: 0,
            should_quit: false,
            conv: vec![ConvItem::system(
                "ollama-duel ready. Press [i] to enter a prompt.",
            )],
            conv_scroll: 0,
            actor_buf: String::new(),
            critic_buf: String::new(),
            mcp_servers,
            mcp_clients,
            mcp_list_state: ListState::default(),
            is_processing: false,
            current_round: 0,
            max_rounds,
            spinner_tick: 0,
            last_tick: Instant::now(),
            status: "Idle".into(),
            available_models: vec![],
            event_tx,
            event_rx,
        }
    }

    // ── MCP initialisation ────────────────────────────────────────────────────

    pub async fn init_mcp_servers(&mut self) {
        let configs: Vec<McpServerConfig> = self
            .config
            .mcp_servers
            .iter()
            .filter(|s| s.enabled)
            .cloned()
            .collect();

        for cfg in configs {
            let idx = self.mcp_servers.iter().position(|s| s.name == cfg.name);
            if let Some(i) = idx {
                self.mcp_servers[i].status = ServerStatus::Connecting;
            }

            let tx = self.event_tx.clone();
            let clients = self.mcp_clients.clone();

            tokio::spawn(async move {
                match mcp::McpClient::connect(&cfg).await {
                    Ok(client) => {
                        let tools: Vec<String> =
                            client.tools.iter().map(|t| t.name.clone()).collect();
                        let tool_count = tools.len();
                        let name = client.name.clone();
                        {
                            let mut map = clients.lock().await;
                            map.insert(name.clone(), client);
                        }
                        let _ = tx.send(BgEvent::McpConnected { server: name, tool_count });
                    }
                    Err(e) => {
                        let _ = tx.send(BgEvent::McpError {
                            server: cfg.name.clone(),
                            error: e.to_string(),
                        });
                    }
                }
            });
        }

        // Fetch available Ollama models in background
        let base_url = self.config.ollama.base_url.clone();
        let tx = self.event_tx.clone();
        tokio::spawn(async move {
            let ollama =
                crate::ollama::OllamaClient::new(&base_url, 10);
            match ollama.list_models().await {
                Ok(models) => {
                    let _ = tx.send(BgEvent::Status(format!(
                        "Ollama OK — {} models available",
                        models.len()
                    )));
                }
                Err(e) => {
                    let _ = tx.send(BgEvent::Error(format!("Cannot reach Ollama: {}", e)));
                }
            }
        });
    }

    // ── Main event loop ───────────────────────────────────────────────────────

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        use crossterm::event::EventStream;
        use futures::StreamExt;

        let mut crossterm_events = EventStream::new();

        loop {
            // Tick spinner if processing
            if self.last_tick.elapsed() >= std::time::Duration::from_millis(80) {
                self.spinner_tick = (self.spinner_tick + 1) % SPINNER.len();
                self.last_tick = Instant::now();
            }

            terminal.draw(|f| ui::render(f, self))?;

            tokio::select! {
                // Terminal input
                Some(Ok(event)) = crossterm_events.next() => {
                    match event {
                        Event::Key(key) => self.handle_key(key),
                        Event::Resize(_, _) => {} // ratatui handles redraw
                        _ => {}
                    }
                }

                // Background events from actor/critic/MCP
                Some(bg) = self.event_rx.recv() => {
                    self.handle_bg(bg);
                }

                // 60 fps minimum refresh
                _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {}
            }

            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    // ── Key handling ──────────────────────────────────────────────────────────

    fn handle_key(&mut self, key: KeyEvent) {
        // Ctrl-C / Ctrl-Q always quit
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') | KeyCode::Char('q') => {
                    self.should_quit = true;
                    return;
                }
                _ => {}
            }
        }

        match self.mode {
            InputMode::Editing => self.handle_key_editing(key),
            InputMode::Normal => self.handle_key_normal(key),
        }
    }

    fn handle_key_normal(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => self.should_quit = true,
            KeyCode::Char('i') | KeyCode::Enter => {
                if !self.is_processing {
                    self.mode = InputMode::Editing;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.conv_scroll = self.conv_scroll.saturating_sub(3);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.conv_scroll = self.conv_scroll.saturating_add(3);
            }
            KeyCode::PageUp => {
                self.conv_scroll = self.conv_scroll.saturating_sub(20);
            }
            KeyCode::PageDown => {
                self.conv_scroll = self.conv_scroll.saturating_add(20);
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.conv_scroll = u16::MAX; // will be clamped in render
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.conv_scroll = 0;
            }
            _ => {}
        }
    }

    fn handle_key_editing(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Enter => self.submit(),
            KeyCode::Esc => {
                self.mode = InputMode::Normal;
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
            }
            KeyCode::Backspace => {
                if self.cursor_pos > 0 {
                    self.cursor_pos -= 1;
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Delete => {
                if self.cursor_pos < self.input.len() {
                    self.input.remove(self.cursor_pos);
                }
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                if self.cursor_pos < self.input.len() {
                    self.cursor_pos += 1;
                }
            }
            KeyCode::Home => self.cursor_pos = 0,
            KeyCode::End => self.cursor_pos = self.input.len(),
            _ => {}
        }
    }

    fn submit(&mut self) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }

        self.input.clear();
        self.cursor_pos = 0;
        self.mode = InputMode::Normal;
        self.is_processing = true;
        self.current_round = 0;
        self.actor_buf.clear();
        self.critic_buf.clear();

        // Add user message to conversation
        self.conv.push(ConvItem::user(&prompt));
        self.conv_scroll = u16::MAX; // scroll to bottom

        // Build prior message history for Ollama (user/assistant only)
        let history: Vec<OllamaMessage> = self
            .conv
            .iter()
            .filter_map(|item| match &item.role {
                ConvRole::User => Some(OllamaMessage::user(&item.content)),
                ConvRole::Actor { .. } => Some(OllamaMessage::assistant(&item.content)),
                _ => None,
            })
            .collect();

        let tx = self.event_tx.clone();
        let config = self.config.clone();
        let clients = self.mcp_clients.clone();

        tokio::spawn(async move {
            if let Err(e) =
                crate::actor_critic::run(prompt, history, config, clients, tx.clone()).await
            {
                let _ = tx.send(BgEvent::Error(e.to_string()));
            }
        });
    }

    // ── Background event handling ─────────────────────────────────────────────

    fn handle_bg(&mut self, event: BgEvent) {
        match event {
            // ── MCP ──
            BgEvent::McpConnected { server, tool_count } => {
                if let Some(s) = self.mcp_servers.iter_mut().find(|s| s.name == server) {
                    let tools: Vec<String> = {
                        if let Ok(clients) = self.mcp_clients.try_lock() {
                            clients
                                .get(&server)
                                .map(|c| c.tools.iter().map(|t| t.name.clone()).collect())
                                .unwrap_or_default()
                        } else {
                            vec![]
                        }
                    };
                    s.status = ServerStatus::Connected;
                    s.tools = tools;
                }
                self.conv.push(ConvItem::system(format!(
                    "MCP '{}' connected — {} tools",
                    server, tool_count
                )));
            }
            BgEvent::McpError { server, error } => {
                if let Some(s) = self.mcp_servers.iter_mut().find(|s| s.name == server) {
                    s.status = ServerStatus::Error(error.clone());
                }
                self.conv.push(ConvItem::system(format!("MCP '{}' error: {}", server, error)));
            }
            BgEvent::McpDisconnected(server) => {
                if let Some(s) = self.mcp_servers.iter_mut().find(|s| s.name == server) {
                    s.status = ServerStatus::Error("disconnected".into());
                }
            }

            // ── Control ──
            BgEvent::RoundStart(n) => {
                self.current_round = n;
                self.status = format!("Round {}/{}", n, self.max_rounds + 1);
                self.actor_buf.clear();
                self.critic_buf.clear();
            }
            BgEvent::Status(msg) => {
                self.status = msg;
            }
            BgEvent::Error(e) => {
                self.is_processing = false;
                self.status = format!("Error: {}", e);
                self.conv.push(ConvItem {
                    role: ConvRole::SystemNote,
                    content: format!("Error: {}", e),
                    score: None,
                    should_revise: None,
                });
            }
            BgEvent::ProcessingDone { .. } => {
                self.is_processing = false;
                self.status = "Idle".into();
                self.conv_scroll = u16::MAX;
                // Remove any stale streaming entries
                self.conv.retain(|item| {
                    !matches!(item.role, ConvRole::ActorStreaming | ConvRole::CriticStreaming)
                });
            }

            // ── Actor ──
            BgEvent::ActorStart { model, round } => {
                self.status = format!("Actor {} — round {}", model, round);
                self.actor_buf.clear();
                // Add a streaming placeholder
                self.conv.push(ConvItem {
                    role: ConvRole::ActorStreaming,
                    content: String::new(),
                    score: None,
                    should_revise: None,
                });
                self.conv_scroll = u16::MAX;
            }
            BgEvent::ActorChunk(chunk) => {
                self.actor_buf.push_str(&chunk);
                // Update the streaming placeholder
                if let Some(item) =
                    self.conv.iter_mut().rev().find(|i| matches!(i.role, ConvRole::ActorStreaming))
                {
                    item.content = self.actor_buf.clone();
                }
                self.conv_scroll = u16::MAX;
            }
            BgEvent::ActorDone { content } => {
                // Replace streaming placeholder with final
                if let Some(item) =
                    self.conv.iter_mut().rev().find(|i| matches!(i.role, ConvRole::ActorStreaming))
                {
                    item.role = ConvRole::Actor {
                        model: self.config.actor.model.clone(),
                        round: self.current_round,
                    };
                    item.content = content;
                }
                self.actor_buf.clear();
            }
            BgEvent::ActorToolCall { server, tool, args } => {
                self.conv.push(ConvItem {
                    role: ConvRole::ToolCall { server, tool: tool.clone() },
                    content: serde_json::to_string_pretty(&args).unwrap_or_default(),
                    score: None,
                    should_revise: None,
                });
                self.conv_scroll = u16::MAX;
            }
            BgEvent::ActorToolResult { tool, result, is_error } => {
                self.conv.push(ConvItem {
                    role: ConvRole::ToolResult { tool, is_error },
                    content: truncate(&result, 800),
                    score: None,
                    should_revise: None,
                });
                self.conv_scroll = u16::MAX;
            }

            // ── Critic ──
            BgEvent::CriticStart { model } => {
                self.status = format!("Critic {} evaluating…", model);
                self.critic_buf.clear();
                self.conv.push(ConvItem {
                    role: ConvRole::CriticStreaming,
                    content: String::new(),
                    score: None,
                    should_revise: None,
                });
                self.conv_scroll = u16::MAX;
            }
            BgEvent::CriticChunk(chunk) => {
                self.critic_buf.push_str(&chunk);
                if let Some(item) = self
                    .conv
                    .iter_mut()
                    .rev()
                    .find(|i| matches!(i.role, ConvRole::CriticStreaming))
                {
                    item.content = self.critic_buf.clone();
                }
                self.conv_scroll = u16::MAX;
            }
            BgEvent::CriticDone { score, feedback, should_revise } => {
                // Replace streaming placeholder
                if let Some(item) = self
                    .conv
                    .iter_mut()
                    .rev()
                    .find(|i| matches!(i.role, ConvRole::CriticStreaming))
                {
                    item.role = ConvRole::Critic { model: self.config.critic.model.clone() };
                    item.content = feedback;
                    item.score = Some(score);
                    item.should_revise = Some(should_revise);
                }
                self.critic_buf.clear();
                self.conv_scroll = u16::MAX;
            }
        }
    }

    // ── Public helpers used by the UI ──────────────────────────────────────────

    pub fn spinner(&self) -> &str {
        if self.is_processing {
            SPINNER[self.spinner_tick % SPINNER.len()]
        } else {
            " "
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…[{} chars]", &s[..max], s.len())
    }
}
