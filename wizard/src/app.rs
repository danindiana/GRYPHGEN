use std::sync::mpsc;
use ratatui::widgets::ListState;
use tui_input::Input;
use crate::config::{Config, load as load_config, save as save_config};
use crate::api;

#[derive(Debug, Clone, PartialEq)]
pub enum DemoItem {
    FastGen,
    StandardGen,
    StrongGen,
    Agent,
    Search,
}

impl DemoItem {
    pub fn title(&self) -> &str {
        match self {
            DemoItem::FastGen => "FAST — Generate code (single function)",
            DemoItem::StandardGen => "STANDARD — Generate a module",
            DemoItem::StrongGen => "STRONG — Complex / multi-step code",
            DemoItem::Agent => "AGENT — Multi-step agentic task",
            DemoItem::Search => "SEARCH — Web search via DuckDuckGo",
        }
    }

    pub fn explain_lines(&self) -> Vec<&str> {
        match self {
            DemoItem::FastGen => vec![
                "FAST tier uses the CODER model directly.",
                "",
                "  Model:   qwen2.5-coder:7b (RTX 3080)",
                "  Target:  < 8s generation time",
                "  Best for: single functions, simple utilities,",
                "           one-liners, basic algorithms",
                "",
                "No planning step — just code.",
            ],
            DemoItem::StandardGen => vec![
                "STANDARD tier: THINKER plans, CODER executes.",
                "",
                "  Models:  deepseek-r1:14b → qwen2.5-coder:7b",
                "  Target:  ~15–20s warm",
                "  Best for: REST endpoints, classes, modules,",
                "           test suites, data structures",
                "",
                "THINKER produces a structured brief (GOAL /",
                "FUNCTIONS / PATTERNS / EDGE_CASES), then CODER",
                "implements it with that context.",
            ],
            DemoItem::StrongGen => vec![
                "STRONG tier: full Think → Code → Review loop.",
                "",
                "  Models:  deepseek-r1:14b ↔ qwen2.5-coder:7b",
                "  Target:  ~30–90s",
                "  Best for: auth systems, security code,",
                "           anything requiring careful review",
                "",
                "THINKER analyses → CODER implements →",
                "THINKER reviews → CODER revises if needed.",
                "Up to 2 revision cycles.",
            ],
            DemoItem::Agent => vec![
                "AGENT mode runs a ReAct (Reason + Act) loop.",
                "",
                "  The agent plans multi-step tasks and uses",
                "  tools (file read/write, code execution) to",
                "  work through them autonomously.",
                "",
                "  Target:  10–60s depending on steps",
                "  Best for: 'write X then do Y then show Z'",
                "",
                "You'll see each step in the output trace.",
            ],
            DemoItem::Search => vec![
                "SEARCH queries DuckDuckGo Instant Answers.",
                "",
                "  No API key required for DDG.",
                "  Returns: title, URL, snippet per result.",
                "",
                "  Best for: quick context lookups before",
                "  generating code that uses an unfamiliar lib.",
                "",
                "Results appear as a scrollable list.",
            ],
        }
    }

    pub fn default_prompt(&self) -> &str {
        match self {
            DemoItem::FastGen => "add two numbers",
            DemoItem::StandardGen => "write a Redis rate limiter for FastAPI",
            DemoItem::StrongGen => "write a JWT auth module with refresh tokens",
            DemoItem::Agent => "create hello.py that prints Hello World, then show its contents",
            DemoItem::Search => "Redis",
        }
    }

    pub fn loading_message(&self) -> &str {
        match self {
            DemoItem::FastGen => "Generating (FAST — CODER only)…",
            DemoItem::StandardGen => "Generating (STANDARD — THINKER plan → CODER)…",
            DemoItem::StrongGen => "Generating (STRONG — Think → Code → Review)…",
            DemoItem::Agent => "Running agent task…",
            DemoItem::Search => "Searching DuckDuckGo…",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResultMeta {
    pub tier: Option<String>,
    pub gen_time: Option<f64>,
    pub tokens: Option<u32>,
    pub steps: Option<u32>,
    pub model: Option<String>,
}

#[derive(Debug)]
pub enum Screen {
    Welcome,
    ConfigCheck,
    SetApiKey,
    SetApiUrl,
    HealthCheck,
    MainMenu,
    DemoExplain(DemoItem),
    DemoPrompt(DemoItem),
    DemoLoading(DemoItem),
    DemoResult { item: DemoItem, output: String, meta: ResultMeta },
    ShowConfig,
    ErrorOverlay { message: String, underlying: Box<Screen> },
}

pub enum ApiResult {
    Health(Result<String, String>),
    Gen(DemoItem, Result<(String, ResultMeta), String>),
    Agent(DemoItem, Result<(String, ResultMeta), String>),
    Search(DemoItem, Result<(String, ResultMeta), String>),
}

pub struct App {
    pub screen: Screen,
    pub config: Config,
    pub list_state: ListState,
    pub input: Input,
    pub scroll_offset: u16,
    pub tick: u64,
    pub tx: mpsc::SyncSender<ApiResult>,
    pub rx: mpsc::Receiver<ApiResult>,
    pub should_quit: bool,
}

pub const MENU_ITEMS: &[&str] = &[
    "1  Generate code         (FAST tier)",
    "2  Generate a module     (STANDARD tier)",
    "3  Complex generation    (STRONG tier)",
    "4  Run an agent task",
    "5  Search the web",
    "6  View configuration",
    "7  Exit",
];

impl App {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel(4);
        let config = load_config();
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        App {
            screen: Screen::Welcome,
            config,
            list_state,
            input: Input::default(),
            scroll_offset: 0,
            tick: 0,
            tx,
            rx,
            should_quit: false,
        }
    }

    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
        // Poll for API results
        if let Ok(result) = self.rx.try_recv() {
            self.handle_api_result(result);
        }
    }

    fn handle_api_result(&mut self, result: ApiResult) {
        match result {
            ApiResult::Health(Ok(_)) => {
                self.screen = Screen::MainMenu;
            }
            ApiResult::Health(Err(e)) => {
                let underlying = Box::new(Screen::MainMenu);
                self.screen = Screen::ErrorOverlay {
                    message: format!("Health check failed:\n{}", e),
                    underlying,
                };
                // Still go to main menu so user can try demos
                self.screen = Screen::MainMenu;
            }
            ApiResult::Gen(item, Ok((output, meta)))
            | ApiResult::Agent(item, Ok((output, meta)))
            | ApiResult::Search(item, Ok((output, meta))) => {
                self.scroll_offset = 0;
                self.screen = Screen::DemoResult { item, output, meta };
            }
            ApiResult::Gen(item, Err(e))
            | ApiResult::Agent(item, Err(e))
            | ApiResult::Search(item, Err(e)) => {
                self.screen = Screen::ErrorOverlay {
                    message: format!("Request failed:\n{}", e),
                    underlying: Box::new(Screen::DemoExplain(item)),
                };
            }
        }
    }

    pub fn on_enter(&mut self) {
        match &self.screen {
            Screen::Welcome => {
                if self.config.is_configured() {
                    self.start_health_check();
                } else {
                    self.input = Input::default();
                    self.screen = Screen::SetApiKey;
                }
            }
            Screen::SetApiKey => {
                let key = self.input.value().trim().to_string();
                if !key.is_empty() {
                    self.config.api_key = key;
                }
                self.input = Input::default();
                self.screen = Screen::SetApiUrl;
            }
            Screen::SetApiUrl => {
                let url = self.input.value().trim().to_string();
                if !url.is_empty() {
                    self.config.api_url = url;
                }
                let _ = save_config(&self.config);
                self.start_health_check();
            }
            Screen::HealthCheck => {}
            Screen::MainMenu => {
                let selected = self.list_state.selected().unwrap_or(0);
                match selected {
                    0 => self.screen = Screen::DemoExplain(DemoItem::FastGen),
                    1 => self.screen = Screen::DemoExplain(DemoItem::StandardGen),
                    2 => self.screen = Screen::DemoExplain(DemoItem::StrongGen),
                    3 => self.screen = Screen::DemoExplain(DemoItem::Agent),
                    4 => self.screen = Screen::DemoExplain(DemoItem::Search),
                    5 => self.screen = Screen::ShowConfig,
                    6 => self.should_quit = true,
                    _ => {}
                }
            }
            Screen::DemoExplain(item) => {
                let item = item.clone();
                let default = item.default_prompt().to_string();
                self.input = Input::new(default);
                self.screen = Screen::DemoPrompt(item);
            }
            Screen::DemoPrompt(item) => {
                let item = item.clone();
                let prompt = self.input.value().trim().to_string();
                self.launch_demo(item, prompt);
            }
            Screen::DemoResult { .. } => {
                self.list_state.select(Some(0));
                self.screen = Screen::MainMenu;
            }
            Screen::ShowConfig => {
                self.screen = Screen::MainMenu;
            }
            Screen::ErrorOverlay { .. } => {
                // Dismissed via on_escape
            }
            _ => {}
        }
    }

    pub fn on_escape(&mut self) {
        match &self.screen {
            Screen::SetApiKey | Screen::SetApiUrl => {
                // Skip config and go directly (may fail later)
                let _ = save_config(&self.config);
                self.start_health_check();
            }
            Screen::DemoExplain(_) | Screen::ShowConfig => {
                self.list_state.select(Some(0));
                self.screen = Screen::MainMenu;
            }
            Screen::DemoPrompt(item) => {
                let item = item.clone();
                self.screen = Screen::DemoExplain(item);
            }
            Screen::DemoResult { .. } => {
                self.list_state.select(Some(0));
                self.screen = Screen::MainMenu;
            }
            Screen::ErrorOverlay { .. } => {
                self.screen = Screen::MainMenu;
            }
            _ => {}
        }
    }

    pub fn on_up(&mut self) {
        match &self.screen {
            Screen::MainMenu => {
                self.list_state.select_previous();
            }
            Screen::DemoResult { .. } => {
                self.scroll_offset = self.scroll_offset.saturating_sub(3);
            }
            _ => {}
        }
    }

    pub fn on_down(&mut self) {
        match &self.screen {
            Screen::MainMenu => {
                let max = MENU_ITEMS.len().saturating_sub(1);
                let next = self.list_state.selected().unwrap_or(0).saturating_add(1).min(max);
                self.list_state.select(Some(next));
            }
            Screen::DemoResult { .. } => {
                self.scroll_offset = self.scroll_offset.saturating_add(3);
            }
            _ => {}
        }
    }

    fn start_health_check(&mut self) {
        self.screen = Screen::HealthCheck;
        let tx = self.tx.clone();
        let url = self.config.effective_url().to_string();
        std::thread::spawn(move || {
            let result = api::health_check(&url);
            let _ = tx.send(ApiResult::Health(result));
        });
    }

    fn launch_demo(&mut self, item: DemoItem, prompt: String) {
        self.screen = Screen::DemoLoading(item.clone());
        let tx = self.tx.clone();
        let url = self.config.effective_url().to_string();
        let key = self.config.api_key.clone();

        match item {
            DemoItem::FastGen | DemoItem::StandardGen | DemoItem::StrongGen => {
                let effort = match item {
                    DemoItem::FastGen => Some("none"),
                    DemoItem::StrongGen => Some("high"),
                    _ => None,
                };
                let effort_owned = effort.map(|s| s.to_string());
                let item_clone = item;
                std::thread::spawn(move || {
                    let result = api::generate(
                        &url,
                        &key,
                        &prompt,
                        effort_owned.as_deref(),
                    )
                    .map(|r| {
                        let meta = ResultMeta {
                            tier: r.tier.clone(),
                            gen_time: r.generation_time,
                            tokens: r.tokens_used,
                            steps: None,
                            model: r.backend_used.clone(),
                        };
                        (r.code, meta)
                    });
                    let _ = tx.send(ApiResult::Gen(item_clone, result));
                });
            }
            DemoItem::Agent => {
                std::thread::spawn(move || {
                    let result = api::agent_run(&url, &key, &prompt).map(|r| {
                        let mut out = String::new();
                        if let Some(trace) = &r.trace {
                            if !trace.is_empty() {
                                for step in trace {
                                    let tool = step.tool.as_deref().unwrap_or("?");
                                    let elapsed = step.elapsed_s.map(|e| format!("  ({:.2}s)", e)).unwrap_or_default();
                                    out.push_str(&format!("─── step {} · {}{} ───\n", step.step.unwrap_or(0), tool, elapsed));
                                    if let Some(res) = &step.result {
                                        let truncated = if res.len() > 300 {
                                            format!("{}…", &res[..300])
                                        } else {
                                            res.clone()
                                        };
                                        out.push_str(&truncated);
                                        out.push('\n');
                                    }
                                    out.push('\n');
                                }
                                out.push_str("─── answer ───\n");
                            }
                        }
                        out.push_str(&r.output);
                        let meta = ResultMeta {
                            tier: Some("agent".to_string()),
                            gen_time: r.total_time_s,
                            tokens: None,
                            steps: r.steps_taken,
                            model: r.model.clone(),
                        };
                        (out, meta)
                    });
                    let _ = tx.send(ApiResult::Agent(DemoItem::Agent, result));
                });
            }
            DemoItem::Search => {
                std::thread::spawn(move || {
                    let result = api::search(&url, &key, &prompt).map(|results| {
                        let text = if results.is_empty() {
                            "No results found.".to_string()
                        } else {
                            results
                                .iter()
                                .enumerate()
                                .map(|(i, r)| {
                                    format!(
                                        "{}. {}\n   {}\n   {}\n",
                                        i + 1,
                                        r.title,
                                        r.url,
                                        r.snippet
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        };
                        let meta = ResultMeta {
                            tier: Some("search".to_string()),
                            gen_time: None,
                            tokens: None,
                            steps: None,
                            model: Some("ddg".to_string()),
                        };
                        (text, meta)
                    });
                    let _ = tx.send(ApiResult::Search(DemoItem::Search, result));
                });
            }
        }
    }
}
