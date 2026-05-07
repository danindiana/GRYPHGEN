mod actor_critic;
mod app;
mod config;
mod mcp;
mod ollama;
mod ui;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::Mutex;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "ollama-duel",
    version,
    about = "MCP-native actor/critic LLM terminal UI powered by Ollama + Ratatui"
)]
struct Cli {
    /// Path to a config file (defaults to ~/.config/ollama-duel/config.toml or ./config.toml)
    #[arg(short, long)]
    config: Option<String>,

    /// Override the actor model name
    #[arg(short, long)]
    actor: Option<String>,

    /// Override the critic model name
    #[arg(short = 'C', long)]
    critic: Option<String>,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama: String,
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // ── Load config ────────────────────────────────────────────────────────
    let mut cfg = match &cli.config {
        Some(path) => config::load_config_from(path)?,
        None => config::load_config()?,
    };

    // CLI overrides
    if let Some(a) = cli.actor { cfg.actor.model = a; }
    if let Some(c) = cli.critic { cfg.critic.model = c; }
    cfg.ollama.base_url = cli.ollama;

    let cfg = Arc::new(cfg);

    // ── File-based logging (never pollutes the TUI) ────────────────────────
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("ollama-duel");
    std::fs::create_dir_all(&log_dir).ok();

    let file_appender = tracing_appender::rolling::daily(&log_dir, "ollama-duel.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ollama_duel=debug".parse().unwrap()),
        )
        .init();

    tracing::info!("ollama-duel starting, actor={}, critic={}", cfg.actor.model, cfg.critic.model);

    // ── Set up terminal ────────────────────────────────────────────────────
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // ── Build app ──────────────────────────────────────────────────────────
    let mcp_clients = Arc::new(Mutex::new(HashMap::new()));
    let mut app = app::App::new(cfg.clone(), mcp_clients);

    // Start MCP server connections and Ollama ping in background
    app.init_mcp_servers().await;

    // ── Run ────────────────────────────────────────────────────────────────
    let run_result = app.run(&mut terminal).await;

    // ── Restore terminal ───────────────────────────────────────────────────
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // Re-raise any error after terminal is restored
    run_result?;
    Ok(())
}
