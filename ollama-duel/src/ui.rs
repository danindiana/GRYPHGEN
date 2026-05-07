use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, BorderType, Borders, List, ListItem, Padding, Paragraph, Scrollbar,
        ScrollbarOrientation, ScrollbarState, Wrap,
    },
    Frame,
};

use crate::app::{App, ConvItem, ConvRole, InputMode, ServerStatus};

// ── Colour palette ─────────────────────────────────────────────────────────────

const C_BORDER: Color = Color::DarkGray;
const C_BORDER_ACTIVE: Color = Color::Cyan;
const C_TITLE: Color = Color::Cyan;
const C_USER: Color = Color::Cyan;
const C_ACTOR: Color = Color::Green;
const C_CRITIC: Color = Color::Yellow;
const C_TOOL_CALL: Color = Color::Magenta;
const C_TOOL_RESULT: Color = Color::Blue;
const C_TOOL_ERR: Color = Color::Red;
const C_SYSTEM: Color = Color::DarkGray;
const C_STREAM: Color = Color::White;
const C_STATUS_OK: Color = Color::Green;
const C_STATUS_ERR: Color = Color::Red;
const C_STATUS_BUSY: Color = Color::Yellow;

// ── Top-level render ──────────────────────────────────────────────────────────

pub fn render(f: &mut Frame, app: &App) {
    let area = f.area();

    // Root layout: header / body / footer
    let root = Layout::vertical([
        Constraint::Length(1), // title bar
        Constraint::Min(0),    // body
        Constraint::Length(3), // input
        Constraint::Length(1), // status bar
    ])
    .split(area);

    render_title_bar(f, app, root[0]);

    // Body: left sidebar / conversation
    let body = Layout::horizontal([
        Constraint::Percentage(24),
        Constraint::Min(0),
    ])
    .split(root[1]);

    render_mcp_panel(f, app, body[0]);
    render_conversation(f, app, body[1]);
    render_input(f, app, root[2]);
    render_status_bar(f, app, root[3]);
}

// ── Title bar ─────────────────────────────────────────────────────────────────

fn render_title_bar(f: &mut Frame, app: &App, area: Rect) {
    let spinning = app.spinner();
    let actor_label = format!("🎭 {}", app.config.actor.model);
    let critic_label = format!("📊 {}", app.config.critic.model);
    let round_label = if app.is_processing {
        format!(
            "  {} Round {}/{}",
            spinning,
            app.current_round,
            app.config.orchestration.max_revision_rounds + 1
        )
    } else {
        String::new()
    };

    let mcp_count = app.mcp_servers.iter().filter(|s| s.status == ServerStatus::Connected).count();
    let mcp_label = format!("MCP:{}", mcp_count);

    let title = Line::from(vec![
        Span::styled(" 🤺 ollama-duel ", Style::new().fg(Color::Black).bg(C_TITLE).add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::styled(&actor_label, Style::new().fg(C_ACTOR)),
        Span::styled(" ↔ ", Style::new().fg(C_BORDER)),
        Span::styled(&critic_label, Style::new().fg(C_CRITIC)),
        Span::styled(&round_label, Style::new().fg(C_STATUS_BUSY)),
        Span::raw("  "),
        Span::styled(&mcp_label, Style::new().fg(C_TITLE)),
    ]);

    f.render_widget(Paragraph::new(title), area);
}

// ── MCP panel (left sidebar) ──────────────────────────────────────────────────

fn render_mcp_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(C_BORDER))
        .title(Span::styled(" MCP ", Style::new().fg(C_TITLE).add_modifier(Modifier::BOLD)))
        .padding(Padding::horizontal(1));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Split inner into servers section and tools section
    let sections = Layout::vertical([
        Constraint::Min(4),    // server list
        Constraint::Length(1), // divider
        Constraint::Min(0),    // tools list
    ])
    .split(inner);

    // ── Server list ──
    let server_items: Vec<ListItem> = if app.mcp_servers.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "No servers configured",
            Style::new().fg(C_SYSTEM),
        )))]
    } else {
        app.mcp_servers
            .iter()
            .map(|s| {
                let (icon, color) = match &s.status {
                    ServerStatus::Connected => ("✓", C_STATUS_OK),
                    ServerStatus::Connecting => ("⟳", C_STATUS_BUSY),
                    ServerStatus::Pending => ("○", C_SYSTEM),
                    ServerStatus::Error(_) => ("✗", C_STATUS_ERR),
                };
                ListItem::new(Line::from(vec![
                    Span::styled(icon, Style::new().fg(color)),
                    Span::raw(" "),
                    Span::styled(&s.name, Style::new().fg(Color::White)),
                    Span::styled(
                        format!(" ({})", s.tools.len()),
                        Style::new().fg(C_SYSTEM),
                    ),
                ]))
            })
            .collect()
    };

    let servers_widget = List::new(server_items).block(
        Block::default()
            .borders(Borders::NONE)
            .title(Span::styled("Servers", Style::new().fg(C_BORDER))),
    );
    f.render_widget(servers_widget, sections[0]);

    // ── Divider ──
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(sections[1].width as usize),
            Style::new().fg(C_BORDER),
        ))),
        sections[1],
    );

    // ── Tools list ──
    let all_tools: Vec<ListItem> = app
        .mcp_servers
        .iter()
        .filter(|s| s.status == ServerStatus::Connected)
        .flat_map(|s| {
            s.tools.iter().map(|t| {
                ListItem::new(Line::from(vec![
                    Span::styled("▸ ", Style::new().fg(C_TOOL_CALL)),
                    Span::styled(t, Style::new().fg(Color::White)),
                ]))
            })
        })
        .collect();

    let tools_widget = List::new(all_tools).block(
        Block::default()
            .borders(Borders::NONE)
            .title(Span::styled("Tools", Style::new().fg(C_BORDER))),
    );
    f.render_widget(tools_widget, sections[2]);
}

// ── Conversation panel (right) ────────────────────────────────────────────────

fn render_conversation(f: &mut Frame, app: &App, area: Rect) {
    let is_editing = app.mode == InputMode::Editing;
    let border_color = if is_editing { C_BORDER } else { C_BORDER_ACTIVE };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border_color))
        .title(Span::styled(" Conversation ", Style::new().fg(C_TITLE).add_modifier(Modifier::BOLD)));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Build lines from conversation items
    let mut lines: Vec<Line> = vec![];

    for item in &app.conv {
        render_conv_item(item, inner.width as usize, &mut lines);
        lines.push(Line::raw(""));
    }

    let total_lines = lines.len() as u16;
    let visible_height = inner.height;

    // Clamp scroll
    let max_scroll = total_lines.saturating_sub(visible_height);
    let scroll = app.conv_scroll.min(max_scroll);

    let paragraph = Paragraph::new(Text::from(lines))
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });

    f.render_widget(paragraph, inner);

    // Scrollbar
    if total_lines > visible_height {
        let mut scroll_state = ScrollbarState::new(max_scroll as usize).position(scroll as usize);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓")),
            area,
            &mut scroll_state,
        );
    }
}

fn render_conv_item(item: &ConvItem, _width: usize, lines: &mut Vec<Line>) {
    match &item.role {
        ConvRole::User => {
            lines.push(Line::from(vec![
                Span::styled("▶ You", Style::new().fg(C_USER).add_modifier(Modifier::BOLD)),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(Color::White),
                )));
            }
        }

        ConvRole::Actor { model, round } => {
            lines.push(Line::from(vec![
                Span::styled("🎭 Actor", Style::new().fg(C_ACTOR).add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!(" [{}] round {}", model, round),
                    Style::new().fg(C_SYSTEM),
                ),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(Color::White),
                )));
            }
        }

        ConvRole::ActorStreaming => {
            lines.push(Line::from(vec![
                Span::styled("🎭 Actor", Style::new().fg(C_ACTOR).add_modifier(Modifier::BOLD)),
                Span::styled(" (streaming…)", Style::new().fg(C_SYSTEM)),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(C_STREAM),
                )));
            }
        }

        ConvRole::Critic { model } => {
            let score = item.score.unwrap_or(0);
            let score_color = score_color(score);
            let should_revise = item.should_revise.unwrap_or(false);
            let verdict = if should_revise { "↺ revise" } else { "✓ accept" };

            lines.push(Line::from(vec![
                Span::styled("📊 Critic", Style::new().fg(C_CRITIC).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" [{}]", model), Style::new().fg(C_SYSTEM)),
                Span::raw("  "),
                Span::styled(
                    format!("Score: {}/10", score),
                    Style::new().fg(score_color).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(
                    verdict,
                    Style::new()
                        .fg(if should_revise { C_STATUS_BUSY } else { C_STATUS_OK })
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(Color::White),
                )));
            }
        }

        ConvRole::CriticStreaming => {
            lines.push(Line::from(vec![
                Span::styled("📊 Critic", Style::new().fg(C_CRITIC).add_modifier(Modifier::BOLD)),
                Span::styled(" (evaluating…)", Style::new().fg(C_SYSTEM)),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(C_STREAM),
                )));
            }
        }

        ConvRole::ToolCall { server, tool } => {
            lines.push(Line::from(vec![
                Span::styled("🔧 Tool call", Style::new().fg(C_TOOL_CALL).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" {}::{}", server, tool), Style::new().fg(C_TOOL_CALL)),
            ]));
            for l in item.content.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(Color::DarkGray),
                )));
            }
        }

        ConvRole::ToolResult { tool, is_error } => {
            let (icon, color) = if *is_error {
                ("✗", C_TOOL_ERR)
            } else {
                ("✓", C_TOOL_RESULT)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{} Result ({})", icon, tool),
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]));
            // Only show first few lines of long results
            let result_lines: Vec<&str> = item.content.lines().take(12).collect();
            for l in &result_lines {
                lines.push(Line::from(Span::styled(
                    format!("  {}", l),
                    Style::new().fg(Color::DarkGray),
                )));
            }
            if item.content.lines().count() > 12 {
                lines.push(Line::from(Span::styled(
                    "  … (truncated)",
                    Style::new().fg(C_SYSTEM),
                )));
            }
        }

        ConvRole::SystemNote => {
            lines.push(Line::from(Span::styled(
                format!("  ℹ  {}", item.content),
                Style::new().fg(C_SYSTEM).add_modifier(Modifier::ITALIC),
            )));
        }
    }
}

// ── Input bar ─────────────────────────────────────────────────────────────────

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let is_editing = app.mode == InputMode::Editing;
    let (border_color, title_text) = if app.is_processing {
        (C_STATUS_BUSY, format!(" {} Processing… ", app.spinner()))
    } else if is_editing {
        (C_BORDER_ACTIVE, " Input — Enter to send, Esc to cancel ".into())
    } else {
        (C_BORDER, " [i] to type, [q] quit, [↑↓/PgUp/PgDn] scroll ".into())
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border_color))
        .title(Span::styled(title_text, Style::new().fg(border_color)));

    let inner = block.inner(area);
    f.render_widget(block, area);

    // Show input text with cursor
    let display = if is_editing {
        let before = &app.input[..app.cursor_pos];
        let after = &app.input[app.cursor_pos..];
        Line::from(vec![
            Span::raw(before),
            Span::styled("█", Style::new().fg(C_BORDER_ACTIVE)),
            Span::styled(after, Style::new().fg(Color::White)),
        ])
    } else if app.is_processing {
        Line::from(Span::styled("⋯", Style::new().fg(C_SYSTEM)))
    } else {
        Line::from(Span::styled(
            "Press [i] or [Enter] to start typing…",
            Style::new().fg(C_SYSTEM).add_modifier(Modifier::ITALIC),
        ))
    };

    f.render_widget(Paragraph::new(display), inner);
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn render_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let status_color = if app.is_processing {
        C_STATUS_BUSY
    } else if app.status.starts_with("Error") {
        C_STATUS_ERR
    } else {
        C_STATUS_OK
    };

    let line = Line::from(vec![
        Span::styled(
            format!(" {} {} ", app.spinner(), app.status),
            Style::new().fg(status_color),
        ),
        Span::styled(
            format!(
                "  actor:{} critic:{} ",
                app.config.actor.model, app.config.critic.model
            ),
            Style::new().fg(C_SYSTEM),
        ),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn score_color(score: u8) -> Color {
    match score {
        9..=10 => Color::LightGreen,
        7..=8 => Color::Green,
        5..=6 => Color::Yellow,
        _ => Color::Red,
    }
}
