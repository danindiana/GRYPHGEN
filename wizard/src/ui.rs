use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{
        Block, Borders, Clear, List, ListItem, Paragraph, Wrap,
    },
};
use crate::app::{App, Screen, DemoItem, MENU_ITEMS};

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

fn spinner_frame(tick: u64) -> &'static str {
    SPINNER[(tick / 3) as usize % SPINNER.len()]
}

pub fn render(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Outer chrome: header + footer
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(0),    // content
            Constraint::Length(3), // footer
        ])
        .split(area);

    render_header(frame, outer[0], app);
    render_footer(frame, outer[2], app);

    let content = outer[1];

    match &app.screen {
        Screen::Welcome => render_welcome(frame, content),
        Screen::ConfigCheck => render_simple_message(frame, content, "Checking configuration…", Color::Yellow),
        Screen::SetApiKey => render_text_input(
            frame,
            content,
            "Enter your API key",
            "Paste your gryphgen API key below.\nGet one at https://api.grug.ai\n\nPress ENTER to confirm, ESC to skip.",
            app,
        ),
        Screen::SetApiUrl => render_text_input(
            frame,
            content,
            "API URL (optional)",
            "Leave blank to use the default: https://api.grug.ai\n\nPress ENTER to confirm or use the default.",
            app,
        ),
        Screen::HealthCheck => render_loading(
            frame,
            content,
            &format!("{} Connecting to {}…", spinner_frame(app.tick), app.config.effective_url()),
            Color::Cyan,
        ),
        Screen::MainMenu => render_main_menu(frame, content, app),
        Screen::DemoExplain(item) => render_demo_explain(frame, content, item),
        Screen::DemoPrompt(item) => render_demo_prompt(frame, content, item, app),
        Screen::DemoLoading(item) => render_loading(
            frame,
            content,
            &format!("{} {}", spinner_frame(app.tick), item.loading_message()),
            Color::Cyan,
        ),
        Screen::DemoResult { item, output, meta } => {
            render_result(frame, content, item, output, meta, app)
        }
        Screen::ShowConfig => render_show_config(frame, content, app),
        Screen::ErrorOverlay { message, .. } => {
            render_main_menu(frame, content, app);
            render_error_popup(frame, area, message);
        }
    }
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let url_display = if app.config.effective_url().len() > 30 {
        app.config.effective_url().to_string()
    } else {
        app.config.effective_url().to_string()
    };
    let title = format!(
        " GRYPHGEN Wizard  {}  v0.1 ",
        if app.config.is_configured() { "✓" } else { "○" }
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let url_line = Paragraph::new(url_display)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    frame.render_widget(url_line, inner);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let hints = match &app.screen {
        Screen::Welcome => " ENTER  start ",
        Screen::SetApiKey | Screen::SetApiUrl => " ENTER  confirm   ESC  skip ",
        Screen::MainMenu => " ↑↓  navigate   ENTER  select   q  quit ",
        Screen::DemoExplain(_) => " ENTER  continue   ESC  back ",
        Screen::DemoPrompt(_) => " ENTER  run   ESC  back ",
        Screen::DemoResult { .. } => " ↑↓  scroll   ENTER/ESC  back to menu ",
        Screen::ShowConfig => " ENTER/ESC  back ",
        Screen::ErrorOverlay { .. } => " ENTER/ESC  dismiss ",
        _ => " q  quit ",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(hints)
            .style(Style::default().fg(Color::DarkGray))
            .alignment(Alignment::Center),
        inner,
    );
}

fn render_welcome(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let text = Text::from(vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Welcome to the GRYPHGEN Operator Wizard",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  GRYPHGEN is a local AI code generation platform."),
        Line::from("  It runs two models on your GPU cluster:"),
        Line::from(""),
        Line::from(vec![
            Span::styled("    THINKER  ", Style::default().fg(Color::Yellow)),
            Span::raw("deepseek-r1:14b  (RTX 5080, GPU 0)  — reasoning & planning"),
        ]),
        Line::from(vec![
            Span::styled("    CODER    ", Style::default().fg(Color::Cyan)),
            Span::raw("qwen2.5-coder:7b (RTX 3080, GPU 1)  — fast code generation"),
        ]),
        Line::from(""),
        Line::from("  This wizard will:"),
        Line::from("    1. Configure your API key (one-time)"),
        Line::from("    2. Verify the connection to api.grug.ai"),
        Line::from("    3. Walk you through live code generation examples"),
        Line::from(""),
        Line::from("  Generated code appears right here in the terminal."),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ENTER to begin →",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
    ]);

    frame.render_widget(
        Paragraph::new(text).wrap(Wrap { trim: false }),
        inner,
    );
}

fn render_text_input(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    instructions: &str,
    app: &App,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    // Instructions
    let inst_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            format!(" {} ", title),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
    let inst_inner = inst_block.inner(chunks[0]);
    frame.render_widget(inst_block, chunks[0]);
    frame.render_widget(
        Paragraph::new(instructions)
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: false }),
        inst_inner,
    );

    // Input field
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Input ");
    let input_inner = input_block.inner(chunks[1]);
    frame.render_widget(input_block, chunks[1]);

    let value = app.input.value();
    let display = if value.is_empty() {
        Span::styled("(type here…)", Style::default().fg(Color::DarkGray))
    } else {
        Span::styled(value, Style::default().fg(Color::White))
    };
    frame.render_widget(Paragraph::new(Line::from(display)), input_inner);

    // Cursor
    let cursor_x = input_inner.x + (app.input.visual_cursor() as u16).min(input_inner.width.saturating_sub(1));
    frame.set_cursor_position((cursor_x, input_inner.y));
}

fn render_loading(frame: &mut Frame, area: Rect, message: &str, color: Color) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Center vertically
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(40),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(inner);

    frame.render_widget(
        Paragraph::new(message)
            .style(Style::default().fg(color).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center),
        v[1],
    );
}

fn render_simple_message(frame: &mut Frame, area: Rect, msg: &str, color: Color) {
    render_loading(frame, area, msg, color);
}

fn render_main_menu(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    // Left: menu
    let menu_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(Span::styled(
            " What would you like to do? ",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
    let menu_inner = menu_block.inner(chunks[0]);
    frame.render_widget(menu_block, chunks[0]);

    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .map(|s| ListItem::new(format!("  {}  ", s)))
        .collect();

    let mut state = app.list_state.clone();
    let list = List::new(items)
        .highlight_style(
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    frame.render_stateful_widget(list, menu_inner, &mut state);

    // Right: contextual help
    let help_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" About ");
    let help_inner = help_block.inner(chunks[1]);
    frame.render_widget(help_block, chunks[1]);

    let selected = app.list_state.selected().unwrap_or(0);
    let demo_items = [
        DemoItem::FastGen,
        DemoItem::StandardGen,
        DemoItem::StrongGen,
        DemoItem::Agent,
        DemoItem::Search,
    ];

    let help_text = if selected < demo_items.len() {
        let item = &demo_items[selected];
        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                item.title(),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];
        for l in item.explain_lines() {
            lines.push(Line::from(Span::raw(l)));
        }
        lines
    } else if selected == 5 {
        vec![
            Line::from(Span::styled(
                "View Configuration",
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from("Shows your current API key (redacted)"),
            Line::from("and the configured endpoint URL."),
        ]
    } else {
        vec![Line::from("Exit the wizard.")]
    };

    frame.render_widget(
        Paragraph::new(help_text)
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: false }),
        help_inner,
    );
}

fn render_demo_explain(frame: &mut Frame, area: Rect, item: &DemoItem) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            format!(" {} ", item.title()),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = vec![Line::from("")];
    for l in item.explain_lines() {
        lines.push(Line::from(Span::raw(format!("  {}", l))));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Press ENTER to continue →",
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
    )));

    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }),
        inner,
    );
}

fn render_demo_prompt(frame: &mut Frame, area: Rect, item: &DemoItem, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    // Top: instruction
    let inst = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow))
        .title(Span::styled(
            format!(" {} — Edit prompt ", item.title()),
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));
    let inst_inner = inst.inner(chunks[0]);
    frame.render_widget(inst, chunks[0]);
    frame.render_widget(
        Paragraph::new(
            "  Edit the prompt below, or press ENTER to run with the default.",
        )
        .style(Style::default().fg(Color::Gray))
        .wrap(Wrap { trim: false }),
        inst_inner,
    );

    // Input
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Prompt ");
    let input_inner = input_block.inner(chunks[1]);
    frame.render_widget(input_block, chunks[1]);

    frame.render_widget(
        Paragraph::new(app.input.value())
            .style(Style::default().fg(Color::White)),
        input_inner,
    );

    let cx = input_inner.x + (app.input.visual_cursor() as u16).min(input_inner.width.saturating_sub(1));
    frame.set_cursor_position((cx, input_inner.y));
}

fn render_result(
    frame: &mut Frame,
    area: Rect,
    item: &DemoItem,
    output: &str,
    meta: &crate::app::ResultMeta,
    app: &App,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    // Meta bar
    let tier_str = meta.tier.as_deref().unwrap_or("?");
    let time_str = meta
        .gen_time
        .map(|t| format!("{:.2}s", t))
        .unwrap_or_else(|| meta.steps.map(|s| format!("{} steps", s)).unwrap_or_default());
    let tokens_str = meta
        .tokens
        .map(|t| format!("{} tokens", t))
        .unwrap_or_default();
    let model_str = meta.model.as_deref().unwrap_or("");

    let meta_text = format!(
        " tier={}  {}  {}  {} ",
        tier_str, time_str, tokens_str, model_str
    );

    let meta_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green))
        .title(Span::styled(
            format!(" {} — Result ", item.title()),
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        ));
    let meta_inner = meta_block.inner(chunks[0]);
    frame.render_widget(meta_block, chunks[0]);
    frame.render_widget(
        Paragraph::new(meta_text).style(Style::default().fg(Color::DarkGray)),
        meta_inner,
    );

    // Output
    let out_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(" Output (↑↓ scroll) ");
    let out_inner = out_block.inner(chunks[1]);
    frame.render_widget(out_block, chunks[1]);
    frame.render_widget(
        Paragraph::new(output)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false })
            .scroll((app.scroll_offset, 0)),
        out_inner,
    );
}

fn render_show_config(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Configuration ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let key_display = if app.config.api_key.len() > 8 {
        format!(
            "{}…{}",
            &app.config.api_key[..4],
            &app.config.api_key[app.config.api_key.len() - 4..]
        )
    } else if app.config.api_key.is_empty() {
        "(not set)".to_string()
    } else {
        "****".to_string()
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  api_key  ", Style::default().fg(Color::Cyan)),
            Span::styled(key_display, Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  api_url  ", Style::default().fg(Color::Cyan)),
            Span::styled(
                app.config.effective_url().to_string(),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  config   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "~/.config/gryphgen/config.toml",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ENTER or ESC to return to the menu.",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_error_popup(frame: &mut Frame, area: Rect, message: &str) {
    let popup_w = area.width.min(60);
    let popup_h = 8u16;
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(popup_w)) / 2,
        y: area.y + (area.height.saturating_sub(popup_h)) / 2,
        width: popup_w,
        height: popup_h,
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red))
        .title(Span::styled(
            " Error ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    frame.render_widget(
        Paragraph::new(message)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false }),
        inner,
    );
}
