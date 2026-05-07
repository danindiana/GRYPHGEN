#![allow(dead_code)]
mod app;
mod api;
mod config;
mod events;
mod ui;

use crossterm::event::{KeyCode, KeyModifiers};
use tui_input::backend::crossterm::EventHandler;
use crossterm::event::Event;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut terminal = ratatui::init();
    let result = run(&mut terminal);
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal) -> color_eyre::Result<()> {
    let mut app = app::App::new();

    loop {
        // Tick app (checks mpsc channel for API results)
        app.tick();

        terminal.draw(|frame| ui::render(frame, &app))?;

        if app.should_quit {
            break;
        }

        // Poll for keyboard events with a short timeout to keep spinner animated
        if let Some(ev) = events::poll(16) {
            match ev {
                events::AppEvent::Key(code, modifiers) => {
                    // Let tui-input handle character input when we're in a text field
                    let in_text_input = matches!(
                        &app.screen,
                        app::Screen::SetApiKey | app::Screen::SetApiUrl | app::Screen::DemoPrompt(_)
                    );

                    match (code, modifiers) {
                        (KeyCode::Char('c'), KeyModifiers::CONTROL)
                        | (KeyCode::Char('q'), KeyModifiers::NONE) => {
                            if !in_text_input {
                                app.should_quit = true;
                            } else if code == KeyCode::Char('c') {
                                app.should_quit = true;
                            }
                        }
                        (KeyCode::Enter, _) => app.on_enter(),
                        (KeyCode::Esc, _) => app.on_escape(),
                        (KeyCode::Up, _) => app.on_up(),
                        (KeyCode::Down, _) => app.on_down(),
                        (KeyCode::Char(c), mods) if in_text_input => {
                            let key_event = crossterm::event::KeyEvent::new(
                                KeyCode::Char(c),
                                mods,
                            );
                            app.input.handle_event(&Event::Key(key_event));
                        }
                        (KeyCode::Backspace, _) if in_text_input => {
                            let key_event = crossterm::event::KeyEvent::new(
                                KeyCode::Backspace,
                                KeyModifiers::NONE,
                            );
                            app.input.handle_event(&Event::Key(key_event));
                        }
                        (KeyCode::Delete, _) if in_text_input => {
                            let key_event = crossterm::event::KeyEvent::new(
                                KeyCode::Delete,
                                KeyModifiers::NONE,
                            );
                            app.input.handle_event(&Event::Key(key_event));
                        }
                        (KeyCode::Left, _) if in_text_input => {
                            let key_event = crossterm::event::KeyEvent::new(
                                KeyCode::Left,
                                KeyModifiers::NONE,
                            );
                            app.input.handle_event(&Event::Key(key_event));
                        }
                        (KeyCode::Right, _) if in_text_input => {
                            let key_event = crossterm::event::KeyEvent::new(
                                KeyCode::Right,
                                KeyModifiers::NONE,
                            );
                            app.input.handle_event(&Event::Key(key_event));
                        }
                        _ => {}
                    }
                }
                events::AppEvent::Tick => {}
            }
        }
    }

    Ok(())
}
