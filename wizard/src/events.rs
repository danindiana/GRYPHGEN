use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use std::time::Duration;

pub enum AppEvent {
    Key(KeyCode, KeyModifiers),
    Tick,
}

pub fn poll(timeout_ms: u64) -> Option<AppEvent> {
    if event::poll(Duration::from_millis(timeout_ms)).unwrap_or(false) {
        if let Ok(Event::Key(KeyEvent { code, kind, modifiers, .. })) = event::read() {
            if kind == KeyEventKind::Press {
                return Some(AppEvent::Key(code, modifiers));
            }
        }
    }
    None
}
