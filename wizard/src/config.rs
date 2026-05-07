use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub api_key: String,
    pub api_url: String,
}

impl Config {
    pub fn is_configured(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn effective_url(&self) -> &str {
        if self.api_url.is_empty() {
            "https://api.grug.ai"
        } else {
            &self.api_url
        }
    }
}

fn config_path() -> PathBuf {
    let mut p = dirs_home();
    p.push(".config/gryphgen/config.toml");
    p
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/root"))
}

pub fn load() -> Config {
    let path = config_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Config::default(),
    };
    let table: toml::Value = match toml::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Config::default(),
    };
    Config {
        api_key: table
            .get("api_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        api_url: table
            .get("api_url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
    }
}

pub fn save(cfg: &Config) -> Result<(), String> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut lines = Vec::new();
    if !cfg.api_key.is_empty() {
        lines.push(format!("api_key = \"{}\"", cfg.api_key));
    }
    let url = if cfg.api_url.is_empty() {
        "https://api.grug.ai"
    } else {
        &cfg.api_url
    };
    lines.push(format!("api_url = \"{}\"", url));
    std::fs::write(&path, lines.join("\n") + "\n").map_err(|e| e.to_string())
}
