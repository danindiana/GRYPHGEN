use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
struct GenPayload<'a> {
    prompt: &'a str,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    include_docs: bool,
    max_tokens: u32,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
pub struct GenResponse {
    pub code: String,
    pub tier: Option<String>,
    pub backend_used: Option<String>,
    pub generation_time: Option<f64>,
    pub tokens_used: Option<u32>,
}

#[derive(Debug, Serialize)]
struct AgentPayload<'a> {
    task: &'a str,
    max_steps: u32,
}

#[derive(Debug, Deserialize)]
pub struct AgentStep {
    pub step: Option<u32>,
    pub tool: Option<String>,
    pub elapsed_s: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct AgentResponse {
    pub output: String,
    pub trace: Option<Vec<AgentStep>>,
    pub steps_taken: Option<u32>,
    pub total_time_s: Option<f64>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    service: Option<String>,
}

pub fn health_check(base_url: &str) -> Result<String, String> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let resp = ureq::get(&url)
        .set("User-Agent", "gryphgen-wizard/0.1")
        .call()
        .map_err(|e| e.to_string())?;
    let body: serde_json::Value = resp.into_json().map_err(|e| e.to_string())?;
    let status = body
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let service = body
        .get("service")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    Ok(format!("{} {}", status, service))
}

pub fn generate(
    base_url: &str,
    key: &str,
    prompt: &str,
    reasoning_effort: Option<&str>,
) -> Result<GenResponse, String> {
    let url = format!(
        "{}/api/v1/code/generate",
        base_url.trim_end_matches('/')
    );
    let payload = GenPayload {
        prompt,
        language: "python",
        reasoning_effort,
        include_docs: true,
        max_tokens: 4096,
        temperature: 0.7,
    };
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .set("X-API-Key", key)
        .set("User-Agent", "gryphgen-wizard/0.1")
        .send_json(ureq::json!(payload))
        .map_err(|e| format!("{}", e))?;
    resp.into_json::<GenResponse>().map_err(|e| e.to_string())
}

pub fn agent_run(base_url: &str, key: &str, task: &str) -> Result<AgentResponse, String> {
    let url = format!("{}/api/v1/agent/run", base_url.trim_end_matches('/'));
    let payload = AgentPayload {
        task,
        max_steps: 10,
    };
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .set("X-API-Key", key)
        .set("User-Agent", "gryphgen-wizard/0.1")
        .timeout(std::time::Duration::from_secs(600))
        .send_json(ureq::json!(payload))
        .map_err(|e| format!("{}", e))?;
    resp.into_json::<AgentResponse>().map_err(|e| e.to_string())
}

pub fn search(base_url: &str, key: &str, query: &str) -> Result<Vec<SearchResult>, String> {
    let url = format!(
        "{}/api/v1/tools/search?q={}&max_results=5",
        base_url.trim_end_matches('/'),
        urlencoding(query)
    );
    let resp = ureq::get(&url)
        .set("X-API-Key", key)
        .set("User-Agent", "gryphgen-wizard/0.1")
        .call()
        .map_err(|e| e.to_string())?;
    let body: SearchResponse = resp.into_json().map_err(|e| e.to_string())?;
    Ok(body.results)
}

fn urlencoding(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c]
            }
            ' ' => vec!['+'],
            c => {
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf);
                bytes
                    .bytes()
                    .flat_map(|b| {
                        let hi = b >> 4;
                        let lo = b & 0xf;
                        vec!['%', hex_char(hi), hex_char(lo)]
                    })
                    .collect()
            }
        })
        .collect()
}

fn hex_char(n: u8) -> char {
    if n < 10 {
        (b'0' + n) as char
    } else {
        (b'a' + n - 10) as char
    }
}
