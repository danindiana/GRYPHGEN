# API Documentation

## Overview

The Novelty LLM System provides a RESTful API for querying large language models with automatic novelty detection and intelligent caching.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API uses JWT-based authentication (to be implemented in future versions).

## Endpoints

### Health Check

#### `GET /health`

Check the health status of the API and its components.

**Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-11-18T12:00:00Z",
  "components": {
    "novelty_engine": "ok",
    "semantic_cache": "ok",
    "response_cache": "ok"
  }
}
```

### Query LLM

#### `POST /query`

Submit a query to the LLM with automatic novelty detection and caching.

**Request Body:**

```json
{
  "prompt": "Explain quantum entanglement",
  "model": "llama2",
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false,
  "documents": [],
  "metadata": {}
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| prompt | string | Yes | - | The user's query (1-10000 chars) |
| model | string | No | null | Specific model to use |
| temperature | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| max_tokens | int | No | 2048 | Maximum tokens to generate |
| stream | boolean | No | false | Enable streaming response |
| documents | array | No | [] | Context documents |
| metadata | object | No | {} | Additional metadata |

**Response:**

```json
{
  "response": "Quantum entanglement is a phenomenon...",
  "model": "llama2",
  "novelty_score": 0.78,
  "novelty_level": "high",
  "cached": false,
  "cache_hit_similarity": null,
  "tokens_used": 156,
  "processing_time_ms": 234.5,
  "timestamp": "2024-11-18T12:00:00Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| response | string | The generated response |
| model | string | Model used for generation |
| novelty_score | float | Novelty score (0.0-1.0) |
| novelty_level | string | Classification: very_low, low, medium, high, very_high |
| cached | boolean | Whether response was from cache |
| cache_hit_similarity | float | Similarity score if from semantic cache |
| tokens_used | integer | Number of tokens in response |
| processing_time_ms | float | Total processing time |
| timestamp | string | Response timestamp |

**Example:**

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the meaning of life?",
    "temperature": 0.8,
    "max_tokens": 1024
  }'
```

### Cache Management

#### `GET /cache/stats`

Get statistics about cache performance.

**Response:**

```json
{
  "semantic_cache": {
    "total_entries": 1245,
    "hit_rate": 0.67,
    "avg_similarity": 0.89,
    "total_hits": 8340,
    "total_misses": 4123
  },
  "response_cache": {
    "used_memory_mb": 145.2,
    "max_memory_mb": 1000,
    "keys": 3456,
    "hit_rate": 0.45,
    "evictions": 234
  }
}
```

**Example:**

```bash
curl http://localhost:8080/cache/stats
```

#### `DELETE /cache/clear`

Clear all caches.

**Response:**

```json
{
  "semantic_cache_cleared": 1245,
  "response_cache_cleared": 3456
}
```

**Example:**

```bash
curl -X DELETE http://localhost:8080/cache/clear
```

### Metrics

#### `GET /metrics`

Prometheus metrics endpoint for monitoring.

**Response:** Prometheus text format

```
# HELP novelty_llm_requests_total Total requests
# TYPE novelty_llm_requests_total counter
novelty_llm_requests_total{endpoint="/query",status="success"} 12345

# HELP novelty_llm_request_duration_seconds Request duration
# TYPE novelty_llm_request_duration_seconds histogram
novelty_llm_request_duration_seconds_bucket{endpoint="/query",le="0.1"} 1234
novelty_llm_request_duration_seconds_bucket{endpoint="/query",le="0.5"} 5678
...
```

**Example:**

```bash
curl http://localhost:8080/metrics
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-11-18T12:00:00Z"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Endpoint does not exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable - System overloaded |

## Rate Limiting

The API implements rate limiting based on:

- Requests per minute
- Tokens per day
- User tier

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1699999999
```

## Pagination

For endpoints that return lists, pagination is supported:

```
GET /queries?page=1&limit=50
```

## Filtering and Sorting

Query parameters for filtering:

```
GET /queries?model=llama2&novelty_level=high&sort=-created_at
```

## WebSocket Streaming

For streaming responses, use WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/query');

ws.send(JSON.stringify({
  prompt: "Explain relativity",
  stream: true
}));

ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data);
  console.log(chunk.token);
};
```

## SDKs

### Python SDK

```python
from novelty_llm import Client

client = Client(base_url="http://localhost:8080")

response = await client.query(
    prompt="What is quantum computing?",
    temperature=0.7
)

print(response.text)
print(f"Novelty: {response.novelty_score}")
```

### JavaScript SDK

```javascript
import { NoveltyLLMClient } from 'novelty-llm-js';

const client = new NoveltyLLMClient({
  baseUrl: 'http://localhost:8080'
});

const response = await client.query({
  prompt: 'What is quantum computing?',
  temperature: 0.7
});

console.log(response.text);
console.log(`Novelty: ${response.noveltyScore}`);
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## API Versioning

The API uses URL versioning:

```
http://localhost:8080/v1/query
```

Current version: `v1`

## Changelog

### v0.1.0 (2024-11-18)

- Initial release
- Basic query endpoint
- Novelty scoring
- Multi-tier caching
- Metrics endpoint
