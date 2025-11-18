# Novelty LLM System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/monitoring-prometheus-orange.svg)](https://prometheus.io/)

**A scalable, production-ready LLM platform with novelty scoring, semantic caching, and multi-tenant support.**

## 🎯 Overview

The Novelty LLM System is an enterprise-grade platform that enhances Large Language Model deployments with intelligent novelty detection and semantic caching. It intelligently routes queries, detects novel inputs, and optimizes resource usage through multi-tier caching strategies.

### Key Features

- 🔍 **Novelty Scoring Engine** - Multi-metric novelty detection using semantic distance, entropy, rarity, and temporal patterns
- 🚀 **Semantic Caching** - Vector-based similarity search for intelligent cache hits
- ⚡ **Response Caching** - High-performance Redis-based exact-match caching
- 🎯 **Priority Scheduling** - Novelty-aware request routing and scheduling
- 📊 **Observability** - Prometheus metrics, structured logging, and distributed tracing
- 🏢 **Multi-tenancy** - Tenant isolation and quota management
- 🔒 **Security** - PII detection, content filtering, and RBAC
- 🐳 **Cloud Native** - Docker, Kubernetes-ready, horizontal scaling

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

The system implements a distributed architecture with multiple specialized components:

```mermaid
flowchart TB
    subgraph "Client Layer"
        UI[OpenWebUI]
        CLI[CLI Client]
        SDK[SDK]
    end

    subgraph "Edge Layer"
        LB[Load Balancer]
        WAF[WAF/CDN]
    end

    subgraph "API Layer"
        GW[API Gateway]
        AUTH[Auth Service]
        RATE[Rate Limiter]
    end

    subgraph "Processing Layer"
        NOV[Novelty Engine]
        ROUTER[Request Router]
    end

    subgraph "Caching Layer"
        SCACHE[Semantic Cache<br/>FAISS/Milvus]
        RCACHE[Response Cache<br/>Redis]
    end

    subgraph "Inference Layer"
        SCHED[Priority Scheduler]
        OLLAMA[Ollama Pool]
    end

    subgraph "Storage Layer"
        PG[(PostgreSQL)]
        VDB[(Vector DB)]
        OBJ[(Object Store)]
    end

    subgraph "Observability"
        PROM[Prometheus]
        GRAF[Grafana]
    end

    UI --> WAF
    CLI --> WAF
    SDK --> WAF
    WAF --> LB
    LB --> GW
    GW --> AUTH
    GW --> RATE
    GW --> NOV
    GW --> ROUTER

    NOV --> VDB
    ROUTER --> SCACHE
    ROUTER --> RCACHE
    SCACHE -.cache miss.-> SCHED
    RCACHE -.cache miss.-> SCHED

    SCHED --> OLLAMA
    OLLAMA --> RCACHE

    GW --> PG
    NOV --> PG
    OLLAMA --> OBJ

    GW --> PROM
    NOV --> PROM
    PROM --> GRAF

    style NOV fill:#90EE90
    style SCACHE fill:#FFD700
    style RCACHE fill:#87CEEB
```

### Component Diagram

```mermaid
graph TB
    subgraph "Novelty Engine"
        EMB[Embedding Service]
        COMP[Novelty Compute]
        SCORE[Scoring Engine]
        IDX[Novelty Index]
    end

    subgraph "Cache System"
        SEM[Semantic Cache]
        RESP[Response Cache]
        META[Metadata Store]
    end

    subgraph "Inference Fabric"
        ORCH[Orchestrator]
        POOL[Model Pools]
        REG[Model Registry]
    end

    EMB --> COMP
    COMP --> SCORE
    SCORE --> IDX

    SEM --> META
    RESP --> META

    ORCH --> POOL
    POOL --> REG

    style EMB fill:#E6F3FF
    style SCORE fill:#FFE6E6
    style SEM fill:#E6FFE6
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/danindiana/GRYPHGEN.git
cd GRYPHGEN/calisota-ai/August_2025/novelty-llm-system

# Start all services
make docker-up

# The API will be available at http://localhost:8080
# Grafana dashboard at http://localhost:3000 (admin/admin)
```

### Test the API

```bash
# Health check
curl http://localhost:8080/health

# Submit a query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "model": "llama2",
    "temperature": 0.7,
    "max_tokens": 2048
  }'

# Check cache statistics
curl http://localhost:8080/cache/stats
```

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Build and start all services
make docker-build
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Option 2: Local Development

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev

# Run development server
make run-dev
```

### Option 3: From Source

```bash
# Install production dependencies
pip install -r requirements.txt

# Run the application
uvicorn src.api.gateway:create_app --factory --host 0.0.0.0 --port 8080
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8080
WORKERS=4

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=50

# PostgreSQL Configuration
POSTGRES_URL=postgresql://novelty:password@localhost:5432/novelty_db
POSTGRES_POOL_SIZE=20

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Novelty Engine
NOVELTY_MODEL=all-MiniLM-L6-v2
NOVELTY_DEVICE=cpu  # or cuda

# Caching
SEMANTIC_CACHE_TTL=3600
SEMANTIC_CACHE_THRESHOLD=0.85
RESPONSE_CACHE_TTL=3600

# Security
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

### Configuration Files

See [`config/`](config/) directory for YAML configuration files:

- `development.yaml` - Development settings
- `production.yaml` - Production settings
- `test.yaml` - Testing settings

## 📚 Usage

### Python SDK

```python
import httpx

# Initialize client
client = httpx.AsyncClient(base_url="http://localhost:8080")

# Query the LLM
response = await client.post("/query", json={
    "prompt": "Explain quantum computing",
    "model": "llama2",
    "temperature": 0.7,
    "max_tokens": 2048
})

data = response.json()
print(f"Response: {data['response']}")
print(f"Novelty Score: {data['novelty_score']}")
print(f"Cached: {data['cached']}")
```

### Novelty Engine API

```python
from src.novelty.engine import NoveltyEngine

# Initialize engine
engine = NoveltyEngine(
    model_name="all-MiniLM-L6-v2",
    k_neighbors=10,
    device="cpu"
)

# Process input
embedding, score = await engine.process(
    text="Your query here",
    user_id="user123",
    tenant_id="tenant456",
    store=True
)

print(f"Novelty Score: {score.score}")
print(f"Level: {score.level}")
print(f"Metrics: {score.metrics}")
```

### Cache Management

```python
from src.cache import SemanticCache, ResponseCache

# Semantic cache
semantic_cache = SemanticCache(
    similarity_threshold=0.85,
    ttl_seconds=3600
)

# Response cache
response_cache = ResponseCache(
    ttl_seconds=3600,
    max_size_mb=1000
)

# Get stats
stats = await response_cache.get_stats()
print(stats)
```

## 📖 API Documentation

### Endpoints

#### `POST /query`

Submit an LLM query with automatic novelty detection and caching.

**Request:**
```json
{
  "prompt": "string",
  "model": "string (optional)",
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false,
  "documents": [],
  "metadata": {}
}
```

**Response:**
```json
{
  "response": "string",
  "model": "string",
  "novelty_score": 0.85,
  "novelty_level": "high",
  "cached": false,
  "cache_hit_similarity": null,
  "tokens_used": 150,
  "processing_time_ms": 234.5,
  "timestamp": "2024-11-18T12:00:00Z"
}
```

#### `GET /health`

Health check endpoint.

#### `GET /cache/stats`

Get cache statistics.

#### `DELETE /cache/clear`

Clear all caches.

#### `GET /metrics`

Prometheus metrics endpoint.

### Interactive API Documentation

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Setup pre-commit hooks
pre-commit install

# Run tests
make test

# Run linters
make lint

# Format code
make format
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_novelty.py -v

# Run with coverage report
make test
```

### Code Quality

```bash
# Format code
make format

# Check formatting
make format-check

# Run linters
make lint

# Type checking
mypy src/
```

## 🚢 Deployment

### Docker Deployment

```bash
# Build images
make docker-build

# Start services
make docker-up

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
make docker-logs
```

### Kubernetes Deployment

See [`docs/kubernetes/`](docs/kubernetes/) for Kubernetes manifests:

```bash
kubectl apply -f docs/kubernetes/
```

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up SSL/TLS certificates
- [ ] Configure authentication and authorization
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategies
- [ ] Set resource limits and quotas
- [ ] Enable security scanning
- [ ] Configure log aggregation
- [ ] Set up CI/CD pipeline

## 📊 Monitoring

### Metrics

The system exposes Prometheus metrics at `/metrics`:

- `novelty_llm_requests_total` - Total number of requests
- `novelty_llm_request_duration_seconds` - Request duration histogram
- `novelty_llm_novelty_score` - Novelty score distribution
- `novelty_llm_cache_hits_total` - Cache hit counters

### Dashboards

Grafana dashboards are available at http://localhost:3000:

- **Overview Dashboard** - System health and metrics
- **Cache Performance** - Cache hit rates and efficiency
- **Novelty Metrics** - Novelty score distributions
- **Resource Usage** - CPU, memory, and storage metrics

### Logging

Structured JSON logging with multiple levels:

```bash
# View application logs
make docker-logs

# Tail specific service
docker-compose -f docker/docker-compose.yml logs -f novelty-llm-api
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters (`make ci-test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GRYPHGEN](https://github.com/danindiana/GRYPHGEN) - Parent project
- [Ollama](https://ollama.ai/) - LLM runtime
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Milvus](https://milvus.io/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embedding models

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/danindiana/GRYPHGEN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danindiana/GRYPHGEN/discussions)
- **Documentation**: [Full Documentation](docs/)

## 🗺️ Roadmap

- [ ] Multi-GPU support
- [ ] Advanced RAG capabilities
- [ ] Knowledge graph integration
- [ ] Fine-tuning support
- [ ] A/B testing framework
- [ ] Cost optimization features
- [ ] Enterprise SSO integration
- [ ] Advanced analytics

---

**Built with ❤️ by the GRYPHGEN team**
