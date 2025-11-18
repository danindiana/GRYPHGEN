# Novelty LLM System Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A production-ready, scalable LLM platform with intelligent novelty detection, semantic caching, and multi-tenant support.

## 📚 Full Implementation Available

This architecture has been **fully implemented** with production-ready code! See the complete system in:

📂 **[novelty-llm-system/](novelty-llm-system/)**

### 🚀 Quick Start

```bash
cd novelty-llm-system
make docker-up
# API available at http://localhost:8080
# Grafana at http://localhost:3000
```

### 📖 Documentation

- **[README](novelty-llm-system/README.md)** - Features, installation, and usage guide
- **[Architecture](novelty-llm-system/docs/ARCHITECTURE.md)** - Detailed system design and data flows
- **[Deployment](novelty-llm-system/docs/DEPLOYMENT.md)** - Docker, Kubernetes, production setup
- **[API Reference](novelty-llm-system/docs/API.md)** - REST API documentation
- **[Tests](novelty-llm-system/tests/)** - Comprehensive test suite

### ✨ Key Features

- ✅ Multi-metric novelty scoring (semantic distance, entropy, rarity, clustering, temporal)
- ✅ Two-tier caching (semantic + response) with Redis and Milvus
- ✅ FastAPI REST API with async/await
- ✅ Docker Compose deployment with monitoring stack
- ✅ Prometheus metrics + Grafana dashboards
- ✅ Ollama integration for LLM inference
- ✅ Latest dependencies (Python 3.11+, all packages updated to Nov 2024)
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Comprehensive test coverage
- ✅ Production-ready configuration

## System Architecture Diagram

```mermaid
flowchart LR
  %% ==== CLIENT & EDGE ====
  subgraph EDGE["Edge & Identity"]
    CDN[CDN / WAF / TLS Termination]
    OIDC[OIDC / SSO<br/>AuthN]
    RATEL[Global Rate Limiter<br/>per-IP, per-User, per-Tier]
  end

  %% ==== FRONTEND ====
  subgraph FE["Frontend"]
    OWUI[OpenWebUI<br/>multi-tenant]
    FE_API[Frontend API<br/>BFF layer]
  end

  %% ==== CONTROL PLANE ====
  subgraph CP["Control Plane"]
    POL[Policy Engine<br/>RBAC, Data-use, PII rules]
    BILL[Billing & Quotas<br/>Stripe/alt, usage metering]
    ADMIN[Admin Console<br/>Triage, Abuse, Keys]
    OBS[Observability<br/>Prometheus + Logs + Traces]
  end

  %% ==== DATA PLANE ====
  subgraph DP["Data Plane"]
    API[Gateway / gRPC+HTTP]
    Q[Queue / Stream<br/>Kafka or Redis Streams]
    ROUTER[Request Router<br/>tier, model, A/B]
    subgraph NOV["Novelty Engine"]
      EMB[Embedding Service<br/>CPU/GPU]
      NCOMP[Novelty Compute<br/>semantic dist, entropy, rarity]
      NIDX[Novelty Index<br/>Scores, Histograms]
    end
    subgraph CACHE["Caches"]
      SCACHE[Semantic Cache<br/>FAISS/Milvus]
      RCACHE[Response Cache<br/>KV: Redis/Memcached]
    end
    subgraph STO["Storage"]
      PG[Postgres/Timescale<br/>Metadata & Usage]
      OBJ[Object Storage<br/>Docs/Uploads]
      IDX[Vector DB<br/>FAISS/Milvus]
    end
    subgraph ETL["Ingestion & ETL"]
      PARSE[Doc Parser/OCR]
      CLN[PII Redaction / Anonymizer]
      CAN[Canary & Quality Checks]
    end
  end

  %% ==== INFERENCE FABRIC ====
  subgraph INF["Inference Fabric Bare-metal Ubuntu"]
    ORCH[Model Orchestrator<br/>per-tenant pools]
    OLLAMA1[Ollama Runtime Pool A<br/>GPU nodes]
    OLLAMA2[Ollama Runtime Pool B<br/>CPU/GPU burst]
    REG[Model Registry<br/>model cards, eval tags]
    SCH[Job Scheduler<br/>FIFO+Priority by novelty/tier]
  end

  %% ==== DATA PRODUCTS ====
  subgraph PROD["Data Products & Enterprise"]
    KGRAPH[Novelty-Weighted Knowledge Graph]
    DATAKIT[Curated Datasets<br/>Anonymized]
    ENT[Enterprise Tenants<br/>VPC / BYO-Key / Private Index]
  end

  %% ==== FLOWS ====
  CDN --> OWUI
  OWUI --> FE_API --> API
  API --> RATEL
  API --> OIDC
  API --> POL
  API --> ROUTER

  ROUTER --> NOV
  ROUTER --> CACHE
  ROUTER --> ETL
  ROUTER --> ORCH

  %% Novelty path
  FE_API -- user msg/docs --> API
  NOV --> EMB
  EMB --> IDX
  NOV --> NIDX
  NOV --> SCACHE

  %% Cache decision
  ROUTER -- "check semantic/response cache" --> SCACHE
  SCACHE -- "HIT =>" --> RCACHE
  SCACHE -- "MISS =>" --> SCH

  %% Inference
  SCH --> ORCH
  ORCH --> OLLAMA1
  ORCH --> OLLAMA2
  OLLAMA1 --> RCACHE
  OLLAMA2 --> RCACHE

  %% Storage and telemetry
  ETL --> CLN --> OBJ
  ETL --> PARSE --> IDX
  API --> PG
  RCACHE --> FE_API
  FE_API --> OWUI

  %% Control plane connections
  OBS --- API
  OBS --- ORCH
  OBS --- OLLAMA1
  OBS --- OLLAMA2
  BILL --- API
  ADMIN --- CP

  %% Productization
  PG --> KGRAPH
  IDX --> KGRAPH
  OBJ --> DATAKIT
  KGRAPH --> ENT
  DATAKIT --> ENT
```
