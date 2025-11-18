# Architecture Documentation

## System Overview

The Novelty LLM System is designed as a distributed, microservices-based platform that provides intelligent LLM query routing, novelty detection, and multi-tier caching.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Client Applications"
        WEB[Web UI]
        CLI[CLI Tools]
        SDK[SDK Libraries]
    end

    subgraph "Edge Layer"
        CDN[CDN/WAF]
        LB[Load Balancer]
    end

    subgraph "API Gateway Layer"
        GW[API Gateway<br/>FastAPI]
        AUTH[Authentication]
        RATELIM[Rate Limiter]
        POLICY[Policy Engine]
    end

    subgraph "Processing Layer"
        ROUTER[Request Router]
        NOV_ENGINE[Novelty Engine]
        EMB[Embedding Service]
    end

    subgraph "Caching Layer"
        SCACHE[Semantic Cache<br/>Vector Search]
        RCACHE[Response Cache<br/>Redis KV]
    end

    subgraph "Inference Layer"
        SCHED[Priority Scheduler]
        POOL1[Ollama Pool A<br/>GPU]
        POOL2[Ollama Pool B<br/>CPU]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL<br/>Metadata)]
        MILVUS[(Milvus<br/>Vectors)]
        REDIS[(Redis<br/>Cache)]
        S3[(S3/MinIO<br/>Objects)]
    end

    subgraph "Observability"
        PROM[Prometheus]
        GRAF[Grafana]
        JAEGER[Jaeger]
    end

    WEB --> CDN
    CLI --> CDN
    SDK --> CDN
    CDN --> LB
    LB --> GW

    GW --> AUTH
    GW --> RATELIM
    GW --> POLICY
    GW --> ROUTER

    ROUTER --> NOV_ENGINE
    NOV_ENGINE --> EMB
    EMB --> MILVUS

    ROUTER --> SCACHE
    ROUTER --> RCACHE

    SCACHE --> MILVUS
    RCACHE --> REDIS

    SCACHE -.miss.-> SCHED
    RCACHE -.miss.-> SCHED

    SCHED --> POOL1
    SCHED --> POOL2

    POOL1 --> RCACHE
    POOL2 --> RCACHE

    GW --> PG
    EMB --> S3

    GW --> PROM
    NOV_ENGINE --> PROM
    SCHED --> PROM
    PROM --> GRAF
    GW --> JAEGER

    style NOV_ENGINE fill:#90EE90
    style SCACHE fill:#FFD700
    style RCACHE fill:#87CEEB
    style SCHED fill:#FFA07A
```

## Request Flow Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Gateway as API Gateway
    participant Auth as Auth Service
    participant Router as Request Router
    participant Novelty as Novelty Engine
    participant SCache as Semantic Cache
    participant RCache as Response Cache
    participant Scheduler as Priority Scheduler
    participant Ollama as Ollama Pool
    participant DB as Database

    Client->>Gateway: POST /query
    Gateway->>Auth: Validate token
    Auth-->>Gateway: Valid (user_id, tenant_id)

    Gateway->>Router: Route request
    Router->>RCache: Check response cache (exact match)

    alt Response Cache Hit
        RCache-->>Client: Cached response (instant)
    else Response Cache Miss
        Router->>Novelty: Compute novelty score
        Novelty->>DB: Fetch embeddings & metadata
        Novelty-->>Router: NoveltyScore = 0.78

        Router->>SCache: Query semantic cache

        alt Semantic Cache Hit (similarity > 0.85)
            SCache-->>Client: Similar cached response
        else Semantic Cache Miss
            Router->>Scheduler: Enqueue (priority=novelty+tier)
            Scheduler->>Ollama: Dispatch to model pool
            Ollama-->>Scheduler: Generated response
            Scheduler->>RCache: Store in response cache
            Scheduler->>SCache: Store in semantic cache
            Scheduler->>DB: Log usage & metadata
            Scheduler-->>Client: Stream response
        end
    end
```

## Component Architecture

### 1. Novelty Engine

```mermaid
flowchart LR
    subgraph Novelty["Novelty Engine"]
        INPUT[Text Input]
        EMB[Embedding Generator<br/>Sentence Transformers]
        METRICS[Metrics Calculator]
        SCORER[Novelty Scorer]
        OUTPUT[Novelty Score]
    end

    subgraph Metrics["Scoring Metrics"]
        SEM[Semantic Distance<br/>35%]
        ENT[Entropy<br/>25%]
        RARE[Rarity<br/>20%]
        CLUST[Cluster Distance<br/>15%]
        TEMP[Temporal Decay<br/>5%]
    end

    INPUT --> EMB
    EMB --> METRICS
    METRICS --> SEM
    METRICS --> ENT
    METRICS --> RARE
    METRICS --> CLUST
    METRICS --> TEMP
    SEM --> SCORER
    ENT --> SCORER
    RARE --> SCORER
    CLUST --> SCORER
    TEMP --> SCORER
    SCORER --> OUTPUT

    style EMB fill:#E6F3FF
    style SCORER fill:#FFE6E6
```

### 2. Caching System

```mermaid
flowchart TB
    subgraph Caching["Multi-Tier Caching"]
        REQ[Query Request]
        L1[L1: Response Cache<br/>Redis KV Store]
        L2[L2: Semantic Cache<br/>Vector Similarity]
        MISS[Cache Miss]
        GEN[Generate Response]
    end

    REQ --> L1
    L1 -->|Hit: Exact Match| RETURN1[Return Cached]
    L1 -->|Miss| L2
    L2 -->|Hit: Similar >0.85| RETURN2[Return Similar]
    L2 -->|Miss| MISS
    MISS --> GEN
    GEN --> STORE[Store in Both Caches]
    STORE --> RETURN3[Return Generated]

    style L1 fill:#87CEEB
    style L2 fill:#FFD700
    style RETURN1 fill:#90EE90
    style RETURN2 fill:#90EE90
    style RETURN3 fill:#90EE90
```

### 3. Inference Orchestration

```mermaid
flowchart TB
    subgraph Orchestration["Inference Orchestration"]
        QUEUE[Priority Queue<br/>Kafka/Redis Streams]
        SCHED[Scheduler<br/>Novelty + Tier Based]

        subgraph Pools["Model Pools"]
            POOL_A[Pool A: GPU<br/>High Priority]
            POOL_B[Pool B: CPU<br/>Standard]
        end

        RESULT[Response Aggregator]
    end

    QUEUE --> SCHED
    SCHED -->|High Novelty<br/>Premium Tier| POOL_A
    SCHED -->|Standard| POOL_B
    POOL_A --> RESULT
    POOL_B --> RESULT

    style POOL_A fill:#FF6B6B
    style POOL_B fill:#4ECDC4
```

## Data Flow

### Embedding Storage

```mermaid
flowchart LR
    subgraph Storage["Vector Storage Pipeline"]
        TEXT[User Text]
        EMB[Generate<br/>Embedding]
        SCORE[Compute<br/>Novelty]

        subgraph Stores["Storage Backends"]
            MILVUS[(Milvus<br/>Vector Index)]
            PG[(PostgreSQL<br/>Metadata)]
            S3[(S3<br/>Documents)]
        end
    end

    TEXT --> EMB
    EMB --> SCORE
    SCORE --> MILVUS
    SCORE --> PG
    TEXT --> S3

    style MILVUS fill:#E6F3FF
    style PG fill:#FFE6F3
    style S3 fill:#E6FFE6
```

## Deployment Architecture

### Kubernetes Deployment

```mermaid
graph TB
    subgraph K8S["Kubernetes Cluster"]
        subgraph Ingress["Ingress Layer"]
            NGINX[NGINX Ingress]
        end

        subgraph App["Application Pods"]
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
        end

        subgraph Workers["Worker Pods"]
            WORKER1[Worker 1<br/>GPU]
            WORKER2[Worker 2<br/>GPU]
            WORKER3[Worker 3<br/>CPU]
        end

        subgraph Data["Data Layer"]
            REDIS_SVC[Redis Service]
            PG_SVC[PostgreSQL Service]
            MILVUS_SVC[Milvus Service]
        end

        subgraph Storage["Persistent Storage"]
            PVC1[PVC: Redis]
            PVC2[PVC: PostgreSQL]
            PVC3[PVC: Milvus]
        end
    end

    NGINX --> API1
    NGINX --> API2
    NGINX --> API3

    API1 --> WORKER1
    API2 --> WORKER2
    API3 --> WORKER3

    API1 --> REDIS_SVC
    API2 --> PG_SVC
    API3 --> MILVUS_SVC

    REDIS_SVC --> PVC1
    PG_SVC --> PVC2
    MILVUS_SVC --> PVC3

    style NGINX fill:#269
    style API1 fill:#2C5
    style API2 fill:#2C5
    style API3 fill:#2C5
```

## Security Architecture

```mermaid
flowchart TB
    subgraph Security["Security Layers"]
        CLIENT[Client Request]
        WAF[Web Application Firewall]
        TLS[TLS Termination]
        AUTH[Authentication<br/>JWT/OIDC]
        AUTHZ[Authorization<br/>RBAC]
        PII[PII Detection<br/>& Redaction]
        AUDIT[Audit Logging]
    end

    CLIENT --> WAF
    WAF --> TLS
    TLS --> AUTH
    AUTH --> AUTHZ
    AUTHZ --> PII
    PII --> AUDIT
    AUDIT --> APP[Application]

    style WAF fill:#FF6B6B
    style AUTH fill:#4ECDC4
    style PII fill:#FFD93D
```

## Scalability Patterns

### Horizontal Scaling

- **API Layer**: Stateless pods behind load balancer
- **Worker Pools**: Auto-scaling based on queue depth
- **Caching**: Redis cluster with sharding
- **Vector DB**: Milvus distributed deployment

### Vertical Scaling

- **GPU Workers**: Scale up for large models
- **Database**: Increase resources for high throughput

### Data Partitioning

- **Tenant Isolation**: Separate collections per tenant
- **Time-based Partitioning**: Archive old data
- **Geographic Distribution**: Multi-region deployment

## Performance Optimization

### Cache Hit Rate Optimization

1. **Semantic Similarity Threshold**: 0.85-0.90
2. **TTL Strategy**: Longer TTL for high-novelty queries
3. **Cache Warming**: Pre-populate with common queries

### Query Optimization

1. **Batch Processing**: Group similar queries
2. **Embedding Reuse**: Cache embeddings
3. **Index Optimization**: Regular FAISS index optimization

## Monitoring & Observability

### Key Metrics

- Request latency (p50, p95, p99)
- Cache hit rates (semantic, response)
- Novelty score distribution
- Model inference time
- Queue depth and wait time

### Dashboards

- System health overview
- Cache performance
- Novelty metrics
- Resource utilization

## Disaster Recovery

### Backup Strategy

- **Database**: Continuous WAL archiving
- **Vector Index**: Daily snapshots
- **Configuration**: Version controlled

### Recovery Procedures

1. Database point-in-time recovery
2. Vector index rebuild from backups
3. Cache warm-up from logs

## Future Enhancements

1. **Multi-Model Support**: A/B testing framework
2. **Fine-tuning Pipeline**: Custom model training
3. **Knowledge Graph**: Novelty-weighted entity graph
4. **Advanced RAG**: Document retrieval integration
5. **Cost Optimization**: Intelligent model routing
