"""FastAPI application and gateway implementation."""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, make_asgi_app
import structlog

from .models import QueryRequest, QueryResponse, HealthResponse, ErrorResponse
from ..novelty.engine import NoveltyEngine
from ..cache.semantic import SemanticCache
from ..cache.response import ResponseCache

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter("novelty_llm_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_DURATION = Histogram(
    "novelty_llm_request_duration_seconds", "Request duration", ["endpoint"]
)
NOVELTY_SCORE = Histogram(
    "novelty_llm_novelty_score", "Novelty scores", buckets=[0.2, 0.4, 0.6, 0.8, 1.0]
)
CACHE_HIT = Counter(
    "novelty_llm_cache_hits_total", "Cache hits", ["cache_type"]  # semantic, response
)


class AppState:
    """Application state container."""

    def __init__(self):
        self.novelty_engine: Optional[NoveltyEngine] = None
        self.semantic_cache: Optional[SemanticCache] = None
        self.response_cache: Optional[ResponseCache] = None
        self.ollama_client = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    logger.info("Starting Novelty LLM System")

    # Initialize components
    app.state.app_state = AppState()

    # Initialize novelty engine
    app.state.app_state.novelty_engine = NoveltyEngine(
        model_name="all-MiniLM-L6-v2", device="cpu"
    )

    # Initialize caches
    app.state.app_state.semantic_cache = SemanticCache(similarity_threshold=0.85, ttl_seconds=3600)
    app.state.app_state.response_cache = ResponseCache(ttl_seconds=3600)

    logger.info("Novelty LLM System started successfully")

    yield

    logger.info("Shutting down Novelty LLM System")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Novelty LLM System",
        description="A scalable LLM platform with novelty scoring and semantic caching",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Routes
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components={
                "novelty_engine": "ok",
                "semantic_cache": "ok",
                "response_cache": "ok",
            },
        )

    @app.post("/query", response_model=QueryResponse, tags=["LLM"])
    async def query(
        request: QueryRequest, req: Request
    ) -> QueryResponse:
        """
        Process LLM query with novelty scoring and caching.

        Args:
            request: Query request
            req: FastAPI request object

        Returns:
            Query response with novelty score
        """
        start_time = time.time()
        app_state = req.app.state.app_state

        try:
            # Check response cache first (exact match)
            cache_result = await app_state.response_cache.get(
                text=request.prompt,
                model=request.model,
            )

            if hasattr(cache_result, "value"):
                # Cache hit
                CACHE_HIT.labels(cache_type="response").inc()
                REQUEST_COUNT.labels(endpoint="/query", status="success").inc()

                processing_time = (time.time() - start_time) * 1000

                return QueryResponse(
                    response=cache_result.value["response"],
                    model=cache_result.value.get("model", "cached"),
                    novelty_score=cache_result.value.get("novelty_score", 0.0),
                    novelty_level=cache_result.value.get("novelty_level", "unknown"),
                    cached=True,
                    cache_hit_similarity=1.0,
                    tokens_used=cache_result.value.get("tokens_used", 0),
                    processing_time_ms=processing_time,
                )

            # Compute novelty score
            embedding, novelty_score = await app_state.novelty_engine.process(
                text=request.prompt, store=True
            )

            # Record novelty score metric
            NOVELTY_SCORE.observe(novelty_score.score)

            # Check semantic cache
            semantic_result = await app_state.semantic_cache.get(
                embedding=embedding,
                text=request.prompt,
            )

            if hasattr(semantic_result, "value"):
                # Semantic cache hit
                CACHE_HIT.labels(cache_type="semantic").inc()
                REQUEST_COUNT.labels(endpoint="/query", status="success").inc()

                processing_time = (time.time() - start_time) * 1000

                response_text = semantic_result.value
                if isinstance(response_text, dict):
                    response_text = response_text.get("response", str(response_text))

                return QueryResponse(
                    response=response_text,
                    model=request.model or "cached-semantic",
                    novelty_score=novelty_score.score,
                    novelty_level=novelty_score.level.value,
                    cached=True,
                    cache_hit_similarity=semantic_result.similarity,
                    tokens_used=0,
                    processing_time_ms=processing_time,
                )

            # No cache hit - generate response
            # TODO: Integrate with Ollama for actual LLM inference
            response_text = f"[Placeholder response for: {request.prompt}]"
            tokens_used = len(response_text.split())

            # Cache the response
            await app_state.response_cache.set(
                text=request.prompt,
                response={
                    "response": response_text,
                    "model": request.model or "default",
                    "novelty_score": novelty_score.score,
                    "novelty_level": novelty_score.level.value,
                    "tokens_used": tokens_used,
                },
                model=request.model,
            )

            await app_state.semantic_cache.set(
                embedding=embedding,
                text=request.prompt,
                response=response_text,
            )

            REQUEST_COUNT.labels(endpoint="/query", status="success").inc()
            processing_time = (time.time() - start_time) * 1000
            REQUEST_DURATION.labels(endpoint="/query").observe(time.time() - start_time)

            return QueryResponse(
                response=response_text,
                model=request.model or "default",
                novelty_score=novelty_score.score,
                novelty_level=novelty_score.level.value,
                cached=False,
                cache_hit_similarity=None,
                tokens_used=tokens_used,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            REQUEST_COUNT.labels(endpoint="/query", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/cache/stats", tags=["Cache"])
    async def cache_stats(req: Request) -> dict:
        """Get cache statistics."""
        app_state = req.app.state.app_state

        semantic_stats = await app_state.semantic_cache.get_stats()
        response_stats = await app_state.response_cache.get_stats()

        return {
            "semantic_cache": semantic_stats,
            "response_cache": response_stats,
        }

    @app.delete("/cache/clear", tags=["Cache"])
    async def clear_cache(req: Request) -> dict:
        """Clear all caches."""
        app_state = req.app.state.app_state

        semantic_cleared = await app_state.semantic_cache.clear()
        response_cleared = await app_state.response_cache.clear()

        return {
            "semantic_cache_cleared": semantic_cleared,
            "response_cache_cleared": response_cleared,
        }

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
