"""RAG (Retrieval-Augmented Generation) endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.calisota.api.main import get_faiss_manager
from src.calisota.rag.faiss_manager import FAISSManager

router = APIRouter()


class AddDocumentsRequest(BaseModel):
    """Request to add documents to RAG."""
    texts: list[str] = Field(..., description="List of text documents to add")
    metadata: Optional[list[dict]] = Field(default=None, description="Optional metadata for each document")


class SearchRequest(BaseModel):
    """RAG search request."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")


@router.post("/add")
async def add_documents(
    request: AddDocumentsRequest,
    faiss: FAISSManager = Depends(get_faiss_manager)
) -> dict:
    """Add documents to the RAG system."""
    try:
        faiss.add_embeddings(request.texts, request.metadata)
        return {
            "success": True,
            "added_count": len(request.texts),
            "total_vectors": faiss.index.ntotal if faiss.index else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(
    request: SearchRequest,
    faiss: FAISSManager = Depends(get_faiss_manager)
) -> dict:
    """Search for similar documents in the RAG system."""
    try:
        results = faiss.search(
            request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_rag_stats(faiss: FAISSManager = Depends(get_faiss_manager)) -> dict:
    """Get RAG system statistics."""
    return faiss.get_stats()


@router.post("/save")
async def save_index(faiss: FAISSManager = Depends(get_faiss_manager)) -> dict:
    """Save FAISS index to disk."""
    try:
        faiss.save_index()
        return {"success": True, "message": "Index saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
