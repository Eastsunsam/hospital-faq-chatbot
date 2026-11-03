"""
System Router
Health checks and diagnostic statistics for the backend.
"""

from fastapi import APIRouter
from app.config import settings
from app.schemas import HealthResponse
from app.services.knowledge_service import knowledge_service, HAS_SKLEARN
from app.services.openai_service import HAS_OPENAI_SDK

router = APIRouter(prefix="/api", tags=["System & Diagnostics"])


@router.get("/health", response_model=HealthResponse)
def health():
    """System health check and configuration status."""
    return HealthResponse(
        status="online",
        openai_model=settings.OPENAI_MODEL,
        openai_sdk_installed=HAS_OPENAI_SDK,
        api_key_configured=settings.is_api_key_configured,
        knowledge_base_loaded=True,
        departments_count=len(knowledge_service.get_all_clinics()),
        clinics_count=len(knowledge_service.get_all_clinics()),
        faq_count=len(knowledge_service.get_all_faqs()),
        indexed_chunks_count=len(knowledge_service._chunks)
    )


@router.get("/stats")
def stats():
    """Detailed backend operational diagnostics."""
    return {
        "status": "healthy",
        "vector_search_engine": "scikit-learn TF-IDF + Cosine Similarity" if HAS_SKLEARN else "keyword-frequency",
        "indexed_chunks": len(knowledge_service._chunks),
        "total_faqs": len(knowledge_service.get_all_faqs()),
        "total_clinics": len(knowledge_service.get_all_clinics()),
        "model_configured": settings.OPENAI_MODEL,
        "is_api_key_set": settings.is_api_key_configured
    }
