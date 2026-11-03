"""
Data Schemas and Pydantic Models for Hospital FAQ Chatbot
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or input message")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Past conversation messages for context")


class SourceReference(BaseModel):
    chunk_id: Optional[str] = None
    title: str
    category: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    source: str
    is_emergency: bool = False
    is_medical_disclaimer: bool = False
    sources: Optional[List[SourceReference]] = []


class ClinicItem(BaseModel):
    id: str
    name: str
    department: str
    aliases: List[str] = []
    services: str
    clinic_days: str
    clinic_hours: str
    requires_verification: bool = True
    chatbot_summary: Optional[str] = None


class DepartmentItem(BaseModel):
    id: str
    name: str
    department: str
    services: str


class FAQItem(BaseModel):
    id: str
    category: str
    question: str
    approved_answer: str
    answer_rule: str
    intent: Optional[str] = None
    sample_user_queries: List[str] = []


class HospitalProfile(BaseModel):
    hospital_name: str
    location: Dict[str, Any]
    contacts: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    openai_model: str
    openai_sdk_installed: bool
    api_key_configured: bool
    knowledge_base_loaded: bool
    departments_count: int
    clinics_count: int
    faq_count: int
    indexed_chunks_count: int
