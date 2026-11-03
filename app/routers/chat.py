"""
Chat Router
Handles user conversational messages, performs guardrail checks, semantic knowledge retrieval,
and connects to pretrained OpenAI models or local knowledge base fallback.
"""

from fastapi import APIRouter, HTTPException
from app.config import settings
from app.schemas import ChatRequest, ChatResponse, SourceReference
from app.services.guardrails_service import guardrails_service
from app.services.knowledge_service import knowledge_service
from app.services.openai_service import openai_service

router = APIRouter(prefix="/api", tags=["Chat"])


def build_local_clinic_response(matched_units: list, user_query: str) -> str:
    """Builds an informative, grounded answer about clinics/departments in local mode."""
    lines = []
    first = matched_units[0]
    dept_name = first.get("department", first.get("name"))

    lines.append(f"Here is the verified information for **{dept_name}**:")
    for u in matched_units:
        lines.append(f"\n- **{u['name']}**")
        lines.append(f"  • **Services:** {u['services']}")
        
        # Ensure emergency unit is never described as 'by appointment'
        if "emergency" in u['name'].lower():
            days_desc = "Immediate walk-in acute emergency care (verify 24/7 status with hospital)"
        else:
            days_desc = u['clinic_days']
            
        lines.append(f"  • **Clinic Days:** {days_desc}")
        lines.append(f"  • **Operating / Closing Hours:** {u['clinic_hours']}")

    if any("not confirmed" in u.get("clinic_hours", "").lower() for u in matched_units):
        lines.append("\n*(Note: Operating and closing hours are currently unconfirmed in hospital records. Please contact the hospital directly to verify exact start and closing times.)*")

    return "\n".join(lines)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # 1. Run Healthcare Safety Guardrails (Emergency, Diagnosis Disclaimer, Unlisted info)
    guardrail_resp, is_emergency, is_disclaimer = guardrails_service.evaluate(user_msg)
    if guardrail_resp:
        return ChatResponse(
            reply=guardrail_resp,
            model="safety-guardrail",
            source="guardrails",
            is_emergency=is_emergency,
            is_medical_disclaimer=is_disclaimer,
            sources=[]
        )

    # 2. Retrieve relevant context chunks using TF-IDF vector search
    retrieved_chunks = knowledge_service.search_chunks(user_msg, top_k=3)
    source_refs = [
        SourceReference(
            chunk_id=c.get("chunk_id"),
            title=c.get("title", ""),
            category=c.get("category", ""),
            relevance_score=c.get("relevance_score")
        )
        for c in retrieved_chunks
    ]

    # 3. If OpenAI API key is configured, use the pretrained LLM
    if settings.is_api_key_configured:
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        try:
            reply = openai_service.generate_reply(
                user_message=user_msg,
                conversation_history=history_dicts,
                retrieved_chunks=retrieved_chunks
            )
            return ChatResponse(
                reply=reply,
                model=settings.OPENAI_MODEL,
                source="openai_api",
                is_emergency=False,
                is_medical_disclaimer=False,
                sources=source_refs
            )
        except Exception as e:
            # Fallback on OpenAI API error to local knowledge base
            matched_faq = knowledge_service.match_faq(user_msg)
            if matched_faq:
                is_faq_disclaimer = (
                    matched_faq.get("category") == "safety_disclaimer"
                    or matched_faq.get("intent") == "medical_diagnosis_inquiry"
                )
                is_faq_emergency = (
                    matched_faq.get("category") == "emergency"
                    or matched_faq.get("intent") == "emergency_assistance"
                )
                return ChatResponse(
                    reply=f"{matched_faq['approved_answer']}\n\n*(Note: Generated via local FAQ fallback because OpenAI returned: {str(e)})*",
                    model="local-faq-fallback",
                    source="faq_fallback",
                    is_emergency=is_faq_emergency,
                    is_medical_disclaimer=is_faq_disclaimer,
                    sources=source_refs
                )
            raise HTTPException(status_code=502, detail=f"OpenAI service error: {str(e)}")

    # 4. Local Reference Mode (Grounded in hospital_service.json, no invented replies)
    # Check A: Exact or Curated FAQ Match
    matched_faq = knowledge_service.match_faq(user_msg)
    if matched_faq:
        is_faq_disclaimer = (
            matched_faq.get("category") == "safety_disclaimer"
            or matched_faq.get("intent") == "medical_diagnosis_inquiry"
        )
        is_faq_emergency = (
            matched_faq.get("category") == "emergency"
            or matched_faq.get("intent") == "emergency_assistance"
        )
        return ChatResponse(
            reply=matched_faq["approved_answer"],
            model="local-rule-engine",
            source="curated_faq",
            is_emergency=is_faq_emergency,
            is_medical_disclaimer=is_faq_disclaimer,
            sources=source_refs
        )

    # Check B: Department or Clinic Match (supports O&G, ANC, GOPD, Paediatrics, Dental, etc.)
    matched_units = knowledge_service.find_department_or_clinics(user_msg)
    if matched_units:
        reply_text = build_local_clinic_response(matched_units, user_msg)
        return ChatResponse(
            reply=reply_text,
            model="local-rule-engine",
            source="clinic_directory",
            sources=source_refs
        )

    # Check C: Strong TF-IDF Knowledge Retrieval Match
    # Require a strong similarity score (> 0.35) so irrelevant queries don't trigger random chunks
    if retrieved_chunks and retrieved_chunks[0].get("relevance_score", 0) > 0.35:
        top_chunk = retrieved_chunks[0]
        return ChatResponse(
            reply=f"**{top_chunk.get('title')}:**\n\n{top_chunk.get('content')}",
            model="local-tfidf-retriever",
            source="knowledge_chunks",
            sources=source_refs
        )

    # Approved Hospital Fallback (Section 7: Unknown, Unlisted or Unconfirmed Services/Programmes)
    profile = knowledge_service.get_hospital_profile()
    contacts = profile.get("contacts", {})
    phone = contacts.get("main_telephone", "07014678906")
    email = contacts.get("email", "eleriuseastsunsam@gmail.com")

    fallback_text = (
        "Information is unavailable from the hospital’s current approved records.\n\n"
        "“I’m unable to confirm that service or programme from the hospital’s current approved information. "
        "If a department, specialist, clinic day, or service is not included in our approved directory, "
        "the hospital treats it as unavailable.”\n\n"
        f"Please contact the hospital directly for confirmation or assistance:\n"
        f"- **Main Telephone:** {phone}\n"
        f"- **Email:** {email}"
    )

    return ChatResponse(
        reply=fallback_text,
        model="hospital-fallback",
        source="unconfirmed_information_policy",
        sources=source_refs
    )
