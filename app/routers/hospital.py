"""
Hospital Data Router
Exposes REST endpoints for hospital services, clinics, departments, location, and contacts.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from app.schemas import ClinicItem, DepartmentItem, FAQItem
from app.services.knowledge_service import knowledge_service

router = APIRouter(prefix="/api", tags=["Hospital Operations"])


@router.get("/services", response_model=List[str])
def get_general_services():
    """Returns the approved general hospital services list."""
    return knowledge_service.get_general_services()


@router.get("/departments", response_model=List[DepartmentItem])
def get_departments():
    """Returns all hospital departments and their sample service scopes."""
    clinics = knowledge_service.get_all_clinics()
    departments = []
    seen = set()
    for c in clinics:
        dept_name = c.get("department")
        if dept_name and dept_name not in seen:
            seen.add(dept_name)
            departments.append(DepartmentItem(
                id=c.get("id", "").replace("_clinic", ""),
                name=c.get("name", dept_name),
                department=dept_name,
                services=c.get("services", "")
            ))
    return departments


@router.get("/clinics", response_model=List[ClinicItem])
def get_clinics():
    """Returns all clinic timetables including days, hours, and verification status."""
    raw = knowledge_service.get_all_clinics()
    results = []
    for c in raw:
        results.append(ClinicItem(
            id=c.get("id", ""),
            name=c.get("name", ""),
            department=c.get("department", ""),
            aliases=c.get("aliases", []),
            services=c.get("services", ""),
            clinic_days=c.get("clinic_days", ""),
            clinic_hours=c.get("clinic_hours", ""),
            requires_verification=c.get("schedule_status", {}).get("requires_verification", True),
            chatbot_summary=c.get("chatbot_summary")
        ))
    return results


@router.get("/clinics/{clinic_id}", response_model=ClinicItem)
def get_clinic_by_id(clinic_id: str):
    """Retrieves details of a specific clinic by ID or name."""
    clinic = knowledge_service.get_clinic_by_identifier(clinic_id)
    if not clinic:
        raise HTTPException(status_code=404, detail=f"Clinic '{clinic_id}' not found.")
    return ClinicItem(
        id=clinic.get("id", ""),
        name=clinic.get("name", ""),
        department=clinic.get("department", ""),
        aliases=clinic.get("aliases", []),
        services=clinic.get("services", ""),
        clinic_days=clinic.get("clinic_days", ""),
        clinic_hours=clinic.get("clinic_hours", ""),
        requires_verification=clinic.get("schedule_status", {}).get("requires_verification", True),
        chatbot_summary=clinic.get("chatbot_summary")
    )


@router.get("/faqs", response_model=List[FAQItem])
def get_faqs():
    """Returns the curated library of hospital FAQs and approved answer rules."""
    raw = knowledge_service.get_all_faqs()
    return [
        FAQItem(
            id=f.get("id", ""),
            category=f.get("category", ""),
            question=f.get("question", ""),
            approved_answer=f.get("approved_answer", ""),
            answer_rule=f.get("answer_rule", ""),
            intent=f.get("intent"),
            sample_user_queries=f.get("sample_user_queries", [])
        )
        for f in raw
    ]


@router.get("/location")
def get_location():
    """Returns official hospital address, directions, landmarks, and transit details."""
    profile = knowledge_service.get_hospital_profile()
    return profile.get("location", {})


@router.get("/contacts")
def get_contacts():
    """Returns verified phone numbers, appointment line, emergency line, email, and website."""
    profile = knowledge_service.get_hospital_profile()
    return profile.get("contacts", {})


@router.get("/info")
def get_hospital_info() -> Dict[str, Any]:
    """Returns high-level hospital overview, services, and clinic directory."""
    profile = knowledge_service.get_hospital_profile()
    return {
        "hospital_profile": profile,
        "general_services": knowledge_service.get_general_services(),
        "departments_and_clinics": [
            {
                "name": c.get("name"),
                "department": c.get("department"),
                "clinic_days": c.get("clinic_days"),
                "clinic_hours": c.get("clinic_hours"),
                "requires_verification": c.get("schedule_status", {}).get("requires_verification", True)
            }
            for c in knowledge_service.get_all_clinics()
        ],
        "faqs_count": len(knowledge_service.get_all_faqs())
    }
