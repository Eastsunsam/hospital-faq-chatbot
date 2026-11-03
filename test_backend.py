"""
Comprehensive Backend Test Suite for Hospital FAQ Chatbot
Tests services, guardrails, TF-IDF vector retrieval, and FastAPI routes.
Runs synchronously and quickly.
"""

import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from app.schemas import ChatRequest
from app.services.knowledge_service import knowledge_service, HAS_SKLEARN
from app.services.guardrails_service import guardrails_service
from app.routers.chat import chat
from app.routers.hospital import (
    get_general_services,
    get_departments,
    get_clinics,
    get_clinic_by_id,
    get_faqs,
    get_location,
    get_contacts,
    get_hospital_info
)
from app.routers.system import health, stats


def run_tests():
    passed = 0
    failed = 0

    def assert_true(condition, test_name):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {test_name}", flush=True)
            passed += 1
        else:
            print(f"  [FAIL] {test_name}", flush=True)
            failed += 1

    print("\n" + "=" * 60, flush=True)
    print(" 🏥 STARTING HOSPITAL FAQ CHATBOT BACKEND TESTS", flush=True)
    print("=" * 60, flush=True)

    # -------------------------------------------------------------
    # 1. KNOWLEDGE SERVICE & VECTOR RETRIEVAL TESTS
    # -------------------------------------------------------------
    print("\n--- 1. Testing Knowledge Service & Vector Search ---", flush=True)
    assert_true(len(knowledge_service._chunks) > 0, f"Indexed {len(knowledge_service._chunks)} knowledge chunks")
    assert_true(HAS_SKLEARN, "Scikit-Learn is active for TF-IDF search")

    # Search tests
    results = knowledge_service.search_chunks("antenatal clinic schedule days", top_k=2)
    assert_true(len(results) > 0, "Semantic search returns relevant chunks")
    assert_true(any("antenatal" in r["title"].lower() for r in results), "Antenatal chunk correctly retrieved")

    results_emergency = knowledge_service.search_chunks("emergency care unit", top_k=2)
    assert_true(len(results_emergency) > 0, "Emergency chunk retrieved")

    # Entity lookups
    anc = knowledge_service.get_clinic_by_identifier("antenatal_clinic")
    assert_true(anc is not None and anc["name"] == "Antenatal Clinic", "Clinic lookup by ID: Antenatal Clinic")

    anc_alias = knowledge_service.get_clinic_by_identifier("maternity clinic")
    assert_true(anc_alias is not None and anc_alias["name"] == "Antenatal Clinic", "Clinic lookup by alias: 'maternity clinic'")

    gopd_alias = knowledge_service.get_clinic_by_identifier("gopd")
    assert_true(gopd_alias is not None and "General Outpatient" in gopd_alias["name"], "Clinic lookup by alias: 'gopd'")

    # -------------------------------------------------------------
    # 2. HEALTHCARE SAFETY GUARDRAILS TESTS
    # -------------------------------------------------------------
    print("\n--- 2. Testing Safety Guardrails (Zero Tolerance) ---", flush=True)
    
    # Emergency tests
    resp, is_emerg, is_disc = guardrails_service.evaluate("Help me, I am having severe chest pain!")
    assert_true(is_emerg is True, "Emergency detected for 'severe chest pain'")
    assert_true("EMERGENCY ALERT" in resp, "Emergency warning returned")

    resp2, is_emerg2, _ = guardrails_service.evaluate("Someone collapsed and is unconscious!")
    assert_true(is_emerg2 is True, "Emergency detected for 'collapsed and is unconscious'")

    # Medical diagnosis disclaimer tests
    resp_diag, is_emerg_d, is_disc_d = guardrails_service.evaluate("Can you diagnose what disease I have based on my symptoms?")
    assert_true(is_disc_d is True, "Medical diagnosis disclaimer triggered")
    assert_true("cannot provide medical diagnoses" in resp_diag.lower(), "Medical disclaimer contains proper warning")

    resp_rx, _, is_disc_rx = guardrails_service.evaluate("What medicine should I take for cough?")
    assert_true(is_disc_rx is True, "Prescription / medication inquiry triggered disclaimer")

    # Unconfirmed information tests
    resp_fee, _, _ = guardrails_service.evaluate("How much is the consultation fee?")
    assert_true(resp_fee is not None and "unable to confirm" in resp_fee.lower(), "Unconfirmed fees query triggers approved fallback")

    # Normal inquiry should NOT trigger guardrails
    resp_normal, is_e_norm, is_d_norm = guardrails_service.evaluate("What days is the antenatal clinic open?")
    assert_true(resp_normal is None and is_e_norm is False and is_d_norm is False, "Normal query passes safety guardrails cleanly")

    # -------------------------------------------------------------
    # 3. HOSPITAL OPERATIONS REST API ENDPOINTS
    # -------------------------------------------------------------
    print("\n--- 3. Testing Hospital Data Endpoints ---", flush=True)
    
    services = get_general_services()
    assert_true(len(services) >= 10, f"Retrieved {len(services)} general services")

    departments = get_departments()
    assert_true(len(departments) >= 8, f"Retrieved {len(departments)} departments")

    clinics = get_clinics()
    assert_true(len(clinics) >= 11, f"Retrieved {len(clinics)} clinic timetables")

    single_clinic = get_clinic_by_id("antenatal_clinic")
    assert_true(single_clinic.name == "Antenatal Clinic", "Single clinic endpoint: Antenatal Clinic retrieved")

    faqs = get_faqs()
    assert_true(len(faqs) == 8, f"Retrieved {len(faqs)} curated FAQs")

    location = get_location()
    assert_true("street_address" in location, "Location endpoint returns street address")

    contacts = get_contacts()
    assert_true("emergency_contact" in contacts and "main_telephone" in contacts, "Contacts endpoint returns verified contact fields")

    info = get_hospital_info()
    assert_true("hospital_profile" in info and "departments_and_clinics" in info, "Hospital info endpoint returns complete overview")

    # -------------------------------------------------------------
    # 4. CHAT ROUTER ENDPOINT (POST /api/chat)
    # -------------------------------------------------------------
    print("\n--- 4. Testing Chat Endpoint (POST /api/chat) ---", flush=True)

    # Test A: General Services
    chat_req_a = ChatRequest(message="What services does the hospital offer?")
    chat_resp_a = chat(chat_req_a)
    assert_true("outpatient consultations" in chat_resp_a.reply.lower(), "Chat responds accurately to general services inquiry")
    assert_true(chat_resp_a.is_emergency is False, "General inquiry has is_emergency=False")

    # Test B: Clinic Schedule
    chat_req_b = ChatRequest(message="Which days does the antenatal clinic hold?")
    chat_resp_b = chat(chat_req_b)
    assert_true("tuesday and thursday" in chat_resp_b.reply.lower(), "Chat returns verified Antenatal Clinic days (Tuesday & Thursday)")

    # Test C: Clinic Hours
    chat_req_c = ChatRequest(message="What time does the paediatric clinic start?")
    chat_resp_c = chat(chat_req_c)
    assert_true("not confirmed" in chat_resp_c.reply.lower(), "Chat correctly states unconfirmed paediatric hours per guidelines")

    # Test D: Emergency Trigger
    chat_req_d = ChatRequest(message="I am having severe chest pain, help!")
    chat_resp_d = chat(chat_req_d)
    assert_true(chat_resp_d.is_emergency is True, "Emergency inquiry flags is_emergency=True")
    assert_true("EMERGENCY ALERT" in chat_resp_d.reply, "Emergency inquiry returns critical alert text")

    # Test E: Medical Diagnosis Query
    chat_req_e = ChatRequest(message="Can you tell me what illness I have?")
    chat_resp_e = chat(chat_req_e)
    assert_true(chat_resp_e.is_medical_disclaimer is True, "Medical diagnosis flags is_medical_disclaimer=True")

    # -------------------------------------------------------------
    # 5. SYSTEM & DIAGNOSTICS ENDPOINTS
    # -------------------------------------------------------------
    print("\n--- 5. Testing System Health & Diagnostics ---", flush=True)
    sys_health = health()
    assert_true(sys_health.status == "online", "Health check status: 'online'")
    assert_true(sys_health.knowledge_base_loaded is True, "Health check: knowledge base loaded")
    assert_true(sys_health.clinics_count >= 11, f"Health check: {sys_health.clinics_count} clinics loaded")

    sys_stats = stats()
    assert_true(sys_stats["status"] == "healthy", "System stats status: 'healthy'")
    assert_true("TF-IDF" in sys_stats["vector_search_engine"], "Vector search engine: TF-IDF active")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print(f" TEST RUN COMPLETE: {passed} PASSED, {failed} FAILED", flush=True)
    print("=" * 60 + "\n", flush=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
