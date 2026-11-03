"""
Docx to JSON Converter for Hospital FAQ Chatbot
Converts hospital_services_and_faq_reference.docx into a structured hospital_service.json
specifically optimized for chatbot systems (LLM prompts, intent matching, RAG/vector retrieval).
Uses standard library modules only (zipfile, xml.etree.ElementTree, json, argparse, re, os, sys, pathlib).
"""

import argparse
import json
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def clean_key(text: str) -> str:
    """Clean text into an idiomatic snake_case key."""
    cleaned = re.sub(r"[\s/\\:\-]+", "_", text.strip())
    cleaned = re.sub(r"[^\w]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def extract_docx_elements(docx_path: str):
    """
    Extracts paragraphs (with style) and tables from the docx file in sequential order.
    """
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    with zipfile.ZipFile(docx_path) as z:
        xml_content = z.read("word/document.xml")

    tree = ET.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = tree.find("w:body", ns)
    if body is None:
        raise ValueError("Invalid DOCX: 'w:body' not found.")

    val_key = f"{{{ns['w']}}}val"
    tc_tag = f"{{{ns['w']}}}tc"
    tr_tag = f"{{{ns['w']}}}tr"
    pstyle_tag = f".//{{{ns['w']}}}pStyle"

    elements = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p_style_elem = child.find(pstyle_tag)
            style = p_style_elem.attrib.get(val_key) if p_style_elem is not None else None
            text = "".join(child.itertext()).strip()
            if text:
                elements.append({"type": "paragraph", "style": style, "text": text})
        elif tag == "tbl":
            table_rows = []
            for row in child.findall(tr_tag):
                row_cells = []
                for cell in row.findall(tc_tag):
                    cell_text = " ".join(cell.itertext()).strip()
                    row_cells.append(cell_text)
                if any(row_cells):
                    table_rows.append(row_cells)
            if table_rows:
                elements.append({"type": "table", "rows": table_rows})

    return elements


def table_to_dicts(rows):
    """Convert table rows (with header in row 0) into list of dicts with clean keys."""
    if not rows or len(rows) < 2:
        return []
    raw_headers = rows[0]
    keys = [clean_key(h) or f"col_{idx}" for idx, h in enumerate(raw_headers)]

    result = []
    for r in rows[1:]:
        item = {}
        for idx, key in enumerate(keys):
            item[key] = r[idx] if idx < len(r) else ""
        result.append(item)
    return result


def build_chatbot_knowledge_base(docx_path: str) -> dict:
    """Parses DOCX and builds an enterprise-ready chatbot knowledge base."""
    elements = extract_docx_elements(docx_path)

    doc_info = {
        "title": elements[0]["text"] if len(elements) > 0 else "HOSPITAL SERVICES & FAQ REFERENCE",
        "subtitle": elements[1]["text"] if len(elements) > 1 else "",
        "disclaimer": elements[2]["text"] if len(elements) > 2 else ""
    }

    # Group elements by sections (Heading1)
    sections = []
    current_section = None
    for elem in elements[3:]:
        if elem["type"] == "paragraph" and elem.get("style") == "Heading1":
            if current_section:
                sections.append(current_section)
            current_section = {"heading": elem["text"], "elements": []}
        else:
            if current_section:
                current_section["elements"].append(elem)
    if current_section:
        sections.append(current_section)

    structured_sections = []
    general_services = []
    raw_departments = []
    clinic_days_table = []
    clinic_hours_table = []
    hospital_location = {}
    contact_information = {}
    unknown_info_section = {}
    sample_faqs_raw = []
    maintenance_guidelines = []
    approval_checklist = []

    for sec in sections:
        heading = sec["heading"]
        sec_elems = sec["elements"]
        sec_num_match = re.match(r"^(\d+)\.\s*(.*)$", heading)
        sec_num = int(sec_num_match.group(1)) if sec_num_match else None
        sec_title = sec_num_match.group(2) if sec_num_match else heading

        intro_text = []
        bullets = []
        tables = []

        for e in sec_elems:
            if e["type"] == "paragraph":
                if e.get("style") == "ListBullet":
                    bullets.append(e["text"])
                else:
                    intro_text.append(e["text"])
            elif e["type"] == "table":
                tables.append(e["rows"])

        structured_sections.append({
            "section_number": sec_num,
            "section_title": sec_title,
            "heading": heading,
            "description": " ".join(intro_text).strip(),
            "bullets": bullets,
            "tables": [table_to_dicts(t) for t in tables]
        })

        if sec_num == 1:
            general_services = bullets
        elif sec_num == 2 and tables:
            raw_departments = table_to_dicts(tables[0])
        elif sec_num == 3 and tables:
            clinic_days_table = table_to_dicts(tables[0])
        elif sec_num == 4 and tables:
            clinic_hours_table = table_to_dicts(tables[0])
        elif sec_num == 5:
            for b in bullets:
                if ":" in b:
                    k, v = b.split(":", 1)
                    hospital_location[clean_key(k)] = v.strip()
        elif sec_num == 6:
            for b in bullets:
                if ":" in b:
                    k, v = b.split(":", 1)
                    contact_information[clean_key(k)] = v.strip()
            if intro_text:
                contact_information["operational_note"] = " ".join(intro_text).strip()
        elif sec_num == 7:
            suggested_resp = ""
            policy_desc = []
            for t in intro_text:
                if "Suggested chatbot response:" in t:
                    continue
                if t.startswith("“") or t.startswith('"') or "unable to confirm" in t:
                    suggested_resp = t.strip()
                elif "Examples of information" not in t:
                    policy_desc.append(t)
            unknown_info_section = {
                "policy": " ".join(policy_desc).strip(),
                "suggested_chatbot_response": suggested_resp,
                "examples_of_unconfirmed_information": bullets
            }
        elif sec_num == 8 and tables:
            sample_faqs_raw = table_to_dicts(tables[0])
        elif sec_num == 9:
            maintenance_guidelines = bullets
        elif sec_num == 10:
            approval_checklist = bullets

    # 1. Departments and Clinics Unified Model
    dept_map = {}
    for d in raw_departments:
        dept_name = d.get("department_unit", d.get("department", "")).strip()
        services = d.get("sample_services", "").strip()
        dept_map[dept_name] = {
            "services": services
        }

    days_lookup = {c.get("clinic", "").strip(): c.get("clinic_day", "").strip() for c in clinic_days_table}
    hours_lookup = {c.get("unit", "").strip(): c.get("hours", "").strip() for c in clinic_hours_table}

    all_unit_names = []
    for c in clinic_days_table:
        name = c.get("clinic", "").strip()
        if name and name not in all_unit_names:
            all_unit_names.append(name)
    for c in clinic_hours_table:
        name = c.get("unit", "").strip()
        if name and name not in all_unit_names:
            all_unit_names.append(name)

    # Department associations and aliases
    aliases_and_depts = {
        "General Outpatient Clinic": {
            "department": "General Outpatient / Family Medicine",
            "aliases": ["gopd", "outpatient clinic", "general consultation", "family medicine clinic"],
            "services": "General consultations, assessment of common health concerns, follow-up visits, and referral when needed."
        },
        "Antenatal Clinic": {
            "department": "Obstetrics and Gynaecology",
            "aliases": ["anc", "prenatal clinic", "maternity clinic", "pregnancy care", "obstetrics clinic"],
            "services": "Pregnancy-related care, antenatal services, and maternal monitoring."
        },
        "Postnatal Clinic": {
            "department": "Obstetrics and Gynaecology",
            "aliases": ["pnc", "postpartum clinic", "post-delivery checkup", "new mother clinic"],
            "services": "Postnatal checkup and postpartum maternal care."
        },
        "Paediatric Clinic": {
            "department": "Paediatrics",
            "aliases": ["children clinic", "pediatrician", "infant care", "kids clinic"],
            "services": "Assessment and care of infants, children, and adolescents."
        },
        "Immunization / Child Welfare Clinic": {
            "department": "Immunization / Child Welfare",
            "aliases": ["vaccination clinic", "vaccine clinic", "baby clinic", "immunisation", "child welfare"],
            "services": "Vaccination and child-welfare services according to the approved schedule and stock availability."
        },
        "Dental Clinic": {
            "department": "Dental Unit",
            "aliases": ["dentist", "teeth", "dental care", "oral health clinic"],
            "services": "Dental assessment and procedures only if confirmed available by the hospital."
        },
        "Surgical Clinic": {
            "department": "Surgical Services",
            "aliases": ["surgery clinic", "surgeon consultation", "operation clinic"],
            "services": "Surgical consultations and pre/post-operative evaluation within hospital capacity."
        },
        "Medical / Specialist Clinic": {
            "department": "Medical / Specialist Clinics",
            "aliases": ["specialist clinic", "internal medicine", "physician clinic", "consultant clinic"],
            "services": "Specialist consultations only for specialties confirmed by the hospital."
        },
        "Family Planning Clinic": {
            "department": "Obstetrics and Gynaecology",
            "aliases": ["contraception clinic", "birth control", "reproductive health"],
            "services": "Family planning counselling and approved contraceptive services."
        },
        "Diabetes / Hypertension Clinic": {
            "department": "Medical / Specialist Clinics",
            "aliases": ["sugar clinic", "bp clinic", "hypertension clinic", "chronic care clinic", "diabetic clinic"],
            "services": "Specialized chronic disease monitoring, blood pressure and blood sugar management."
        },
        "Emergency Unit": {
            "department": "Emergency Services",
            "aliases": ["er", "a&e", "accident and emergency", "urgent care", "casualty"],
            "services": "Emergency assessment, stabilization, and immediate acute medical care."
        }
    }

    departments_and_clinics = []
    for unit_name in all_unit_names:
        meta = aliases_and_depts.get(unit_name, {
            "department": "Hospital Clinic Unit",
            "aliases": [unit_name.lower()],
            "services": "Clinical care and consultation."
        })
        if "emergency" in unit_name.lower():
            default_days = "Immediate walk-in emergency care (verify 24/7 operating status)"
        else:
            default_days = "By appointment / To be confirmed"
        clinic_days = days_lookup.get(unit_name, default_days)
        clinic_hours = hours_lookup.get(unit_name, "Not confirmed")
        days_confirmed = not ("example only" in clinic_days.lower() or "to be confirmed" in clinic_days.lower())
        hours_confirmed = not ("not confirmed" in clinic_hours.lower())

        summary_answer = (
            f"{unit_name} ({meta['department']}): "
            f"Provides {meta['services']} "
            f"Clinic schedule is {clinic_days}, and hours are currently {clinic_hours}."
        )

        departments_and_clinics.append({
            "id": clean_key(unit_name),
            "name": unit_name,
            "department": meta["department"],
            "aliases": meta["aliases"],
            "services": meta["services"],
            "clinic_days": clinic_days,
            "clinic_hours": clinic_hours,
            "schedule_status": {
                "days_confirmed": days_confirmed,
                "hours_confirmed": hours_confirmed,
                "requires_verification": not (days_confirmed and hours_confirmed)
            },
            "chatbot_summary": summary_answer
        })

    # Add any pure departments that did not have a dedicated clinic row
    clinic_dept_names = {u["department"] for u in departments_and_clinics}
    for dept_name, info in dept_map.items():
        if dept_name not in clinic_dept_names and not any(dept_name in u["name"] for u in departments_and_clinics):
            departments_and_clinics.append({
                "id": clean_key(dept_name),
                "name": dept_name,
                "department": dept_name,
                "aliases": [dept_name.lower()],
                "services": info["services"],
                "clinic_days": "Operating according to hospital schedule",
                "clinic_hours": "Operating during hospital business hours (to be confirmed)",
                "schedule_status": {
                    "days_confirmed": False,
                    "hours_confirmed": False,
                    "requires_verification": True
                },
                "chatbot_summary": f"{dept_name}: Provides {info['services']}"
            })

    # 2. Enriched FAQ Knowledge Base
    faq_enhancements = [
        {
            "id": "faq_hospital_services",
            "category": "services",
            "question": "What services does the hospital offer?",
            "sample_user_queries": [
                "What services do you offer?",
                "What medical services are available?",
                "What kind of treatments can I get here?",
                "Do you do lab tests and surgery?"
            ],
            "approved_answer": (
                "The hospital provides general outpatient consultations, emergency assessment and stabilization, "
                "inpatient ward care, maternal health (antenatal & postnatal care), child health and immunization, "
                "laboratory investigations, pharmacy dispensing, minor surgical procedures and wound care, "
                "health education, and specialist referrals."
            ),
            "answer_rule": "List only services present in the approved service directory.",
            "intent": "ask_services"
        },
        {
            "id": "faq_antenatal_clinic_days",
            "category": "clinic_schedule",
            "question": "Which days does the antenatal clinic hold?",
            "sample_user_queries": [
                "Which days does the antenatal clinic hold?",
                "When is antenatal clinic open?",
                "What days can pregnant women come for checkups?",
                "Is antenatal clinic open on Monday?"
            ],
            "approved_answer": (
                "The Antenatal Clinic operates on Tuesday and Thursday. "
                "(Please note: this is an example schedule for testing and should be confirmed with the hospital)."
            ),
            "answer_rule": "Return the verified antenatal clinic days. If not confirmed, say so.",
            "intent": "ask_clinic_days"
        },
        {
            "id": "faq_paediatric_clinic_hours",
            "category": "clinic_schedule",
            "question": "What time does the paediatric clinic start?",
            "sample_user_queries": [
                "What time does the paediatric clinic start?",
                "What are the opening hours for children clinic?",
                "When can I bring my child to the clinic?",
                "What time does pediatrics open?"
            ],
            "approved_answer": (
                "Operating hours for the Paediatric Clinic are currently not confirmed in our system. "
                "Please contact the hospital directly to confirm start and closing times."
            ),
            "answer_rule": "Return verified hours. If missing, direct the visitor to the hospital.",
            "intent": "ask_clinic_hours"
        },
        {
            "id": "faq_dental_unit_availability",
            "category": "departments",
            "question": "Does the hospital have a dental unit?",
            "sample_user_queries": [
                "Does the hospital have a dental unit?",
                "Do you have a dentist?",
                "Can I get dental treatment here?",
                "Is there a dental clinic available?"
            ],
            "approved_answer": (
                "Yes, the hospital directory includes a Dental Unit for dental assessment and procedures. "
                "The clinic day is listed as Friday (example schedule; subject to hospital confirmation)."
            ),
            "answer_rule": "Answer yes only if the unit is confirmed in the approved directory.",
            "intent": "check_service_availability"
        },
        {
            "id": "faq_hospital_location",
            "category": "location",
            "question": "Where is the hospital located?",
            "sample_user_queries": [
                "Where is the hospital located?",
                "What is your address?",
                "How do I get to the hospital?",
                "Where are you situated?",
                "Can you give me directions?"
            ],
            "approved_answer": (
                f"The hospital is located at {hospital_location.get('street_address', '[street address]')}, "
                f"{hospital_location.get('town_city', '[town/city]')}, {hospital_location.get('state', '[state]')}. "
                f"Landmarks & directions: {hospital_location.get('landmarks_directions', '[directions]')}. "
                f"Public transport & parking: {hospital_location.get('public_transport_parking_information', '[transport/parking]')}"
            ),
            "answer_rule": "Use the verified address and approved directions.",
            "intent": "ask_location"
        },
        {
            "id": "faq_contact_hospital",
            "category": "contact",
            "question": "How can I contact the hospital?",
            "sample_user_queries": [
                "How can I contact the hospital?",
                "What is your phone number?",
                "What is the hospital's email address?",
                "How do I reach the appointment desk?"
            ],
            "approved_answer": (
                f"You can contact the hospital via: "
                f"Main Telephone: {contact_information.get('main_telephone', '[phone number]')}; "
                f"Appointment Line: {contact_information.get('appointment_line', '[appointment line]')}; "
                f"Email: {contact_information.get('email', '[email]')}; "
                f"Website: {contact_information.get('website', '[website]')}. "
                f"For urgent medical emergencies, call {contact_information.get('emergency_contact', '[emergency contact]')}."
            ),
            "answer_rule": "Provide only verified contact details.",
            "intent": "ask_contact"
        },
        {
            "id": "faq_medical_diagnosis_disclaimer",
            "category": "safety_disclaimer",
            "question": "Can you tell me what illness I have?",
            "sample_user_queries": [
                "Can you tell me what illness I have?",
                "What is wrong with me?",
                "I have stomach pain and headache, what disease is it?",
                "Can you prescribe medication for my cough?",
                "Diagnose my symptoms"
            ],
            "approved_answer": (
                "I am an automated hospital assistant and cannot provide medical diagnoses, interpret symptoms, "
                "or prescribe treatments. Please consult a qualified doctor or healthcare professional at our hospital "
                "for personal medical evaluation."
            ),
            "answer_rule": "Explain that the FAQ assistant cannot diagnose and advise contacting a qualified clinician.",
            "intent": "medical_diagnosis_inquiry"
        },
        {
            "id": "faq_medical_emergency_guidance",
            "category": "emergency",
            "question": "I may be having a medical emergency. What should I do?",
            "sample_user_queries": [
                "I may be having a medical emergency. What should I do?",
                "I am having severe chest pain!",
                "Help, someone collapsed!",
                "Heavy bleeding, what do I do?",
                "Emergency!"
            ],
            "approved_answer": (
                "If you or someone nearby is experiencing a medical emergency, seek immediate emergency medical care. "
                f"Contact the hospital emergency line at {contact_information.get('emergency_contact', '[emergency contact]')} "
                "or proceed directly to the nearest Emergency Department. Do not rely on an online chatbot for emergency situations."
            ),
            "answer_rule": "Display the hospital-approved emergency instruction and verified contact details; do not attempt AI diagnosis.",
            "intent": "emergency_assistance"
        }
    ]

    # 3. Intent Specifications for Chatbot Engine
    intents = [
        {
            "intent_name": "emergency_assistance",
            "priority": "HIGH_SAFETY_CRITICAL",
            "description": "Triggered when a user expresses an urgent medical crisis or life-threatening situation.",
            "keywords": ["emergency", "urgent", "chest pain", "dying", "heart attack", "collapsed", "bleeding", "unconscious", "ambulance"],
            "action": "trigger_emergency_protocol",
            "response_text": (
                "If you or someone nearby is experiencing a medical emergency, please seek immediate emergency care. "
                f"Call our emergency contact at {contact_information.get('emergency_contact', '[emergency number]')} "
                "or proceed immediately to the nearest Emergency Department. Do not wait for chatbot replies."
            )
        },
        {
            "intent_name": "medical_diagnosis_inquiry",
            "priority": "HIGH_SAFETY_CRITICAL",
            "description": "Triggered when a user asks for medical diagnosis, symptom checks, or prescriptions.",
            "keywords": ["diagnose", "illness", "symptoms", "disease", "medicine to take", "prescription", "cure", "pain", "fever"],
            "action": "trigger_medical_disclaimer",
            "response_text": (
                "I cannot provide clinical diagnoses, interpret symptoms, or suggest medications. "
                "Please schedule an appointment or visit the General Outpatient Clinic to consult with a qualified physician."
            )
        },
        {
            "intent_name": "ask_services",
            "priority": "STANDARD",
            "description": "Triggered when a user asks what general services or facilities the hospital offers.",
            "keywords": ["services", "treatment", "facilities", "what do you treat", "care options"],
            "action": "list_approved_services",
            "response_text": (
                "The hospital offers the following approved services: " +
                ", ".join(general_services)
            )
        },
        {
            "intent_name": "ask_clinic_schedule",
            "priority": "STANDARD",
            "description": "Triggered when a user asks for clinic timetables, days, or hours.",
            "keywords": ["clinic days", "schedule", "when is clinic", "opening hours", "opening time", "timetable"],
            "action": "lookup_clinic_schedule",
            "response_text": "Please specify the clinic or department name (e.g. Antenatal, Paediatric, Dental, Surgical, Outpatient) to check its timetable."
        },
        {
            "intent_name": "ask_location",
            "priority": "STANDARD",
            "description": "Triggered when a user asks for address, directions, or parking.",
            "keywords": ["location", "address", "where are you", "directions", "landmarks", "parking", "bus", "transport"],
            "action": "provide_location_details",
            "response_text": (
                f"Address: {hospital_location.get('street_address', '')}, {hospital_location.get('town_city', '')}, {hospital_location.get('state', '')}. "
                f"Directions: {hospital_location.get('landmarks_directions', '')}."
            )
        },
        {
            "intent_name": "ask_contact",
            "priority": "STANDARD",
            "description": "Triggered when a user asks how to call, email, or reach the hospital.",
            "keywords": ["phone", "call", "telephone", "email", "contact", "appointment desk", "website", "social media"],
            "action": "provide_contact_channels",
            "response_text": (
                f"Main Phone: {contact_information.get('main_telephone', '')}; "
                f"Appointment Line: {contact_information.get('appointment_line', '')}; "
                f"Email: {contact_information.get('email', '')}; "
                f"Website: {contact_information.get('website', '')}."
            )
        },
        {
            "intent_name": "fallback_unconfirmed_information",
            "priority": "FALLBACK",
            "description": "Triggered when information is missing, unverified, contradictory, or unapproved.",
            "keywords": ["fees", "cost", "price", "doctor availability today", "bed availability", "insurance package"],
            "action": "fallback_safeguard",
            "response_text": unknown_info_section.get(
                "suggested_chatbot_response",
                "I’m unable to confirm that information from the hospital’s current approved information. Please contact the hospital directly for confirmation."
            )
        }
    ]

    # 4. Retrieval Knowledge Chunks (RAG / Semantic Vector Database format)
    retrieval_chunks = []
    chunk_id = 1

    # Chunk 1: Hospital Overview & General Services
    retrieval_chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "category": "services",
        "title": "Hospital Overview and General Services",
        "keywords": ["general services", "outpatient", "emergency", "inpatient", "maternal", "pharmacy", "lab"],
        "content": (
            "Hospital Overview: The hospital provides comprehensive healthcare services including: "
            + "; ".join(general_services)
        )
    })
    chunk_id += 1

    # Chunk 2..N: Each Department & Clinic
    for unit in departments_and_clinics:
        retrieval_chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "category": "clinic_timetable",
            "title": f"Clinic Timetable: {unit['name']}",
            "keywords": unit["aliases"] + [unit["name"].lower(), unit["department"].lower()],
            "content": (
                f"Unit: {unit['name']}. Department: {unit['department']}. "
                f"Scope of Services: {unit['services']}. "
                f"Clinic Days: {unit['clinic_days']}. "
                f"Operating Hours: {unit['clinic_hours']}. "
                f"Verification Note: Days confirmed: {unit['schedule_status']['days_confirmed']}, "
                f"Hours confirmed: {unit['schedule_status']['hours_confirmed']}."
            )
        })
        chunk_id += 1

    # Chunk: Location & Transport
    retrieval_chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "category": "location",
        "title": "Hospital Location and Directions",
        "keywords": ["location", "address", "landmarks", "directions", "parking", "public transport"],
        "content": (
            f"Hospital Name: {hospital_location.get('hospital_name', '[Name]')}. "
            f"Street Address: {hospital_location.get('street_address', '[Street]')}. "
            f"City/Town: {hospital_location.get('town_city', '[City]')}, State: {hospital_location.get('state', '[State]')}. "
            f"Directions and Landmarks: {hospital_location.get('landmarks_directions', '[Directions]')}. "
            f"Public Transport and Parking: {hospital_location.get('public_transport_parking_information', '[Parking/Transport]')}."
        )
    })
    chunk_id += 1

    # Chunk: Contacts
    retrieval_chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "category": "contact",
        "title": "Contact Details and Appointment Booking",
        "keywords": ["contact", "phone number", "emergency phone", "email", "appointment line", "website"],
        "content": (
            f"Main Telephone: {contact_information.get('main_telephone', '[Phone]')}. "
            f"Appointment Booking Line: {contact_information.get('appointment_line', '[Appointment Line]')}. "
            f"Emergency Contact: {contact_information.get('emergency_contact', '[Emergency Line]')}. "
            f"Official Email: {contact_information.get('email', '[Email]')}. "
            f"Official Website: {contact_information.get('website', '[Website]')}. "
            f"Social Media: {contact_information.get('official_social_media_pages', '[Social Media]')}."
        )
    })
    chunk_id += 1

    # Chunk: Safety & Medical Emergency Protocol
    retrieval_chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "category": "safety",
        "title": "Medical Emergency and Diagnosis Disclaimer Protocol",
        "keywords": ["emergency", "chest pain", "critical", "diagnose", "medical advice"],
        "content": (
            "Safety Protocol: The chatbot does not provide medical diagnoses or symptom assessments. "
            "In case of medical emergencies, visitors must immediately call emergency numbers or visit the emergency unit. "
            "The bot must never guess doctor availability, bed counts, or consultation prices."
        )
    })

    # Assemble complete schema
    chatbot_schema = {
        "chatbot_metadata": {
            "name": "Hospital FAQ Chatbot Knowledge Base",
            "version": "1.0.0",
            "source_document": os.path.basename(docx_path),
            "intended_use": "Conversational AI knowledge base for hospital website & messaging channels",
            "system_prompt": (
                "You are an official Hospital FAQ Assistant. Your role is to provide accurate, verified information "
                "about hospital services, clinic timetables, location, and contacts. Strictly follow these guardrails:\n"
                "1. NEVER give medical diagnoses, symptom assessments, or prescribe medications.\n"
                "2. In an emergency, immediately instruct the user to seek emergency care or call the verified emergency number.\n"
                "3. If any detail (hours, fees, doctor availability, appointment slots) is unverified or missing, "
                "state that you cannot confirm it from current approved records and refer them to official hospital contacts.\n"
                "4. Be compassionate, concise, and professional."
            ),
            "safety_guardrails": {
                "medical_diagnosis_rule": "The assistant cannot diagnose illnesses or advise on medications. Always refer to a qualified medical clinician.",
                "emergency_rule": "Direct users experiencing life-threatening symptoms to emergency services immediately.",
                "unconfirmed_information_policy": unknown_info_section.get("policy", ""),
                "approved_fallback_response": unknown_info_section.get(
                    "suggested_chatbot_response",
                    "I’m unable to confirm that information from the hospital’s current approved information. Please contact the hospital directly for confirmation."
                ),
                "unconfirmed_triggers": unknown_info_section.get("examples_of_unconfirmed_information", [])
            }
        },
        "hospital_profile": {
            "hospital_name": hospital_location.get("hospital_name", "[Enter approved hospital name]"),
            "location": hospital_location,
            "contacts": contact_information
        },
        "intents": intents,
        "faq_knowledge_base": faq_enhancements,
        "departments_and_clinics": departments_and_clinics,
        "general_services": general_services,
        "retrieval_knowledge_chunks": retrieval_chunks,
        "operational_governance": {
            "maintenance_rules": maintenance_guidelines,
            "pre_launch_approval_checklist": approval_checklist
        },
        "document_reference": {
            "preamble": doc_info,
            "sections": structured_sections
        }
    }

    return chatbot_schema


def main():
    parser = argparse.ArgumentParser(description="Convert Hospital Reference DOCX to a Chatbot-Optimized JSON.")
    parser.add_argument("input", nargs="?", default=None, help="Path to input .docx file")
    parser.add_argument("output", nargs="?", default=None, help="Path to output .json file")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    docx_path = args.input or str(base_dir / "hospital_services_and_faq_reference.docx")
    output_path = args.output or str(base_dir / "hospital_service.json")

    print(f"Reading DOCX from: {docx_path}")
    data = build_chatbot_knowledge_base(docx_path)

    print(f"Writing Chatbot JSON to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"\nSuccessfully structured JSON for chatbot!\n"
        f"File: '{os.path.basename(output_path)}'\n"
        f" - {len(data['intents'])} Chatbot Intents\n"
        f" - {len(data['faq_knowledge_base'])} Enriched FAQ Q&A Records\n"
        f" - {len(data['departments_and_clinics'])} Department & Clinic Entities (with aliases and schedule confirmation)\n"
        f" - {len(data['retrieval_knowledge_chunks'])} RAG / Vector Retrieval Chunks\n"
        f" - Safety Guardrails, System Prompt & Fallback Handlers configured\n"
    )


if __name__ == "__main__":
    main()
