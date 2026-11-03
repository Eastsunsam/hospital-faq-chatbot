"""
Guardrails Service
Enforces critical healthcare safety protocols:
1. Immediate Medical Emergency detection & escalation.
2. Medical Diagnosis & Prescription prohibition disclaimers.
3. Unconfirmed or unlisted information fallback validation (Section 7).
"""

import re
from typing import Optional, Tuple
from app.services.knowledge_service import knowledge_service


class GuardrailsService:
    # High-risk emergency keywords and patterns
    EMERGENCY_PATTERNS = [
        r"\bemergency\b",
        r"\bchest pain\b",
        r"\bheart attack\b",
        r"\bsevere bleeding\b",
        r"\bheavy bleeding\b",
        r"\bcan'?t breathe\b",
        r"\bdifficulty breathing\b",
        r"\bshortness of breath\b",
        r"\bunconscious\b",
        r"\bcollapsed\b",
        r"\bpassed out\b",
        r"\bstroke\b",
        r"\bseizure\b",
        r"\bconvulsion\b",
        r"\bdying\b",
        r"\bchoking\b",
        r"\bsevere burn\b",
        r"\bambulance\b",
        r"\bpoisoning\b"
    ]

    # Medical diagnosis inquiry triggers
    DIAGNOSIS_PATTERNS = [
        r"\bdiagnos(e|is|ing)\b",
        r"\bwhat illness\b",
        r"\bwhat disease\b",
        r"\bwhat is wrong with me\b",
        r"\bwhat medicine\b",
        r"\bprescrib(e|tion)\b",
        r"\bwhat drug\b",
        r"\bcheck my symptoms\b",
        r"\bi have symptoms\b",
        r"\bcure for my\b",
        r"\btell me what illness\b"
    ]

    # Questions asking for things explicitly marked unconfirmed or unlisted in hospital guidelines (Section 7)
    UNCONFIRMED_PATTERNS = [
        r"\bnot listed\b",
        r"\bdepartment not listed\b",
        r"\bclinic not listed\b",
        r"\bunlisted\b",
        r"\bnot in (the )?(directory|records|schedule|list)\b",
        r"\bmissing (department|clinic|service)\b",
        r"\bunavailable\b",
        r"\bhow much (is|are|does|costs?)\b",
        r"\bfees?\b",
        r"\bprices?\b",
        r"\bcost of\b",
        r"\bbed availability\b",
        r"\bis dr\.?\s+[a-z]+ (available|in today)\b",
        r"\bwhich doctor is on duty\b"
    ]

    @classmethod
    def check_emergency(cls, user_message: str) -> Optional[str]:
        """Returns emergency response if life-threatening query is detected."""
        text = user_message.lower()
        for pattern in cls.EMERGENCY_PATTERNS:
            if re.search(pattern, text):
                contacts = knowledge_service.get_hospital_profile().get("contacts", {})
                emergency_num = contacts.get("emergency_contact", "Call local emergency services (e.g. 911 / 112)")
                return (
                    "🚨 **MEDICAL EMERGENCY ALERT:**\n\n"
                    "If you or someone nearby is experiencing a medical emergency, please seek immediate emergency medical care.\n\n"
                    f"- **Hospital Emergency Contact:** {emergency_num}\n"
                    "- Please proceed immediately to the nearest hospital Emergency Department.\n"
                    "- **Important:** Do not wait for online chatbot replies during a medical crisis."
                )
        return None

    @classmethod
    def check_medical_disclaimer(cls, user_message: str) -> Optional[str]:
        """Returns disclaimer if user asks for clinical diagnosis or prescription."""
        text = user_message.lower()
        for pattern in cls.DIAGNOSIS_PATTERNS:
            if re.search(pattern, text):
                return (
                    "⚠️ **Medical Advice Disclaimer:**\n\n"
                    "I am an automated hospital reference assistant and cannot provide medical diagnoses, "
                    "interpret clinical symptoms, or prescribe medications.\n\n"
                    "Please visit our **General Outpatient / Family Medicine Clinic** or schedule an appointment "
                    "with a qualified doctor for personalized medical evaluation."
                )
        return None

    @classmethod
    def check_unconfirmed_request(cls, user_message: str) -> Optional[str]:
        """
        Returns approved hospital unknown response if unverified operational data
        or unlisted departments/clinics are requested (Section 7).
        """
        text = user_message.lower()
        for pattern in cls.UNCONFIRMED_PATTERNS:
            if re.search(pattern, text):
                contacts = knowledge_service.get_hospital_profile().get("contacts", {})
                phone = contacts.get("main_telephone", "07014678906")
                email = contacts.get("email", "eleriuseastsunsam@gmail.com")
                return (
                    "Information is unavailable from the hospital’s current approved directory.\n\n"
                    "“I’m unable to confirm that information from the hospital’s current approved information. "
                    "If a clinic, department, or service is not listed in the approved schedule, it must be treated as unconfirmed.”\n\n"
                    f"Please contact the hospital directly for verified information:\n"
                    f"- **Main Telephone:** {phone}\n"
                    f"- **Email:** {email}"
                )
        return None

    @classmethod
    def evaluate(cls, user_message: str) -> Tuple[Optional[str], bool, bool]:
        """
        Evaluates input against all guardrails.
        Returns: (response_text, is_emergency, is_medical_disclaimer)
        """
        # 1. Emergency is top priority
        emergency_resp = cls.check_emergency(user_message)
        if emergency_resp:
            return emergency_resp, True, False

        # 2. Medical diagnosis disclaimer
        disclaimer_resp = cls.check_medical_disclaimer(user_message)
        if disclaimer_resp:
            return disclaimer_resp, False, True

        # 3. Unconfirmed or unlisted department info check (Section 7)
        unconfirmed_resp = cls.check_unconfirmed_request(user_message)
        if unconfirmed_resp:
            return unconfirmed_resp, False, False

        return None, False, False


guardrails_service = GuardrailsService()
