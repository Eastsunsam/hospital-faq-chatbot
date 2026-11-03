"""
OpenAI Integration Service
Handles communication with pretrained OpenAI models (e.g. gpt-4o-mini).
Integrates grounded knowledge retrieval and conversation history.
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

from app.config import settings
from app.services.knowledge_service import knowledge_service


class OpenAIService:
    def __init__(self):
        self.model = settings.OPENAI_MODEL

    def _build_grounded_system_prompt(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a grounded system instruction with safety rules and relevant context."""
        meta = knowledge_service.get_metadata()
        profile = knowledge_service.get_hospital_profile()
        loc = profile.get("location", {})
        contacts = profile.get("contacts", {})
        guardrails = meta.get("safety_guardrails", {})

        prompt_lines = [
            "You are the official Hospital FAQ Assistant.",
            "Your role is to answer questions about hospital services, clinics, hours, location, and contacts.",
            "",
            "=== CRITICAL HEALTHCARE SAFETY RULES ===",
            "1. NO MEDICAL DIAGNOSIS: Never diagnose illnesses, interpret symptoms, or prescribe medications. Always direct personal health inquiries to a qualified medical doctor.",
            "2. EMERGENCY PROTOCOL: In life-threatening emergencies, immediately urge the user to seek emergency care or call the hospital emergency number.",
            f"   - Emergency Line: {contacts.get('emergency_contact', 'Call emergency services immediately.')}",
            "3. ACCURACY & UNCONFIRMED INFO: Ground your answers strictly in the verified hospital facts below. If information (such as fees, doctor availability, or exact hours) is missing or marked 'Not confirmed', state clearly that it is not confirmed and refer them to official contacts.",
            f"   - Standard Fallback: \"{guardrails.get('approved_fallback_response', 'I am unable to confirm that information from the hospital current approved information. Please contact the hospital directly.')}\"",
            "",
            "=== BASIC HOSPITAL INFORMATION ===",
            f"Name: {profile.get('hospital_name', 'Hospital')}",
            f"Address: {loc.get('street_address', '')}, {loc.get('town_city', '')}, {loc.get('state', '')}",
            f"Directions: {loc.get('landmarks_directions', '')}",
            f"Transport/Parking: {loc.get('public_transport_parking_information', '')}",
            f"Main Phone: {contacts.get('main_telephone', '')}",
            f"Appointment Line: {contacts.get('appointment_line', '')}",
            f"Email: {contacts.get('email', '')}",
            f"Website: {contacts.get('website', '')}",
            "",
            "=== RELEVANT HOSPITAL KNOWLEDGE BASE EXCERPTS ==="
        ]

        if relevant_chunks:
            for c in relevant_chunks:
                prompt_lines.append(f"[{c.get('title', 'Knowledge Chunk')}]:")
                prompt_lines.append(c.get("content", ""))
                prompt_lines.append("")
        else:
            # Fallback to general services overview if no specific chunks matched
            prompt_lines.append("General Services Overview:")
            for s in knowledge_service.get_general_services()[:8]:
                prompt_lines.append(f"- {s}")
            prompt_lines.append("")

        prompt_lines.extend([
            "=== RESPONSE STYLE ===",
            "- Concise, compassionate, polite, and helpful.",
            "- Use clean markdown bullet points for lists.",
            "- Never hallucinate hospital policies, names, or contact numbers."
        ])

        return "\n".join(prompt_lines)

    def generate_reply(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Calls the OpenAI API with grounded prompt and conversation history."""
        if not settings.is_api_key_configured:
            raise ValueError("OpenAI API key is not configured in .env.")

        system_prompt = self._build_grounded_system_prompt(retrieved_chunks or [])
        messages = [{"role": "system", "content": system_prompt}]

        # Include past history (last 6 messages)
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        # Call via official SDK if available
        if HAS_OPENAI_SDK:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            return resp.choices[0].message.content

        # Fallback via direct HTTP
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_data = e.read().decode("utf-8")
            try:
                msg = json.loads(err_data).get("error", {}).get("message", err_data)
            except Exception:
                msg = err_data
            raise RuntimeError(f"OpenAI API Error ({e.code}): {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI API: {str(e)}")


openai_service = OpenAIService()
