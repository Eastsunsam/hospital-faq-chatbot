"""
Knowledge Service
Loads and indexes hospital_service.json, providing fast TF-IDF vector retrieval
and structured entity lookup for clinics, departments, and FAQs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from app.config import settings

# Common healthcare abbreviations mapped to canonical department / clinic names
ABBREVIATIONS = {
    "o&g": "Obstetrics and Gynaecology",
    "o & g": "Obstetrics and Gynaecology",
    "o and g": "Obstetrics and Gynaecology",
    "obgyn": "Obstetrics and Gynaecology",
    "ob/gyn": "Obstetrics and Gynaecology",
    "obstetrics": "Obstetrics and Gynaecology",
    "gynaecology": "Obstetrics and Gynaecology",
    "maternity": "Obstetrics and Gynaecology",
    "anc": "Antenatal Clinic",
    "pnc": "Postnatal Clinic",
    "gopd": "General Outpatient Clinic",
    "opd": "General Outpatient Clinic",
    "family medicine": "General Outpatient Clinic",
    "er": "Emergency Unit",
    "a&e": "Emergency Unit",
    "casualty": "Emergency Unit",
    "peds": "Paediatric Clinic",
    "pediatric": "Paediatric Clinic",
    "paediatric": "Paediatric Clinic",
    "children": "Paediatric Clinic",
    "kids": "Paediatric Clinic",
    "dental": "Dental Clinic",
    "teeth": "Dental Clinic",
    "dentist": "Dental Clinic",
    "surgery": "Surgical Clinic",
    "surgeon": "Surgical Clinic",
    "eye": "Eye Clinic",
    "ent": "ENT Clinic",
    "pharmacy": "Pharmacy",
    "chemist": "Pharmacy",
    "dispensary": "Pharmacy",
    "lab": "Laboratory Services",
    "laboratory": "Laboratory Services",
    "vaccine": "Immunization / Child Welfare Clinic",
    "immunization": "Immunization / Child Welfare Clinic",
    "immunisation": "Immunization / Child Welfare Clinic"
}


class KnowledgeService:
    def __init__(self, kb_path: Path = settings.KNOWLEDGE_BASE_PATH):
        self.kb_path = kb_path
        self._raw_data: Dict[str, Any] = {}
        self._chunks: List[Dict[str, Any]] = []
        self._vectorizer: Optional[Any] = None
        self._tfidf_matrix: Optional[Any] = None
        self.load()

    def load(self):
        """Loads and indexes the knowledge base from JSON."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

        self._chunks = self._raw_data.get("retrieval_knowledge_chunks", [])

        # Build TF-IDF search index over knowledge chunks if scikit-learn is available
        if HAS_SKLEARN and self._chunks:
            corpus = []
            for c in self._chunks:
                text_to_index = f"{c.get('title', '')} {' '.join(c.get('keywords', []))} {c.get('content', '')}"
                corpus.append(text_to_index)

            self._vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

    def search_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search across retrieval knowledge chunks.
        Returns top_k matching chunks with similarity scores.
        """
        if not query.strip() or not self._chunks:
            return []

        # Vector search using TF-IDF + Cosine Similarity
        if HAS_SKLEARN and self._vectorizer is not None and self._tfidf_matrix is not None:
            query_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
            top_indices = scores.argsort()[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0.05:
                    chunk = dict(self._chunks[idx])
                    chunk["relevance_score"] = round(score, 4)
                    results.append(chunk)
            return results

        # Fallback keyword matching
        q_tokens = set(query.lower().split())
        matched = []
        for c in self._chunks:
            c_text = f"{c.get('title', '')} {' '.join(c.get('keywords', []))} {c.get('content', '')}".lower()
            overlap = sum(1 for token in q_tokens if token in c_text)
            if overlap > 0:
                chunk = dict(c)
                chunk["relevance_score"] = float(overlap)
                matched.append(chunk)

        matched.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matched[:top_k]

    def get_hospital_profile(self) -> Dict[str, Any]:
        return self._raw_data.get("hospital_profile", {})

    def get_metadata(self) -> Dict[str, Any]:
        return self._raw_data.get("chatbot_metadata", {})

    def get_general_services(self) -> List[str]:
        return self._raw_data.get("general_services", [])

    def get_all_clinics(self) -> List[Dict[str, Any]]:
        return self._raw_data.get("departments_and_clinics", [])

    def find_department_or_clinics(self, query: str) -> List[Dict[str, Any]]:
        """
        Finds matching clinics or department units for a query,
        using strict word boundaries to avoid false positives (e.g. 'here' matching 'er').
        """
        q_clean = query.lower().strip()
        q_norm_id = q_clean.replace("-", "_").replace(" ", "_")
        matches = []

        # 1. Direct ID match first
        for unit in self.get_all_clinics():
            u_id = unit.get("id", "").lower()
            if u_id == q_norm_id or u_id == q_clean:
                matches.append(unit)
        if matches:
            return matches

        # 2. Check abbreviation table with strict word boundaries
        target_name = None
        for abbr, mapped in ABBREVIATIONS.items():
            pattern = r"\b" + re.escape(abbr) + r"\b"
            if re.search(pattern, q_clean):
                target_name = mapped
                break

        for unit in self.get_all_clinics():
            u_name = unit.get("name", "").lower()
            u_dept = unit.get("department", "").lower()
            u_id = unit.get("id", "").lower()

            if target_name:
                t_lower = target_name.lower()
                if t_lower in u_name or t_lower in u_dept or u_name in t_lower:
                    matches.append(unit)
                    continue

            # Word boundary matching for department, unit name, or ID
            if re.search(r"\b" + re.escape(u_name) + r"\b", q_clean):
                matches.append(unit)
                continue
            if re.search(r"\b" + re.escape(u_dept) + r"\b", q_clean):
                matches.append(unit)
                continue
            if re.search(r"\b" + re.escape(u_id.replace("_", " ")) + r"\b", q_clean) or u_id in q_clean:
                matches.append(unit)
                continue

            # Word boundary matching for aliases
            for alias in unit.get("aliases", []):
                alias_clean = alias.lower().strip()
                if not alias_clean:
                    continue
                if re.search(r"\b" + re.escape(alias_clean) + r"\b", q_clean):
                    matches.append(unit)
                    break

        return matches

    def get_clinic_by_identifier(self, query: str) -> Optional[Dict[str, Any]]:
        """Finds a single clinic by ID, name, or alias."""
        q_norm = query.lower().strip().replace("-", "_").replace(" ", "_")
        for unit in self.get_all_clinics():
            if unit.get("id", "").lower() == q_norm:
                return unit
        clinics = self.find_department_or_clinics(query)
        return clinics[0] if clinics else None

    def get_all_faqs(self) -> List[Dict[str, Any]]:
        return self._raw_data.get("faq_knowledge_base", [])

    def match_faq(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks if the query closely matches any curated FAQ."""
        q_lower = query.lower().strip()
        for faq in self.get_all_faqs():
            candidates = [faq.get("question", "").lower()] + [s.lower() for s in faq.get("sample_user_queries", [])]
            for cand in candidates:
                # Require significant substring or word overlap
                if len(cand) > 10 and (cand in q_lower or q_lower in cand):
                    return faq
        return None


knowledge_service = KnowledgeService()
