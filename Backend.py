import asyncio
import json
import logging
import os
import re
import uuid
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import io
from fastapi import UploadFile
import PyPDF2
import docx
import base64
from xai_sdk import Client
from xai_sdk.chat import user, assistant, system
from xai_sdk.tools import web_search, x_search

# ==================== LOGGING & ENV ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()
GROK_API_KEY   = os.getenv("GROK_API_KEY")
BASE_CHAT_URL  = os.getenv("BASE_CHAT_URL", "https://juristmind.onrender.com")

# ==================== MODEL CONSTANTS ====================
THINKING_MODEL = "grok-4-1-fast"   # Call 1: case researcher (agentic search + reasoning)
FAST_MODEL     = "grok-3-fast"     # Greetings / predefined answers only
WRITING_MODEL  = "grok-4-1-fast"   # Call 2: answer writer

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: No timeout on Call 1. Case research runs to completion. It is better
# to wait and get the right case than to cut it short and serve a weaker answer.
# Slowness is handled by streaming status messages to the user ("Still searching...")
# ─────────────────────────────────────────────────────────────────────────────


# ==================== LEGAL AREA MAP ====================
LEGAL_AREA_MAP = {
    # Constitutional / Fundamental Rights
    "police": "fundamental rights Nigeria",
    "arrest": "fundamental rights Nigeria",
    "detention": "fundamental rights Nigeria",
    "bail": "fundamental rights bail Nigeria",
    "torture": "fundamental rights Nigeria",
    "harassment": "fundamental rights Nigeria",
    "brutality": "fundamental rights Nigeria",
    "right": "fundamental rights Nigeria",
    "constitution": "constitutional law Nigeria",
    "freedom": "fundamental rights Nigeria",
    # Criminal
    "murder": "homicide criminal law Nigeria",
    "manslaughter": "homicide criminal law Nigeria",
    "theft": "theft stealing criminal law Nigeria",
    "robbery": "armed robbery criminal law Nigeria",
    "rape": "sexual assault criminal law Nigeria",
    "assault": "assault battery criminal law Nigeria",
    "fraud": "fraud criminal law Nigeria",
    "forgery": "forgery criminal law Nigeria",
    "corruption": "corruption criminal law Nigeria",
    # Contract
    "contract": "contract law Nigeria",
    "agreement": "contract law Nigeria",
    "offer": "contract law Nigeria",
    "acceptance": "contract law Nigeria",
    "consideration": "contract law Nigeria",
    "breach": "contract breach Nigeria",
    "misrepresentation": "misrepresentation contract Nigeria",
    # Tort
    "negligence": "negligence tort law Nigeria",
    "defamation": "defamation libel slander Nigeria",
    "nuisance": "nuisance tort law Nigeria",
    "trespass": "trespass tort law Nigeria",
    "damages": "damages tort law Nigeria",
    # Land / Property
    "land": "land law property Nigeria",
    "property": "property law Nigeria",
    "tenancy": "landlord tenant Nigeria",
    "rent": "landlord tenant Nigeria",
    "mortgage": "mortgage property law Nigeria",
    # Family
    "marriage": "family law marriage Nigeria",
    "divorce": "divorce matrimonial causes Nigeria",
    "custody": "child custody family law Nigeria",
    "inheritance": "succession inheritance Nigeria",
    "will": "succession testamentary Nigeria",
    # Employment
    "employ": "employment labour law Nigeria",
    "dismiss": "wrongful dismissal employment Nigeria",
    "termination": "wrongful dismissal employment Nigeria",
    "salary": "employment labour law Nigeria",
    "worker": "employment labour law Nigeria",
    # Company
    "company": "company law Nigeria",
    "director": "company law directors Nigeria",
    "shareholder": "company law shareholders Nigeria",
    "insolvency": "insolvency bankruptcy Nigeria",
    # Election
    "election": "election law Nigeria",
    "tribunal": "election tribunal Nigeria",
    "governorship": "election governorship Nigeria",
    # Evidence
    "evidence": "evidence law Nigeria",
    "confession": "confessional statement evidence Nigeria",
    "witness": "witness evidence law Nigeria",
    "hearsay": "hearsay evidence Nigeria",
}

def detect_legal_area(query: str) -> str:
    """Return the best-matching legal area label for the query."""
    q_lower = query.lower()
    for keyword, area in LEGAL_AREA_MAP.items():
        if keyword in q_lower:
            return area
    return "Nigerian law"


# ==================== QUESTION PARSER ====================
def parse_question_structure(query: str) -> Dict:
    """
    Detect whether this query has numbered or lettered sub-parts.
    Also extract sub-issues for parallel case research only when sub-parts
    cover genuinely different legal areas.
    """
    patterns = {
        'main_numbers':       r'\b(?:question\s+)?(\d+)[.:\)]',
        'sub_letters':        r'\b(\d+)\s*\(([a-z])\)',
        'sub_roman':          r'\b(\d+)\s*\(([ivx]+)\)',
        'standalone_letters': r'^\s*([a-z])\)',
        'standalone_roman':   r'^\s*([ivx]+)\)',
    }
    question_parts     = []
    has_multiple_parts = False

    main_nums = re.findall(patterns['main_numbers'], query, re.IGNORECASE)
    if main_nums:
        question_parts.extend([f"Question {n}" for n in main_nums])

    sub_letters = re.findall(patterns['sub_letters'], query)
    if sub_letters:
        question_parts.extend([f"Question {num}({letter})" for num, letter in sub_letters])
        has_multiple_parts = True

    sub_roman = re.findall(patterns['sub_roman'], query)
    if sub_roman:
        question_parts.extend([f"Question {num}({roman})" for num, roman in sub_roman])
        has_multiple_parts = True

    if re.search(patterns['standalone_letters'], query, re.MULTILINE):
        has_multiple_parts = True

    sub_issues = _extract_sub_issues(query)

    return {
        'has_parts':                    has_multiple_parts or len(question_parts) > 1,
        'parts':                        question_parts if question_parts else ['Single Question'],
        'requires_structured_response': has_multiple_parts,
        'sub_issues':                   sub_issues,
    }


def _extract_sub_issues(query: str) -> List[Tuple[str, str]]:
    """
    Split a multi-part scenario into individual sub-questions so each can get
    its own parallel case research — BUT only when sub-parts genuinely cover
    DIFFERENT legal areas. If all sub-parts are about the same area (e.g. the
    elements of one tort), a single search is enough and this returns [].
    """
    pattern = r'(?:^|\n)\s*(?:\(([a-z]|[ivx]+)\)|([a-z]|[ivx]+)\))\s*'
    splits  = re.split(pattern, query, flags=re.MULTILINE | re.IGNORECASE)

    if len(splits) <= 1:
        return []

    sub_issues = []
    i = 1
    while i < len(splits):
        label = splits[i] or (splits[i + 1] if i + 1 < len(splits) else None)
        text  = splits[i + 2] if i + 2 < len(splits) else ""
        if label and text and text.strip():
            sub_issues.append((f"({label})", text.strip()))
        i += 3

    if len(sub_issues) < 2:
        return sub_issues

    # Only run parallel searches if sub-parts span genuinely different legal areas
    areas = {detect_legal_area(text) for _, text in sub_issues}
    if len(areas) == 1:
        return []   # Same area — single case search is sufficient

    return sub_issues


# ==================== DRAFT DETAIL CHECKER ====================
DRAFT_REQUIRED_FIELDS = {
    "tenancy":           ["Landlord full name", "Tenant full name", "Property address",
                          "Monthly/annual rent amount (₦)", "Duration of tenancy"],
    "employment":        ["Employer name/company", "Employee full name", "Job title",
                          "Monthly salary (₦)", "Start date", "Place of work"],
    "sale":              ["Seller full name", "Buyer full name",
                          "Description of item/property being sold", "Agreed price (₦)"],
    "loan":              ["Lender name", "Borrower name", "Loan amount (₦)",
                          "Repayment schedule", "Interest rate (if any)"],
    "partnership":       ["Names of all partners", "Business name",
                          "Profit and loss sharing ratio", "Duration of partnership"],
    "mou":               ["Names of all parties", "Purpose of the MOU",
                          "Obligations of each party"],
    "affidavit":         ["Deponent full name", "Facts to be deposed",
                          "Purpose / court or body it is for"],
    "power of attorney": ["Donor full name", "Attorney full name",
                          "Scope of authority granted"],
    "will":              ["Testator full name", "Names of beneficiaries",
                          "Assets to be distributed", "Executor name"],
    "default":           ["Names of all parties involved",
                          "Subject matter of the document", "Key terms and conditions"],
}

DRAFT_TYPE_KEYWORDS = {
    "tenancy":           ["tenancy", "lease", "rent agreement"],
    "employment":        ["employment", "appointment letter", "work contract"],
    "sale":              ["sale", "purchase", "sale of goods", "bill of sale"],
    "loan":              ["loan", "lending", "credit agreement"],
    "partnership":       ["partnership", "business partnership"],
    "mou":               ["mou", "memorandum of understanding"],
    "affidavit":         ["affidavit", "sworn statement"],
    "power of attorney": ["power of attorney", "poa"],
    "will":              ["will", "last will", "testament"],
}

def detect_draft_type(query: str) -> str:
    q_lower = query.lower()
    for dtype, keywords in DRAFT_TYPE_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return dtype
    return "default"

def get_required_fields(draft_type: str) -> List[str]:
    return DRAFT_REQUIRED_FIELDS.get(draft_type, DRAFT_REQUIRED_FIELDS["default"])

def draft_has_sufficient_details(query: str, draft_type: str) -> bool:
    """
    Return True if the query already contains enough detail to draft immediately.
    Heuristic: >60 words AND at least 2 named entities OR a monetary amount present.
    """
    word_count = len(query.split())
    has_names  = len(re.findall(r'\b[A-Z][a-z]+\b', query)) >= 2
    has_amount = bool(re.search(
        r'₦|NGN|\d[\d,]*\s*(naira|kobo|months?|years?|k\b)', query, re.IGNORECASE))
    return word_count > 60 and (has_names or has_amount)

def build_draft_details_request(draft_type: str) -> str:
    fields   = get_required_fields(draft_type)
    fields_list = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(fields))
    doc_name    = draft_type.replace("_", " ").title() if draft_type != "default" else "document"
    return (
        f"I'd be happy to draft that {doc_name} for you. "
        f"To produce an accurate and complete draft, please provide the following details:\n\n"
        f"{fields_list}\n\n"
        f"Once you share these, I'll prepare the full draft immediately."
    )


# ==================== DETECTORS ====================
def is_problem_question(query: str) -> bool:
    q_lower = query.lower()
    scenario_keywords = [
        'advise', 'what are the rights', 'can he', 'can she', 'can they',
        'what remedies', 'what action', 'sue', 'claim', 'liable',
        'what are his rights', 'what should', 'is entitled to'
    ]
    has_character_names = bool(re.search(
        r'\b(ade|bola|chidi|emeka|fatima|kunle|ngozi|yemi|tunde|amina)\b', q_lower))
    has_narrative = bool(re.search(
        r'\b(entered into|agreed|signed|purchased|sold|was|were)\b', q_lower))
    has_scenario_phrase = any(kw in q_lower for kw in scenario_keywords)
    has_facts = 'facts:' in q_lower or 'scenario:' in q_lower
    return (has_scenario_phrase or has_facts or
            (has_character_names and has_narrative) or
            ('advise' in q_lower and len(query.split()) > 20))

def is_law_school_question(query: str) -> bool:
    """
    Distinguish a law school / academic exam question from a real-person legal problem.
    Law school questions: constructed scenario with multiple parties, "advise the parties",
    "discuss", "critically analyse", or explicit academic/exam signals.
    """
    q_lower = query.lower()
    academic_signals = [
        'law school', 'faculty of law', 'university', 'examine', 'discuss',
        'critically analyse', 'critically analyze', 'with reference to',
        'in the light of', 'advise all parties', 'advise the parties',
        'what offences', 'what offence', 'liability of', 'examine the legal',
        'using decided cases', 'with the aid of', 'question 1', 'question 2',
        'question 3', 'answer all questions', 'answer any',
    ]
    named_chars = len(re.findall(
        r'\b(ade|bola|chidi|emeka|fatima|kunle|ngozi|yemi|tunde|amina|john|mary|peter|paul)\b',
        q_lower))
    has_question_number = bool(re.search(r'\b(question\s+\d+|q\s*\d+)\b', q_lower))
    has_academic_signal = any(sig in q_lower for sig in academic_signals)
    # 3+ named characters in a scenario = classic law school problem
    return has_academic_signal or has_question_number or named_chars >= 3

def is_greeting(query: str) -> bool:
    q_lower = query.lower().strip()
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy|sup|yo)\b',
        r'^(what\'s up|whats up|how are you|how\'s it going|hows it going)\b',
        r'^(nice to meet you|pleasure to meet you)\b',
    ]
    for pattern in greeting_patterns:
        if re.search(pattern, q_lower):
            return True
    if len(query.split()) <= 3 and any(
            w in q_lower for w in ['hi', 'hello', 'hey', 'morning', 'afternoon', 'evening']):
        return True
    return False

def generate_greeting_response(query: str) -> str:
    q_lower = query.lower()
    if 'morning' in q_lower:
        return "Good morning! I'm JuristMind, your AI legal assistant. How can I help you with Nigerian law today?"
    elif 'afternoon' in q_lower:
        return "Good afternoon! I'm JuristMind, here to assist with your legal questions. What can I help you with?"
    elif 'evening' in q_lower:
        return "Good evening! I'm JuristMind, ready to help with Nigerian legal matters. What's your question?"
    elif any(p in q_lower for p in ["how are you", "how's it going", "what's up"]):
        return "I'm doing well, thank you! I'm JuristMind, your AI legal assistant. How can I assist you today?"
    return "Hello! I'm JuristMind, your AI legal assistant specialising in Nigerian law. What can I help you with today?"

def is_continuation_request(query: str) -> bool:
    q_lower = query.lower().strip()
    continuation_phrases = ['continue', 'go on', 'proceed', 'keep going', 'more', 'next']
    return any(phrase == q_lower or q_lower.startswith(phrase) for phrase in continuation_phrases)

def is_definitional_query(query: str) -> bool:
    """
    Simple "what is X", "define X", "meaning of X" with no legal depth signals.
    These skip Call 1 and go straight to a plain explanation in Call 2.
    """
    q_lower = query.lower()
    definitional_patterns = [
        r'^(what is|what are|define|definition of|meaning of|explain)\s',
        r'^(who is|who are|what does .+ mean)',
    ]
    legal_depth_signals = ['case', 'section', 'act', 'statute', 'court', 'decided',
                           'held', 'principle', 'doctrine', 'judicial']
    is_definitional = any(re.search(p, q_lower) for p in definitional_patterns)
    has_legal_depth  = any(sig in q_lower for sig in legal_depth_signals)
    # Only treat as pure definitional if short AND no legal depth signals
    return is_definitional and not has_legal_depth and len(query.split()) < 15


# ==================== QUERY CLASSIFICATION ====================
def classify_query(query: str) -> Dict:
    q_lower = query.lower()
    empty_structure = {
        'has_parts': False, 'parts': [],
        'requires_structured_response': False, 'sub_issues': []
    }

    if is_continuation_request(query):
        return {'intent': 'continuation', 'needs_search': False, 'use_sections_cases': False,
                'is_draft': False, 'is_project': False, 'is_past_question': False,
                'is_problem_question': False, 'is_law_school_question': False,
                'is_fact_specific': False, 'is_section_specific': False,
                'is_principle': False, 'use_irac': False, 'detail_level': 'continuation',
                'question_structure': empty_structure}

    if is_greeting(query):
        return {'intent': 'greeting', 'needs_search': False, 'use_sections_cases': False,
                'is_draft': False, 'is_project': False, 'is_past_question': False,
                'is_problem_question': False, 'is_law_school_question': False,
                'is_fact_specific': False, 'is_section_specific': False,
                'is_principle': False, 'use_irac': False, 'detail_level': 'minimal',
                'question_structure': empty_structure}

    question_structure = parse_question_structure(query)

    legal_kw     = ['case','law','statute','section','court','decision','ruling','legal',
                    'jurisdiction','precedent','act','constitution']
    principle_kw = ['settled law','trite law','principle','doctrine','position of the law',
                    'legal principle','established that']
    simple_kw    = ['what is','define','explain','meaning of']
    entity_kw    = ['who is','where is','what about','tell me about']
    draft_kw     = ['draft','write a','prepare a','template for','create a','compose a']
    project_kw   = ['law project','thesis','dissertation','final year project',
                    'research paper','write project','academic paper']
    fact_kw      = ['facts','details','events','parties','outcome','chronology','what happened']
    pastq_kw     = ['past question','exam question','answer this question',
                    'examination question','practice question','test question']
    irac_kw      = ['irac','issue rule application conclusion','use irac','irac format',
                    'using irac','apply irac','irac method']

    use_irac        = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in irac_kw)
    is_problem      = is_problem_question(query)
    is_law_school   = is_law_school_question(query)
    is_past_q       = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in pastq_kw)
    is_legal        = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in legal_kw)
    is_principle    = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in principle_kw)
    is_section      = bool(re.search(r'\b(section|article|s\.)\s*\d+', q_lower))
    is_case_spec    = bool(re.search(r'\b(case of|judgment in)\b|\bv\.\s*\w+', q_lower))
    is_fact_spec    = is_case_spec and any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in fact_kw)
    is_simple_nl    = not is_legal and not is_principle and any(
        re.search(rf'\b{re.escape(k)}\b', q_lower) for k in simple_kw)
    is_entity       = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in entity_kw)
    is_definitional = is_definitional_query(query)
    is_draft        = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in draft_kw)
    is_project      = any(re.search(rf'\b{re.escape(k)}\b', q_lower) for k in project_kw)

    needs_search = (len(query.split()) > 5 or is_past_q or is_problem or is_law_school or
                    is_section or is_principle or is_case_spec or is_legal)

    if is_past_q or is_problem or is_law_school or is_project: detail_level = 'comprehensive'
    elif is_draft:                                              detail_level = 'detailed'
    elif is_case_spec or is_principle:                         detail_level = 'detailed'
    elif is_section:                                           detail_level = 'moderate'
    elif is_simple_nl or is_definitional:                      detail_level = 'brief'
    else:                                                      detail_level = 'moderate'

    # ── INTENT PRIORITY ───────────────────────────────────────────────────
    # draft/project first — prevents draft queries being misclassified as
    # problem_question (which would trigger unwanted IRAC-like structure).
    # definitional early — bypasses case research for simple "what is X" queries.
    if is_draft:              intent = 'draft'
    elif is_project:          intent = 'law_project'
    elif is_definitional:     intent = 'definitional'
    elif is_law_school:       intent = 'law_school_problem'
    elif is_problem:          intent = 'problem_question'
    elif is_past_q:           intent = 'past_question'
    elif is_section:          intent = 'legal_section'
    elif is_case_spec:        intent = 'legal_case'
    elif is_principle:        intent = 'legal_principle'
    elif is_simple_nl:        intent = 'simple_non_law'
    elif is_entity:           intent = 'entity_query'
    elif is_legal:            intent = 'general_legal'
    else:                     intent = 'general_search'

    # Drafts, definitional, simple, entity queries skip Call 1
    use_sections_cases = bool(
        intent not in ('draft', 'definitional', 'simple_non_law', 'entity_query') and
        (is_case_spec or is_project or is_section or is_principle or
         is_past_q or is_problem or is_law_school or is_legal)
    )

    return {
        'intent':                 intent,
        'needs_search':           needs_search,
        'use_sections_cases':     use_sections_cases,
        'is_draft':               is_draft,
        'is_project':             is_project,
        'is_past_question':       is_past_q,
        'is_problem_question':    is_problem,
        'is_law_school_question': is_law_school,
        'is_fact_specific':       is_fact_spec,
        'is_section_specific':    is_section,
        'is_principle':           is_principle,
        'use_irac':               use_irac,
        'detail_level':           detail_level,
        'question_structure':     question_structure,
    }


# ==================== SANITIZATION ====================
def sanitize_response(response: str) -> str:
    patterns = [
        (r'\bGrok\b', 'JuristMind'), (r'\bGrok-3\b', 'JuristMind'),
        (r'\bGrok 3\b', 'JuristMind'), (r'\bGrok-4\b', 'JuristMind'),
        (r'\bGrok 4\b', 'JuristMind'), (r'\bGrok 4 Fast\b', 'JuristMind'),
        (r'\bGrok-4-1-fast\b', 'JuristMind'), (r'\bgrok-3-mini\b', 'JuristMind'),
        (r'\bxAI\b', 'Jurist Mind AI Technology'),
        (r'\bX\.AI\b', 'Jurist Mind AI Technology'),
        (r'\bcreated by xAI\b', 'developed by Jurist Mind AI Technology Nig. LTD'),
        (r'\bdeveloped by xAI\b', 'developed by Jurist Mind AI Technology Nig. LTD'),
        (r'\bX\.com\b', 'the platform'), (r'\bX platform\b', 'the platform'),
        (r'\bTwitter/X\b', 'social media'), (r'\bon X\b', 'on social media'),
    ]
    for pattern, replacement in patterns:
        response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
    return response


# ==================== CHAT HISTORY ====================
def load_chat_history(chat_id: str) -> Optional[Dict]:
    try:
        with open(f"public/chats/{chat_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_chat_history(chat_id: str, history: List[Dict]):
    os.makedirs("public/chats", exist_ok=True)
    with open(f"public/chats/{chat_id}.json", "w") as f:
        json.dump({"id": chat_id, "history": history}, f, indent=2)

def get_relevant_context(history: List[Dict], max_exchanges: int = 3) -> List[Dict]:
    if not history:
        return []
    is_project_conv = any(
        'project' in msg.get('content', '').lower() or
        'chapter' in msg.get('content', '').lower()
        for msg in history[-6:] if isinstance(msg.get('content'), str)
    )
    if is_project_conv:
        max_exchanges = 4
    relevant = []
    exchange_count = 0
    for i in range(len(history) - 1, -1, -1):
        relevant.insert(0, history[i])
        if history[i]['role'] == 'user':
            exchange_count += 1
        if exchange_count >= max_exchanges:
            break
    return relevant

def get_previous_intent(history: List[Dict]) -> Optional[str]:
    """
    Look back through history to identify what the last substantive intent was,
    so continuation requests can inherit the right format.
    """
    for msg in reversed(history):
        content = msg.get('content', '')
        if isinstance(content, str):
            if 'operative provisions' in content.lower() or 'execution' in content.lower():
                return 'draft'
            if 'chapter' in content.lower() or 'oscola' in content.lower():
                return 'law_project'
            if 'the issue is whether' in content.lower():
                return 'law_school_problem'
    return None


# ==================== DOCUMENT EXTRACTION ====================
async def extract_document_text(file: UploadFile) -> str:
    contents = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(contents))
            text = ''.join(p.extract_text() + '\n' for p in reader.pages if p.extract_text())
            if not text.strip():
                return f"Document '{file.filename}': [PDF appears empty]\n"
        elif filename.endswith(('.docx', '.doc')):
            d = docx.Document(io.BytesIO(contents))
            text = '\n'.join(p.text for p in d.paragraphs if p.text.strip())
            if not text.strip():
                return f"Document '{file.filename}': [Document appears empty]\n"
        elif filename.endswith('.txt'):
            text = contents.decode('utf-8', errors='replace')
        else:
            return f"Document '{file.filename}': [Unsupported file type]\n"
        return f"**Document '{file.filename}':**\n{text}\n"
    except Exception as e:
        logger.error(f"Error extracting {file.filename}: {e}")
        return f"Document '{file.filename}': [Error: {str(e)}]\n"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║   CALL 1 — CASE FINDER                                              ║
# ║                                                                      ║
# ║   SKIPPED for: draft, definitional, simple_non_law, entity_query    ║
# ║   PARALLEL for: multi-area law school sub-questions                 ║
# ║   NO TIMEOUT — runs to completion every time.                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

CASE_FINDER_SYSTEM = """You are a Nigerian legal case researcher. Your ONLY job is to 
find a real Nigerian case and return structured JSON. You do NOT write legal analysis.

You think carefully before responding. You only report what you actually found 
in search results. You never invent facts or decisions.

Return ONLY valid JSON in this exact structure — no preamble, no explanation:

{
  "status": "found_direct" | "found_related" | "not_found",
  "case_name": "Full Party Name v Other Party Name",
  "year": "YYYY",
  "court": "Name of the court that decided this case",
  "legal_area": "The area of law this case covers",
  "facts": "2-4 sentence narrative of what actually happened. Who did what to whom. Be specific. If uncertain write: UNVERIFIED",
  "decision": "What the court specifically decided or ordered — quote directly from source if possible. If not found write: UNVERIFIED",
  "source_found": true | false,
  "confidence": "high" | "medium" | "low",
  "notes": "Any important caveat or limitation about what you found"
}

STRICT RULES:
- Search first. Read the search result text. Only report what you found there.
- facts field: what happened between the parties (narrative, not legal conclusion)
- decision field: what the court said or ordered (their actual words if possible)
- If a search result has a case name but no facts/decision text → confidence: low, both fields: UNVERIFIED
- If using a related case (same area, not exact topic) → status: found_related
- If nothing found → status: not_found, confidence: low
- NEVER write a decision that matches the query topic but was not in your search result
- Return ONLY the JSON. Nothing else."""


async def _run_case_finder(query: str, legal_area: str, current_year: int) -> Dict:
    """
    Core case finder — no timeout, runs to completion.
    Logs a heartbeat every 10s so slow searches are visible in server logs.
    """
    finder_prompt = f"""The user's legal query is: "{query}"
Detected legal area: {legal_area}
Current year: {current_year}

Follow this search waterfall:

STEP 1 — Search for cases DIRECTLY on this query:
  Try: "{query} Nigeria court case judgment"
  Try: "{query} Nigeria decided held Supreme Court Court of Appeal"
  Read the results. Did you find an actual case with real facts and a real decision?
  → If YES: status "found_direct". Fill facts and decision from what you read.
  → If NO: go to Step 2.

STEP 2 — Search for RELATED cases in the same legal area:
  Try: "{legal_area} landmark case Nigeria court judgment held"
  Try: "{legal_area} Nigeria Supreme Court decided"
  Read the results. Find a case in the same area of law.
  → If YES: status "found_related". Note this in the notes field.
  → If NO: go to Step 3.

STEP 3 — If both failed: status "not_found". Facts and decision: UNVERIFIED.

For facts: describe what happened between the parties — who sued whom, what they did.
For decision: what did the court actually say or order? Quote if possible.

Return ONLY the JSON object now."""

    client = Client(api_key=GROK_API_KEY)
    chat   = client.chat.create(model=THINKING_MODEL, tools=[web_search(), x_search()])
    chat.append(system(CASE_FINDER_SYSTEM))
    chat.append(user(finder_prompt))

    full_response = ""
    start_time    = time.time()

    for response, chunk in chat.stream():
        if chunk.content:
            full_response += chunk.content
        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 10 == 0:
            logger.info(f"Case finder still running... {elapsed}s elapsed for: {query[:50]}")

    full_response = re.sub(r'```json\s*', '', full_response.strip())
    full_response = re.sub(r'```\s*', '',  full_response)

    json_match = re.search(r'\{[\s\S]*\}', full_response)
    if json_match:
        return json.loads(json_match.group())
    return {"status": "not_found", "confidence": "low",
            "source_found": False, "notes": "No parseable JSON returned"}


async def call1_find_case(query: str, classification: Dict) -> Dict:
    """Single-question case finder. Runs to completion — no timeout."""
    if not classification.get('use_sections_cases', False):
        return {"status": "not_found", "confidence": "low", "source_found": False}

    legal_area   = detect_legal_area(query)
    current_year = datetime.now().year

    try:
        case_data = await _run_case_finder(query, legal_area, current_year)
        logger.info(
            f"Call 1 → status={case_data.get('status')} | "
            f"confidence={case_data.get('confidence')} | "
            f"case='{case_data.get('case_name', 'none')}'"
        )
        return case_data
    except json.JSONDecodeError as e:
        logger.error(f"Call 1 JSON error: {e}")
        return {"status": "not_found", "confidence": "low",
                "source_found": False, "notes": f"JSON parse error: {str(e)}"}
    except Exception as e:
        logger.error(f"Call 1 failed: {e}")
        return {"status": "not_found", "confidence": "low",
                "source_found": False, "notes": f"Error: {str(e)}"}


async def call1_find_cases_parallel(
    sub_issues: List[Tuple[str, str]], classification: Dict
) -> Dict[str, Dict]:
    """
    For multi-area law school sub-questions: one case search per sub-issue, all in parallel.
    Each search runs to completion. Returns dict keyed by label: {"(a)": {...}, "(b)": {...}}
    """
    current_year = datetime.now().year

    async def find_one(label: str, issue_text: str) -> Tuple[str, Dict]:
        try:
            result = await _run_case_finder(
                issue_text, detect_legal_area(issue_text), current_year
            )
            logger.info(
                f"Parallel {label} → {result.get('status')} | {result.get('case_name','none')}"
            )
            return label, result
        except Exception as e:
            logger.error(f"Parallel case search {label} failed: {e}")
            return label, {"status": "not_found", "confidence": "low",
                           "source_found": False, "notes": str(e)}

    results = await asyncio.gather(*[find_one(lbl, txt) for lbl, txt in sub_issues])
    return {label: data for label, data in results}


# ==================== CASE CONTEXT BUILDER ====================
def build_case_context_block(case_data: Dict) -> str:
    """Convert a single Call 1 JSON result into a structured anchor block for the writer."""
    status     = case_data.get('status', 'not_found')
    confidence = case_data.get('confidence', 'low')

    if status == 'not_found' or confidence == 'low':
        return """╔══════════════════════════════════════════════════════════╗
║  CASE RESEARCH RESULT: NO VERIFIED CASE FOUND           ║
╚══════════════════════════════════════════════════════════╝

A waterfall search was completed. No Nigerian case with confirmed facts
and a confirmed court decision was found for this query.

WRITER INSTRUCTION:
  ✗ Do NOT cite any case by name
  ✗ Do NOT invent a case name, facts, or holding
  ✓ Provide statutory analysis ONLY
  ✓ Quote the applicable section(s) of Nigerian law (as amended)
  ✓ State the legal position from statute and established principles
  ✓ Be transparent: "No specific case was located for this point during research."
"""

    case_name  = case_data.get('case_name', 'Unknown Case')
    year       = case_data.get('year', '')
    court      = case_data.get('court', '')
    legal_area = case_data.get('legal_area', '')
    facts      = case_data.get('facts', 'UNVERIFIED')
    decision   = case_data.get('decision', 'UNVERIFIED')
    notes      = case_data.get('notes', '')
    is_related = (status == 'found_related')

    facts_ok    = facts    not in ('UNVERIFIED', '', None)
    decision_ok = decision not in ('UNVERIFIED', '', None)

    block = f"""╔══════════════════════════════════════════════════════════╗
║  CASE RESEARCH RESULT: {"RELATED CASE" if is_related else "CASE FOUND":20s}  Confidence: {confidence.upper():6s} ║
╚══════════════════════════════════════════════════════════╝

  Case Name : {case_name} ({year})
  Court     : {court}
  Area      : {legal_area}
{"  ⚠️  RELATED CASE — same legal area, not identical topic" if is_related else ""}
{f"  Notes     : {notes}" if notes else ""}

──────────────────────────────────────────────────────────
FACTS (what actually happened between the parties):
{facts if facts_ok else "⚠️  Facts could not be verified from source text."}

COURT'S DECISION:
{decision if decision_ok else "⚠️  Decision text could not be verified from source text."}

──────────────────────────────────────────────────────────
WRITER INSTRUCTIONS — READ CAREFULLY:

"""
    if not facts_ok and not decision_ok:
        block += f"""Both FACTS and DECISION are UNVERIFIED.

  ✓ You may note: "{case_name} ({year}) has been cited in {legal_area}."
  ✗ Do NOT state what it held or describe its facts.
  ✓ Follow with statutory analysis only.
  ✓ State: "The specific facts and decision could not be confirmed during research.
    Please verify at NigeriaLII.org or official law reports."
"""
    elif facts_ok and not decision_ok:
        block += f"""FACTS available. DECISION is UNVERIFIED.

  ✓ Narrate the facts in your own words (2-3 sentences).
  ✓ Introduce: "In {case_name} ({year}), [narrate facts]..."
  ✗ Do NOT state what the court held.
  ✓ Write: "The court's specific decision could not be confirmed from available sources."
  ✓ Follow with statutory analysis.
"""
    elif not facts_ok and decision_ok:
        block += f"""FACTS are UNVERIFIED. DECISION available.

  ✗ Do NOT narrate facts you don't have.
  ✓ Introduce directly: "In {case_name} ({year}), the court held that [decision text]."
  ✓ Explain what this decision means for the query.
"""
    else:
        if is_related:
            block += f"""Both FACTS and DECISION confirmed. RELATED case.

  ✓ Introduce: "While no case directly on this point was found, a closely related
    case in the same area of law is instructive — {case_name} ({year})."
  ✓ Narrate FACTS in 2-3 sentences.
  ✓ State the DECISION — quote or closely paraphrase.
  ✓ Draw the legal principle and apply it to the query.
  ✗ Do NOT present as directly decided on the user's exact topic.
"""
        else:
            block += f"""Both FACTS and DECISION confirmed. Full citation permitted.

  ✓ Narrate FACTS in 2-3 sentences — make it a story.
  ✓ State the DECISION — quote or closely paraphrase.
  ✓ Apply the principle to the user's query.
  ✓ Introduce naturally in your analysis.
"""
    return block


def build_multi_case_context_block(cases_by_part: Dict[str, Dict]) -> str:
    """
    Combined context block for multi-part law school problems where each
    sub-question has its own parallel case research result.
    """
    if not cases_by_part:
        return build_case_context_block({"status": "not_found", "confidence": "low"})

    combined  = "╔══════════════════════════════════════════════════════════╗\n"
    combined += "║  CASE RESEARCH: MULTI-PART RESULTS (PARALLEL SEARCH)    ║\n"
    combined += "╚══════════════════════════════════════════════════════════╝\n\n"
    combined += "Each sub-question was researched independently. Apply the relevant\n"
    combined += "case context when addressing each part, as instructed below.\n\n"

    for label, case_data in cases_by_part.items():
        combined += f"═══ SUB-QUESTION {label} ═══\n"
        combined += build_case_context_block(case_data)
        combined += "\n"

    return combined


# ==================== ANSWER PROMPT BUILDER ====================
def build_answer_prompt(
    query: str,
    classification: Dict,
    case_context: str,
    document_texts: List[str] = None,
    has_images: bool = False
) -> str:
    """
    Build the prompt for Call 2 based on intent.

    Key rules:
      - draft:             no case context, no analysis preamble, direct document
      - law_school_problem: IRAC per issue always (this is the correct academic format)
      - problem_question:  adviser prose — IRAC only if user explicitly asked
      - past_question:     exam prose — IRAC only if user explicitly asked
      - definitional/simple: plain explanation, no case context, no restructuring
      - everything else:   statutory + case analysis in flowing paragraphs
    """
    intent             = classification['intent']
    use_irac           = classification.get('use_irac', False)
    question_structure = classification.get('question_structure', {})
    current_year       = datetime.now().year

    prompt = f"User query: {query}\n\n"
    if document_texts:
        prompt += "**Attached Documents:**\n" + "\n\n".join(document_texts) + "\n\n"
    if has_images:
        prompt += "**Attached Images:** Analyse all visual content thoroughly.\n\n"

    # Case context is ONLY injected for legal analysis intents, not drafts or definitions
    if intent not in ('draft', 'definitional', 'simple_non_law', 'entity_query'):
        prompt += case_context + "\n\n"

    prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    prompt += "YOUR ROLE: ANSWER WRITER\n"
    prompt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # ─── DRAFT ────────────────────────────────────────────────────────────
    if intent == 'draft':
        prompt += f"""ABSOLUTE RULES FOR DRAFTING:
  1. Produce the document DIRECTLY. No preamble, no legal analysis, no IRAC.
     The very first line of your response must be the document TITLE.
  2. You MAY search for relevant Nigerian statute sections to include in the
     Legal Basis section at the END of the document (as amended {current_year}).
  3. Write in formal legal document style throughout.

**DOCUMENT STRUCTURE:**
  TITLE
  DATE
  PARTIES (full names, descriptions, addresses)
  RECITALS / BACKGROUND
  OPERATIVE PROVISIONS (numbered clauses)
  GENERAL / BOILERPLATE CLAUSES (governing law, dispute resolution, notices)
  EXECUTION (signature blocks)
  LEGAL BASIS (applicable Nigerian statute sections, as amended {current_year})

End with:
⚖️ This is a template draft. Consult a qualified lawyer before use: https://Lawyernearme.com.ng

Length: 800–1500 words
"""

    # ─── LAW SCHOOL PROBLEM — IRAC per issue, always ─────────────────────
    elif intent == 'law_school_problem':
        prompt += f"""ABSOLUTE RULES:
  1. This is an academic law school question. Apply IRAC to EVERY distinct legal issue.
  2. Follow WRITER INSTRUCTIONS in the case context block above — exactly.
  3. Search for and quote relevant statutory sections (as amended {current_year}).
  4. Write in formal academic prose.

**LAW SCHOOL PROBLEM — IRAC FORMAT:**

First, read the entire scenario and IDENTIFY every distinct legal issue present.
(There are often 3–6 separate issues — find them all before you start writing.)

For EACH issue, write:

  **[Issue Heading — e.g. "Issue 1: Formation of Contract"]**

  ISSUE: "The issue is whether..."

  RULE:
    • Quote the relevant statute section verbatim (as amended {current_year})
    • Present the case from this issue's case context (follow its writer instructions)

  APPLICATION:
    • Apply the rule to the specific facts — reason step by step
    • Do not skip logical steps

  CONCLUSION:
    • "Therefore, [party] is/is not [liable/entitled...] because..."

After all issues, write a FINAL OVERALL CONCLUSION summarising the legal position
of each party and available remedies.

Length: 600–1000 words
"""
        if question_structure.get('has_parts'):
            parts_list = ', '.join(question_structure.get('parts', []))
            prompt += f"\nAddress each numbered part separately: {parts_list}\n"

    # ─── REAL-PERSON PROBLEM — adviser prose ─────────────────────────────
    elif intent == 'problem_question':
        if use_irac:
            prompt += f"""IRAC FORMAT (explicitly requested):

ABSOLUTE RULES:
  1. Follow WRITER INSTRUCTIONS in the case context block above — exactly.
  2. Search for and quote the relevant statutory section (as amended {current_year}).

ISSUE → RULE (statute + case) → APPLICATION → CONCLUSION
Length: 500–800 words
"""
        else:
            prompt += f"""ABSOLUTE RULES:
  1. Write as a SENIOR LEGAL ADVISER giving practical advice.
  2. Follow WRITER INSTRUCTIONS in the case context block above — exactly.
  3. Search for and quote the relevant statutory section (as amended {current_year}).
  4. Do NOT use IRAC headings. Write in flowing paragraphs.

**LEGAL ADVICE FORMAT:**
  Para 1 — Briefly identify the legal issues arising from the facts.
  Para 2 — State the applicable Nigerian law: quote the relevant statute section.
  Para 3 — Present and apply the case from the case context (follow its instructions).
  Para 4 — Apply the law to the specific facts, reasoning through each issue.
  Para 5 — Practical advice: available remedies, what action to take, caveats.

Length: 500–800 words
"""
        if question_structure.get('has_parts'):
            parts_list = ', '.join(question_structure.get('parts', []))
            prompt += f"\nAddress each part under its own heading: {parts_list}\n"

    # ─── PAST / EXAM QUESTION ─────────────────────────────────────────────
    elif intent == 'past_question':
        if use_irac:
            prompt += f"""IRAC FORMAT (explicitly requested):

ABSOLUTE RULES:
  1. Follow WRITER INSTRUCTIONS in the case context block — exactly.
  2. Search and quote relevant statute section (as amended {current_year}).

ISSUE → RULE → APPLICATION → CONCLUSION. Length: 400–600 words
"""
        else:
            prompt += f"""ABSOLUTE RULES:
  1. Answer as a well-prepared law student in an examination.
  2. Follow WRITER INSTRUCTIONS in the case context block — exactly.
  3. Search and quote relevant statute section (as amended {current_year}).
  4. Do NOT use IRAC sub-headings. Write in paragraphs.

**EXAM ANSWER FORMAT:**
  Introduction — identify the legal issues.
  Legal Framework — quote the statute section and present the case.
  Analysis — apply the law methodically.
  Conclusion — firm, clear statement.

Length: 400–600 words
"""

    # ─── SECTION ANALYSIS ────────────────────────────────────────────────
    elif intent == 'legal_section':
        prompt += f"""ABSOLUTE RULES:
  1. Search for and quote the current section text (as amended {current_year}) verbatim.
  2. Follow WRITER INSTRUCTIONS in the case context block — exactly.

**SECTION ANALYSIS:**
  Quote the section verbatim. Explain in plain English. Give a practical example.
  Show how courts have applied it using the case from the context block.
Length: 200–400 words
"""

    # ─── LEGAL CASE / PRINCIPLE / GENERAL ────────────────────────────────
    elif intent in ('legal_case', 'legal_principle', 'general_legal', 'general_search'):
        prompt += f"""ABSOLUTE RULES:
  1. Follow WRITER INSTRUCTIONS in the case context block — exactly.
  2. Search and quote the relevant statute section (as amended {current_year}).
  3. Do NOT use IRAC headings. Write in flowing paragraphs.

**LEGAL ANALYSIS:**
  Open with a clear statement of the legal position.
  Set out the statutory basis — quote the section.
  Present and apply the case from the context block.
  Close with a practical conclusion.
Length: 400–700 words
"""

    # ─── LAW PROJECT / THESIS ─────────────────────────────────────────────
    elif intent == 'law_project':
        prompt += f"""ABSOLUTE RULES:
  1. Follow WRITER INSTRUCTIONS in the case context block — exactly.
  2. OSCOLA citations throughout.
  3. Formal academic prose.

**LAW PROJECT / THESIS CHAPTER:**
  One chapter at a time. Proper headings and sub-headings.
  Cite all statutes (as amended {current_year}) and cases per OSCOLA.
Length: 1000–2000 words per chapter
"""

    # ─── DEFINITIONAL / SIMPLE / ENTITY — plain explanation ──────────────
    elif intent in ('definitional', 'simple_non_law', 'entity_query'):
        prompt += f"""**PLAIN EXPLANATION:**
  Answer directly and clearly — no case context injection, no IRAC.
  Define/explain in plain English first.
  If it has a legal dimension under Nigerian law, explain it concisely,
  quoting any directly relevant statute section (as amended {current_year}).
  Write in 2–4 clear paragraphs.
Length: 150–350 words
"""

    else:
        prompt += f"""**RESPONSE:**
Clear paragraphs. Verified statute section (as amended {current_year}).
Case from context if available. No IRAC headings unless requested.
300–500 words.
"""

    prompt += "\nClose with a brief, helpful follow-up suggestion for the user.\n"
    prompt += "\nNow write the response as JuristMind.\n"
    return prompt


# ==================== CALL 2: WRITER ====================
async def query_grok_writer(messages: List[Dict], classification: Dict):
    """
    Call 2: streaming writer.
    Searches for statutory sections only — never for cases (that is Call 1's job).
    """
    if not GROK_API_KEY:
        yield {"type": "error", "message": "No API key configured."}
        return

    try:
        client = Client(api_key=GROK_API_KEY)
        tools  = [web_search()] if classification.get('needs_search', False) else []

        system_prompt      = ""
        processed_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            else:
                processed_messages.append(msg)

        chat = client.chat.create(model=WRITING_MODEL, tools=tools)
        if system_prompt:
            chat.append(system(system_prompt))
        for msg in processed_messages:
            if msg['role'] == 'user':
                chat.append(user(msg['content']))
            elif msg['role'] == 'assistant':
                chat.append(assistant(msg['content']))

        last_yield_time = time.time()
        citations       = None

        for response, chunk in chat.stream():
            now = time.time()
            if now - last_yield_time > 5:
                yield {"type": "ping"}
                last_yield_time = now
            if chunk.content:
                sanitized = sanitize_response(chunk.content)
                yield {"type": "content", "delta": sanitized}
                last_yield_time = now
            if response.citations:
                citations = response.citations

        if citations:
            filtered = []
            for c in citations:
                if isinstance(c, dict) and 'url' in c:
                    url_lower = c['url'].lower()
                    if 'x.com' not in url_lower and 'twitter.com' not in url_lower:
                        c['priority'] = 1 if ('.gov.ng' in url_lower or '.edu' in url_lower) else 2
                        filtered.append(c)
            filtered.sort(key=lambda x: x.get('priority', 2))
            if filtered:
                yield {"type": "citations", "data": filtered}

    except Exception as e:
        logger.exception(f"Writer call failed: {e}")
        yield {"type": "error", "message": f"API error: {str(e)}"}


# ==================== GENERAL ANSWERS ====================
GENERAL_ANSWERS = {
    "what is your name": "I am JuristMind, your AI legal assistant specialising in Nigerian law.",
    "who created you": "I was developed by Jurist Mind AI Technology Nig. LTD, founded by Oluwaseun Ogun, to assist with legal research and provide accessible legal information.",
    "who developed you": "I was developed by Jurist Mind AI Technology Nig. LTD to make legal knowledge more accessible.",
    "who made you": "I was created by Jurist Mind AI Technology Nig. LTD to assist with Nigerian legal matters.",
    "how old are you": "I was recently launched to help people understand and navigate Nigerian law more easily.",
    "what can you do": "I can help with legal research, explain Nigerian laws, draft legal documents, analyse cases, answer past questions, assist with law projects, and provide legal advice on scenarios. How can I help you today?"
}

def handle_general_query(question: str) -> Optional[str]:
    if is_greeting(question):
        return generate_greeting_response(question)
    q = question.lower().strip()
    for key, answer in GENERAL_ANSWERS.items():
        if key in q:
            return answer
    return None


# ==================== TEMPLATE LOADING ====================
def load_template(template_name: str) -> str:
    try:
        with open(f"templates/{template_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "{content}"


# ==================== FASTAPI APP ====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def root():
    return JSONResponse({
        "message": "Welcome to JuristMind — Your AI Legal Assistant for Nigerian Law",
        "version": "7.0.0",
        "architecture": (
            "Three paths: "
            "(1) Draft — detail check then direct draft, NO Call 1. "
            "(2) Definitional/simple — direct plain explanation, NO Call 1. "
            "(3) Legal analysis — Call 1 (runs to completion, parallel for multi-area) "
            "→ Call 2 (law_school_problem=IRAC always, problem_question=adviser prose, "
            "past_question=exam prose, IRAC only if explicitly requested elsewhere)."
        )
    })

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

@app.get("/ask")
async def ask_get():
    return JSONResponse({
        "message": "Please use POST /ask",
        "required_fields": {
            "question": "Your legal question",
            "chat_id":  "(optional) Continue existing conversation",
            "files":    "(optional) Documents or images"
        }
    })


# ==================== MAIN ENDPOINT ====================
@app.post("/ask")
async def ask_question(request: Request):
    """
    JuristMind — three-path architecture:

    PATH A — GREETING / PREDEFINED:
      Instant response, no model API call (or FAST_MODEL for greetings).

    PATH B — DRAFT:
      Missing details → ask politely (no model call).
      Has details     → Call 2 directly (no Call 1 at all).
      Draft goes straight to document. No legal analysis preamble.

    PATH C — ALL LEGAL ANALYSIS:
      Call 1 (THINKING_MODEL): case finder, runs to completion, no timeout.
        Multi-area sub-questions → parallel searches via asyncio.gather.
      Call 2 (WRITING_MODEL): answer writer anchored to Call 1's case context.
        law_school_problem → IRAC per issue (always, correct academic format).
        problem_question   → adviser prose (IRAC only if user asked).
        past_question      → exam prose (IRAC only if user asked).
        definitional/simple → plain explanation (Call 1 skipped).
    """
    form     = await request.form()
    question = form.get("question", "").strip()
    chat_id  = form.get("chat_id")
    files    = form.getlist("files")

    if not question and not files:
        return JSONResponse({"error": "Please provide a question or upload files"})

    document_texts = []
    image_contents = []

    if files:
        for file in files:
            if isinstance(file, UploadFile):
                contents = await file.read()
                filename = file.filename.lower()
                if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    ext = 'jpeg' if filename.endswith('.jpg') else filename.split('.')[-1]
                    image_contents.append((ext, base64.b64encode(contents).decode('utf-8')))
                else:
                    document_texts.append(await extract_document_text(file))

    has_images     = len(image_contents) > 0
    classification = classify_query(question)
    general_answer = handle_general_query(question)

    logger.info(
        f"Query: '{question[:80]}' | "
        f"Intent: {classification['intent']} | "
        f"IRAC: {classification['use_irac']} | "
        f"LawSchool: {classification['is_law_school_question']}"
    )

    async def generate():
        nonlocal chat_id
        full_text = ""
        citations = []
        has_error = False

        full_history = []
        if chat_id:
            chat_data = load_chat_history(chat_id)
            if chat_data:
                full_history = chat_data.get("history", [])
        else:
            chat_id = str(uuid.uuid4())

        user_content = question
        if document_texts: user_content += "\n\n[Documents attached]"
        if has_images:      user_content += "\n\n[Images attached]"

        if has_images:
            user_msg = {"role": "user", "content": [
                {"type": "text", "text": user_content},
                *[{"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}
                  for ext, b64 in image_contents]
            ]}
        else:
            user_msg = {"role": "user", "content": user_content}

        full_history.append(user_msg)

        # ════════════════════════════════════════════════════════════════
        # PATH A — GREETING / PREDEFINED
        # ════════════════════════════════════════════════════════════════
        if general_answer:
            sanitized = sanitize_response(general_answer)
            full_text = sanitized
            full_history.append({"role": "assistant", "content": sanitized})
            yield f"data: {json.dumps({'content': sanitized})}\n\n"

        # ════════════════════════════════════════════════════════════════
        # PATH B — DRAFT (bypass Call 1 entirely)
        # ════════════════════════════════════════════════════════════════
        elif classification['intent'] == 'draft':
            draft_type = detect_draft_type(question)

            # Check if we previously asked for details
            last_assistant_msg = next(
                (m['content'] for m in reversed(full_history[:-1])
                 if m['role'] == 'assistant' and isinstance(m.get('content'), str)),
                ""
            )
            already_asked_for_details = (
                "please provide the following details" in last_assistant_msg.lower()
            )

            if not already_asked_for_details and not draft_has_sufficient_details(question, draft_type):
                # Ask for details — zero model API calls
                details_req = build_draft_details_request(draft_type)
                sanitized   = sanitize_response(details_req)
                full_text   = sanitized
                full_history.append({"role": "assistant", "content": sanitized})
                yield f"data: {json.dumps({'content': sanitized})}\n\n"

            else:
                # Sufficient details present — go straight to drafting
                yield f"data: {json.dumps({'type': 'status', 'message': '✍️ Drafting your document...'})}\n\n"

                system_content = (
                    "You are JuristMind, an expert AI legal assistant specialising in Nigerian law. "
                    "DRAFTING RULE: When drafting legal documents, produce the document directly "
                    "and immediately. The first line of your response must be the document TITLE. "
                    "Do NOT include any preamble, issue identification, legal analysis, or IRAC "
                    "before the document itself. Structure is always: "
                    "Title → Parties → Recitals → Operative Provisions → General Clauses → "
                    "Execution → Legal Basis. "
                    "You may search for relevant Nigerian statute sections to cite in Legal Basis."
                )

                draft_prompt = build_answer_prompt(
                    question, classification, "",
                    document_texts, has_images
                )

                messages_to_send = [{"role": "system", "content": system_content}]
                if len(full_history) > 1:
                    messages_to_send.extend(get_relevant_context(full_history[:-1]))

                if has_images:
                    messages_to_send.append({"role": "user", "content": [
                        {"type": "text", "text": draft_prompt},
                        *[{"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}
                          for ext, b64 in image_contents]
                    ]})
                else:
                    messages_to_send.append({"role": "user", "content": draft_prompt})

                suffix = ""
                template = load_template("default_legal")
                if '{content}' in template:
                    parts  = template.split('{content}', 1)
                    prefix = sanitize_response(parts[0])
                    suffix = sanitize_response(parts[1]) if len(parts) > 1 else ""
                    if prefix:
                        yield f"data: {json.dumps({'content': prefix})}\n\n"
                        full_text += prefix

                async for item in query_grok_writer(messages_to_send, classification):
                    if item["type"] == "content":
                        yield f"data: {json.dumps({'content': item['delta']})}\n\n"
                        full_text += item["delta"]
                    elif item["type"] == "citations":
                        citations = item["data"]
                    elif item["type"] == "ping":
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    elif item["type"] == "error":
                        has_error = True
                        err = f"\n\n⚠️ **Error:** {item['message']}\n\nPlease try again."
                        yield f"data: {json.dumps({'content': err})}\n\n"
                        full_text += err

                if suffix and not has_error:
                    yield f"data: {json.dumps({'content': suffix})}\n\n"
                    full_text += suffix

        # ════════════════════════════════════════════════════════════════
        # PATH C — ALL LEGAL ANALYSIS: Call 1 → Call 2
        # ════════════════════════════════════════════════════════════════
        else:
            # ── CALL 1: Case research (no timeout — runs to completion) ──
            case_data     = {"status": "not_found", "confidence": "low", "source_found": False}
            cases_by_part = {}
            sub_issues    = classification['question_structure'].get('sub_issues', [])

            if classification.get('use_sections_cases', False):
                yield f"data: {json.dumps({'type': 'status', 'message': '⚖️ Researching Nigerian case law...'})}\n\n"

                try:
                    if sub_issues and len(sub_issues) > 1:
                        # Multi-area law school problem — parallel search per sub-issue
                        yield f"data: {json.dumps({'type': 'status', 'message': f'⚖️ Searching {len(sub_issues)} legal issues in parallel...'})}\n\n"
                        cases_by_part = await call1_find_cases_parallel(sub_issues, classification)
                        best = next(
                            (v for v in cases_by_part.values() if v.get('status') != 'not_found'),
                            {"status": "not_found", "confidence": "low", "source_found": False}
                        )
                        case_data = best
                    else:
                        # Single question or same-area multi-part
                        case_data = await call1_find_case(question, classification)
                except Exception as e:
                    logger.error(f"Call 1 exception: {e}")
                    case_data = {"status": "not_found", "confidence": "low",
                                 "source_found": False, "notes": str(e)}

            # Build case context block
            if cases_by_part:
                case_context = build_multi_case_context_block(cases_by_part)
            else:
                case_context = build_case_context_block(case_data)

            logger.info(
                f"Case research done → "
                f"status={case_data.get('status')} | "
                f"confidence={case_data.get('confidence')} | "
                f"case='{case_data.get('case_name', 'none')}' | "
                f"sub_parts={len(cases_by_part)}"
            )

            # ── CALL 2: Writing model ─────────────────────────────────────
            yield f"data: {json.dumps({'type': 'status', 'message': '✍️ Writing response...'})}\n\n"

            system_content = (
                "You are JuristMind, an expert AI legal assistant specialising in Nigerian law. "
                "You write clear, accurate, well-structured legal responses. "
                "CRITICAL — IRAC RULES: "
                "  (1) law_school_problem → ALWAYS use IRAC per distinct legal issue. "
                "      This is the correct academic format for law school questions. "
                "  (2) ALL OTHER INTENTS → Do NOT use IRAC headings (Issue/Rule/Application/"
                "      Conclusion) unless the user has explicitly asked for IRAC. "
                "      Write naturally as a senior legal adviser: identify the issue, "
                "      state the law, apply it, conclude — but in flowing prose. "
                "CRITICAL — DRAFT RULE: When drafting, start directly with the document TITLE. "
                "      No preamble, no analysis before the document. "
                "CRITICAL — CASE RULES: "
                "  ✗ Never invent case names, facts, or holdings. "
                "  ✓ Follow the writer instructions in the case context block exactly. "
                "  ✓ If no case is provided, give authoritative statutory analysis and "
                "    say so transparently."
            )

            answer_prompt    = build_answer_prompt(
                question, classification, case_context,
                document_texts, has_images
            )
            messages_to_send = [{"role": "system", "content": system_content}]

            if len(full_history) > 1:
                messages_to_send.extend(get_relevant_context(full_history[:-1]))

            if has_images:
                messages_to_send.append({"role": "user", "content": [
                    {"type": "text", "text": answer_prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}}
                      for ext, b64 in image_contents]
                ]})
            else:
                messages_to_send.append({"role": "user", "content": answer_prompt})

            suffix = ""
            if classification['intent'] == 'law_project':
                template = load_template("law_project")
                if '{content}' in template:
                    parts  = template.split('{content}', 1)
                    prefix = sanitize_response(parts[0])
                    suffix = sanitize_response(parts[1]) if len(parts) > 1 else ""
                    if prefix:
                        yield f"data: {json.dumps({'content': prefix})}\n\n"
                        full_text += prefix

            async for item in query_grok_writer(messages_to_send, classification):
                if item["type"] == "content":
                    yield f"data: {json.dumps({'content': item['delta']})}\n\n"
                    full_text += item["delta"]
                elif item["type"] == "citations":
                    citations = item["data"]
                elif item["type"] == "ping":
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                elif item["type"] == "error":
                    has_error = True
                    err = f"\n\n⚠️ **Error:** {item['message']}\n\nPlease try again."
                    yield f"data: {json.dumps({'content': err})}\n\n"
                    full_text += err

            if suffix and not has_error:
                yield f"data: {json.dumps({'content': suffix})}\n\n"
                full_text += suffix

            # Research metadata footer
            status     = case_data.get('status', 'not_found')
            confidence = case_data.get('confidence', 'low')
            c_name     = case_data.get('case_name', '')
            c_year     = case_data.get('year', '')

            if status != 'not_found' and confidence != 'low' and c_name:
                meta = f"\n\n---\n*Research: **{c_name} ({c_year})**"
                if status == 'found_related':
                    meta += " — related case (same legal area)"
                meta += f" | Confidence: {confidence}*"
                if cases_by_part:
                    meta += f" | {len(cases_by_part)} issues searched in parallel"
                yield f"data: {json.dumps({'content': meta})}\n\n"
                full_text += meta

        # ── Save history non-blocking ─────────────────────────────────────
        if not has_error or full_text.strip():
            full_history.append({"role": "assistant", "content": full_text})
            asyncio.create_task(asyncio.to_thread(save_chat_history, chat_id, full_history))

        # ── Sources ───────────────────────────────────────────────────────
        sources = []
        if citations:
            for c in citations:
                if isinstance(c, dict) and 'url' in c:
                    sources.append({
                        "title":    c.get("title", c.get("url", ""))[:100],
                        "url":      c.get("url", ""),
                        "priority": c.get("priority", 2)
                    })
            if sources and "Sources:" not in full_text:
                src_text = "\n\n**Sources:**\n" + "\n".join(
                    [f"- {s['title']}: {s['url']}" for s in sources[:5]])
                yield f"data: {json.dumps({'content': src_text})}\n\n"

        chat_url = f"{BASE_CHAT_URL}/public/chats/{chat_id}.json"
        yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'chat_url': chat_url, 'sources': sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== CHAT RETRIEVAL ====================
@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    chat_data = load_chat_history(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return JSONResponse(chat_data)


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
