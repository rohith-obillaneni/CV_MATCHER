import os
import re
import json
import openai
import streamlit as st
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
from scipy.special import expit # For sigmoid activation on cross-encoder scores
from dotenv import load_dotenv
from functools import lru_cache
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import requests
import asyncio
from openai import AsyncOpenAI
from urllib.parse import urljoin, urlparse

# ─────────────────────────────────────────────────
# CONFIGURATION & HYPERPARAMETERS
# ─────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX = "head-of-ai"
TOP_K = 10

# --- SCORING WEIGHTS RE-CALIBRATED FOR HIGHER, MORE ACCURATE SCORES ---
PINE_WEIGHT_FINAL_SCORE_CONTRIBUTION = 0.25  # Increased for stronger base contribution
CROSS_ENCODER_MAX_POINTS = 2.50             # Increased for stronger relevance signal
SKILL_COVERAGE_MAX_POINTS = 3.0            # Increased to better reward specific skill matches
RELEVANT_EXPERIENCE_MAX_POINTS = 4.0       # HEAVILY INCREASED to better reward senior candidates (distribution refined below)
ACHIEVEMENT_MAX_POINTS = 2.5              # Increased to reward demonstrated impact more significantly
SENIORITY_MAX_POINTS = 1.25                 # Increased to strongly reward senior titles
PROBLEM_SOLVING_MAX_POINTS = 1.25           # Increased
METADATA_SKILL_BONUS = 0.35                # Maintained
EVIDENCE_SOURCE_BONUS_QUERY = 0.15         # Maintained
EVIDENCE_SOURCE_BONUS_SKILLS = 0.1         # Maintained
EVIDENCE_SOURCE_PENALTY_NONE = -0.1        # Reduced penalty for no explicit evidence

CROSS_ENCODER_MIN_CONTRIBUTION = 0.4       # Lowered to give points to moderately relevant candidates
MIN_SIM_THRESHOLD = 0.55
MAX_EVIDENCE = 3
FUZZY_THRESHOLD = 0.85
LLM_SECTOR_CONFIDENCE_THRESHOLD = 0.7

openai.api_key = OPENAI_API_KEY
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(PINECONE_INDEX)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Ensure NLTK data is available
import nltk
nltk.data.path.append("nltk_data")
from nltk.tokenize import sent_tokenize

# Global list of allowed sectors for direct input check
ALLOWED_SECTORS = [
    "Marketing Agency", "Market Research", "FinTech", "Retail", "HealthTech", "EdTech", "LegalTech",
    "SaaS", "Consulting", "AI", "Telecommunications", "Logistics", "Cybersecurity",
    "HRTech", "Manufacturing", "Finance", "Media", "Nonprofit", "Public Sector", "Biotechnology",
    "Automotive", "Energy", "Real Estate", "Aerospace", "Defence", "Entertainment", "Hospitality",
    "Agriculture", "Mining", "Construction", "Education", "Healthcare", "Government",
    "Property Management", "Facilities Management", "Utilities", "Infrastructure Development",
    "Environmental Services", "Waste Management", "Security Services",
    "Professional Services",
    "Staffing & Recruitment"
]

@st.cache_resource
def load_models():
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
    transformer = models.Transformer("BAAI/bge-large-en-v1.5")
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    embedder = SentenceTransformer(modules=[transformer, pooling])
    embedder.to("cpu")
    return cross_encoder, embedder

cross_encoder, embedder = load_models()

# ─────────────────────────────────────────────────
# DOMAIN DETECTION + SECTOR INFERENCE
# ─────────────────────────────────────────────────
def is_probable_domain(text):
    """Checks if the input text likely contains a domain."""
    match = re.search(r"(https?://)?([a-zA-Z0-9.-]+\.[a-z]{2,})(/[\S]*)?", text)
    return bool(match)

def extract_domain(text):
    """Extracts the domain from a URL or text."""
    match = re.search(r"(https?://)?([a-zA-Z0-9.-]+\.[a-z]{2,})(/[\S]*)?", text)
    if match:
        return match.group(2).lower()
    return ""

def _extract_links_from_page(soup, base_url, max_links=10):
    """Extracts internal links from a BeautifulSoup object, prioritizing navigation."""
    links = set()
    parsed_base_url = urlparse(base_url)
    base_domain = parsed_base_url.netloc

    nav_tags = soup.find_all(['nav'])
    for nav in nav_tags:
        for a in nav.find_all('a', href=True):
            full_url = urljoin(base_url, a['href'])
            parsed_full_url = urlparse(full_url)
            if parsed_full_url.netloc == base_domain and parsed_full_url.path not in ["/", ""]:
                links.add(full_url)
                if len(links) >= max_links:
                    return list(links)

    if len(links) < max_links:
        for a in soup.find_all('a', href=True):
            full_url = urljoin(base_url, a['href'])
            parsed_full_url = urlparse(full_url)
            if parsed_full_url.netloc == base_domain and \
               not parsed_full_url.query and not parsed_full_url.fragment and \
               not re.search(r'\.(pdf|zip|jpg|png|gif|xml|rss)$', parsed_full_url.path.lower()) and \
               parsed_full_url.path not in ["/", ""] and \
               "contact" not in parsed_full_url.path.lower() and \
               "careers" not in parsed_full_url.path.lower():
                links.add(full_url)
                if len(links) >= max_links:
                    break
    return list(links)

async def fetch_and_process_website_content(domain: str) -> str:
    domain = domain.strip().lower()
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]
    base_url = f"https://{domain}"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CVMatcherBot/1.0; +https://yourwebsite.com/bot)"
    }
    content_pool = []
    visited_urls = set()
    urls_to_visit = [base_url]
    crawl_limit = 5

    for current_url in urls_to_visit:
        if current_url in visited_urls or len(visited_urls) > crawl_limit:
            continue
        visited_urls.add(current_url)

        try:
            res = requests.get(current_url, timeout=8, headers=headers)

            if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""):
                continue

            soup = BeautifulSoup(res.text, 'html.parser')

            for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
                script_or_style.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('section')
            text = main_content.get_text(separator=" ", strip=True) if main_content else soup.get_text(separator=" ", strip=True)

            text = re.sub(r'Home\s*About Us\s*Services\s*Contact Us.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'(©\s*\d{4}\s*|All Rights Reserved|Privacy Policy|Terms of Service).*', '', text, flags=re.DOTALL | re.IGNORECASE)

            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            if len(cleaned_text) > 100:
                content_pool.append(cleaned_text[:2000])

            if len(visited_urls) <= crawl_limit:
                new_links = _extract_links_from_page(soup, current_url, max_links=3)
                for link in new_links:
                    if link not in visited_urls and urlparse(link).netloc == domain:
                        urls_to_visit.append(link)

        except requests.exceptions.RequestException:
            pass
        except Exception:
            pass

    full_cleaned_content = " ".join(content_pool)
    if len(full_cleaned_content) > 3000:
        full_cleaned_content = full_cleaned_content[:3000]

    return full_cleaned_content if len(full_cleaned_content) > 100 else ""

async def infer_sector_from_text(text_content: str) -> dict:
    # ALLOWED_SECTORS is now a global constant
    if not text_content or len(text_content.strip()) < 50:
        return {"sector": [], "description": "Insufficient content to classify.", "confidence": 0.0, "rationale": "Not enough textual content."}

    prompt = f"""
You are an expert business analyst. Your task is to precisely classify a company's sector based on its website content.

**Follow these steps:**
1.  **Analyze Core Function:** First, read the text and determine the company's primary business model. What do they *actually do or sell*? (e.g., "They provide consulting services," "They build and sell software," "They operate an e-commerce platform").
2.  **Select Specific Sectors:** Based on this core function, select up to two of the most specific, applicable sectors from the ALLOWED_SECTORS list. For example, a company providing outsourced financial experts is better classified as "Professional Services" or "Consulting" within the "Finance" domain, rather than just "Finance".
3.  **Generate Output:** Create a JSON object with your findings.

**ALLOWED SECTORS:**
{json.dumps(ALLOWED_SECTORS)}

**⚠️ Crucial Rules:**
-   **Evidence-Based:** Your classification MUST be based ONLY on the provided text. Do not invent or assume.
-   **Specificity is Key:** Choose the most granular sector possible. If a company provides services, "Consulting" or "Professional Services" is better than a broad industry name.
-   **Low Confidence Rationale:** ONLY provide a `rationale` if your confidence is below {LLM_SECTOR_CONFIDENCE_THRESHOLD}. The rationale must explain *why* the text was ambiguous or insufficient. Otherwise, `rationale` must be an empty string.

**JSON Output Format:**
{{
  "sector": ["PrimarySector", "SecondarySector"],
  "description": "A neutral, one-sentence summary of the company's core business function.",
  "confidence": 0.9,
  "rationale": "Explain low confidence here, or leave empty."
}}

**Example:**
-   **Text:** "We offer fractional CFOs to help startups scale."
-   **Analysis:** The core function is providing expert services.
-   **Output:** {{"sector": ["Consulting", "Finance"], "description": "Provides fractional CFO services to businesses.", "confidence": 0.95, "rationale": ""}}

**WEBSITE CONTENT TO ANALYZE:**
\"\"\"{text_content.strip()}\"\"\"
""".strip()

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        sector = parsed.get("sector", [])
        if isinstance(sector, str):
            sector = [sector]
        if not isinstance(sector, list):
            sector = []

        valid_sector = [s for s in sector if s in ALLOWED_SECTORS]
        return {
            "sector": valid_sector,
            "description": parsed.get("description", "").strip(),
            "confidence": parsed.get("confidence", 0.0),
            "rationale": parsed.get("rationale", "")
        }
    except Exception:
        return {"sector": [], "description": "Error during GPT-based classification.", "confidence": 0.0, "rationale": "An exception occurred."}

# ─────────────────────────────────────────────────
# QUERY + SKILL EXTRACTION
# ─────────────────────────────────────────────────
async def extract_skills_from_query(query: str) -> list[str]:
    VAGUE_TERMS = {
        "professional background", "experience", "expertise", "background",
        "knowledge", "understanding", "skills", "ability", "competence",
        "capability", "track record", "acumen", "career"
    }

    prompt = f"""
You are an expert CV screener.

Extract only the top 3 to 7 **concrete, specific skills or competencies** from this sentence describing an ideal candidate.
Focus on **clear nouns and actionable phrases** such as "machine learning", "strategic planning", "financial modelling", "client management", "leadership".
Avoid vague, subjective, or generic terms such as "alignment with company needs", "exceptional", "proactive", or phrases that do not represent tangible skills.

Output a **clean JSON list of skills** only.

Candidate requirement:
\"\"\"{query}\"\"\"
"""

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        skills = json.loads(resp.choices[0].message.content)

        filtered_skills = []
        for s in skills:
            if not isinstance(s, str): continue
            s_clean = s.strip().lower()
            if len(s_clean) <= 2: continue
            if not re.search(r"[a-zA-Z]", s_clean): continue
            if s_clean in VAGUE_TERMS: continue
            filtered_skills.append(s_clean)
        return filtered_skills
    except Exception:
        return []

def clean_query(q: str) -> str:
    """Cleans the query by removing generic terms and extra spaces."""
    q = re.sub(r"\b(professionals?|candidates?|individuals?|leaders?|background(?: in)?|experience(?: in)?|qualifications?|competencies?|seeking|looking for)\b", "", q, flags=re.I)
    q = re.sub(r"[\"'“”‘’,.]+", "", q)
    return re.sub(r'\s+', ' ', q.strip())

async def generate_query(user_input: str, sector_hint: list[str] = None, company_desc: str = "") -> str:
    sector_hint = [s for s in sector_hint if isinstance(s, str) and s.strip()] if sector_hint else []
    hints = []
    if sector_hint:
        hints.append(f"The company operates in the {', '.join(sector_hint)} sector(s).")
    if company_desc:
        hints.append(f"The company's core business is: {company_desc}.")
    hint_txt = " ".join(hints) if hints else "No specific company context provided."

    prompt = f"""
You are an expert recruiter. Your goal is to create a highly effective search query for a CV database.

Based on the following user input and company context, generate **ONE concise and specific sentence** that describes the ideal candidate's **required skills, primary responsibilities, and relevant domain expertise**.

Focus on:
- **Actionable skills/competencies**: (e.g., "strategic planning", "data analysis", "software development", "project management").
- **Specific technologies/tools**: (e.g., "Python", "AWS", "SQL", "CRM platforms").
- **Relevant industry domains or business areas**: (e.g., "FinTech operations", "healthcare consulting", "supply chain optimization").
- **Impacts or outcomes**: (e.g., "driving innovation", "improving efficiency", "managing complex projects").

Exclude:
- Generic adjectives (e.g., "excellent", "strong", "proactive").
- Vague phrases (e.g., "proven ability", "well-rounded", "great communicator").
- Redundant information.

Company Context: {hint_txt}

User Input:
\"\"\"{user_input}\"\"\"

Example:
User Input: "Looking for someone good at AI and finance for a bank"
Generated Query: "Candidate requires expertise in artificial intelligence and financial modeling for banking sector operations."

User Input: "Product manager for an EdTech company"
Generated Query: "Candidate skilled in product lifecycle management, market research, and educational technology product development."

Generated Query:
"""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        rewritten = resp.choices[0].message.content.strip()
        if len(rewritten) < 20 or "ideal candidate" in rewritten.lower():
            return clean_query(user_input)
        return clean_query(rewritten)
    except Exception:
        return clean_query(user_input)

# ─────────────────────────────────────────────────
# EMBEDDING + EVIDENCE
# ─────────────────────────────────────────────────
@lru_cache(maxsize=512)
def get_cached_embedding(text: str):
    """Caches embeddings to avoid recomputing for identical texts."""
    return embedder.encode(text, convert_to_tensor=True)

@lru_cache(maxsize=256)
def get_sentence_embeddings(text):
    """Splits text into sentences and embeds them."""
    sentences = sent_tokenize(re.sub(r'\s+', ' ', text))
    filtered = [s for s in sentences if 30 <= len(s) <= 250 and not s.isupper()]
    if not filtered:
        if len(text.strip()) > 50:
            return [text.strip()[:200]], embedder.encode([text.strip()[:200]], convert_to_tensor=True)
        return [], None
    return [], None # Return empty list and None if no valid sentences found to avoid errors


def extract_evidence(query: str, cv_text: str, skills: list[str]) -> tuple[str, str]:
    sentences, sent_embs = get_sentence_embeddings(cv_text)
    if not sentences or sent_embs is None:
        fallback_text = cv_text.strip()
        if len(fallback_text) > 300:
            return f"No direct sentence evidence found. General CV preview:\n• {fallback_text[:300]}...", "none_partial"
        elif len(fallback_text) > 50:
            return f"No direct sentence evidence found. General CV preview:\n• {fallback_text}...", "none_partial"
        return "No meaningful content found in CV for evidence.", "none_empty"

    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, sent_embs)[0].cpu().numpy().tolist()

    candidates = [(sc, s) for sc, s in zip(scores, sentences) if sc >= 0.45]
    if candidates:
        source = "query"
        top = sorted(candidates, key=lambda x: x[0], reverse=True)
        deduped = []
        seen_phrases = set()
        for _, s in top:
            if s not in seen_phrases and all(SequenceMatcher(None, s, existing_s).ratio() < 0.9 for existing_s in seen_phrases):
                deduped.append(s)
                seen_phrases.add(s)
            if len(deduped) >= MAX_EVIDENCE:
                break
        bullets = [f"• {s if s.endswith('.') else s + '.'}" for s in deduped]
        return "\n".join(bullets), source

    skill_matches = []
    for skill in skills:
        keywords = re.findall(r'\w+', skill.lower())
        matched = [s for s in sentences if any(kw in s.lower() for kw in keywords if len(kw) > 2)]
        for m in matched:
            skill_matches.append((skill, m))

    if skill_matches:
        source = "skills"
        seen_sentences = set()
        bullets = []
        skill_matches_sorted = sorted(skill_matches, key=lambda x: len(x[1]), reverse=True)
        for skill, sent in skill_matches_sorted:
            if len(bullets) >= MAX_EVIDENCE:
                break
            if sent not in seen_sentences:
                bullets.append(f"• ({skill.capitalize()}) {sent.strip()}")
                seen_sentences.add(sent)
        return "\n".join(bullets), source

    fallback_text = cv_text.strip()
    if len(fallback_text) > 300:
        return f"No specific evidence found. General CV preview:\n• {fallback_text[:300]}...", "none_partial"
    elif len(fallback_text) > 50:
        return f"No specific evidence found. General CV preview:\n• {fallback_text}...", "none_partial"
    return "No meaningful content found in CV for evidence.", "none_empty"

async def extract_detailed_candidate_profile(client, cv_text: str, required_skills: list[str]) -> dict:
    cv_text_truncated = cv_text[:5000]

    prompt = f"""
You are a meticulous and literal HR data extraction engine. Your task is to parse the provided CV text and extract specific data points into a structured JSON format. **Adhere STRICTLY to the instructions. Do not invent or assume, but you may infer if the context strongly implies information not explicitly stated.**

**Required Skills for Context:** {', '.join(required_skills) if required_skills else "General relevant skills"}

**Candidate CV Text:**
\"\"\"{cv_text_truncated}\"\"\"

**JSON EXTRACTION INSTRUCTIONS:**
{{
    "total_years_experience": "Calculate the total professional work experience in years. Sum the durations of all listed roles (e.g., '2015 – 2020' is 5 years, 'Jan 2020 - Jun 2022' is 2.5 years). If start/end dates are not present for all roles, estimate conservatively from the earliest to latest date mentioned. Provide a single number (e.g., 12.5). If not possible to determine, return 0.0.",
    "relevant_experience_years": "Calculate the total years of experience from roles where the responsibilities are **strongly aligned or directly related** to the **Required Skills for Context** or the overall domain/sector implied by the skills. Sum the durations for only these relevant roles. If a role from 2010-2020 is only 50% relevant, count it as 5 years. Conservatively estimate if precise dates or explicit relevance percentages are not available. Return a single number (e.g., 8.0). If no roles are relevant, return 0.0.",
    "key_achievements_summary": [
        "Extract 2-3 of the most impactful, QUANTIFIABLE achievements from the CV. These should be direct quotes or tight summaries containing numbers, percentages, or clear business outcomes (e.g., 'Managed a $5M budget', 'Increased user engagement by 25%'). Prioritize quantifiable results. If no such achievements are found, return an empty list []."
    ],
    "seniority_indicators": [
        "STRICTLY extract the specific job titles from the CV text that denote seniority (e.g., 'Senior Product Manager', 'Chief Technology Officer', 'Head of Analytics', 'Director', 'VP'). DO NOT list generic keywords like 'Lead' or 'Manager' unless they are part of an actual senior title in the text. If no senior titles are found, return an empty list []."
    ],
    "technologies_used": [
        "List specific technologies, software, or programming languages explicitly mentioned in the CV (e.g., 'Python', 'Tableau', 'AWS', 'SAP'). Additionally, infer commonly associated tools or languages if a specific skill or project strongly implies their use (e.g., 'machine learning' implies 'Python', 'TensorFlow', 'PyTorch'). If none are mentioned or implied, return an empty list []."
    ],
    "problem_solving_evidence": "Find and quote ONE concise sentence from the CV that demonstrates a specific problem the candidate solved or an innovative solution they implemented. Avoid generic mission statements. If no direct evidence is found, return the string 'No specific evidence found.'"
}}
"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed_profile = json.loads(content)

        # --- Robust Post-processing ---
        for key in ["key_achievements_summary", "seniority_indicators", "technologies_used"]:
            if not isinstance(parsed_profile.get(key), list):
                parsed_profile[key] = []
            else: # Clean up empty strings or non-strings from lists
                parsed_profile[key] = [item for item in parsed_profile[key] if isinstance(item, str) and item.strip()]


        for key in ["total_years_experience", "relevant_experience_years"]:
            val = parsed_profile.get(key, 0)
            if not isinstance(val, (int, float)):
                # Attempt to extract number if model fails to return a clean one
                num_search = re.search(r'\d+\.?\d*', str(val))
                parsed_profile[key] = float(num_search.group()) if num_search else 0.0
            else:
                 parsed_profile[key] = float(val) # Ensure float for consistency


        if not isinstance(parsed_profile.get("problem_solving_evidence"), str) or not parsed_profile.get("problem_solving_evidence").strip() or "no specific evidence" in parsed_profile.get("problem_solving_evidence", "").lower():
             parsed_profile["problem_solving_evidence"] = "No specific evidence found."

        return parsed_profile
    except Exception:
        # Fallback for any parsing or API error
        return {
            "total_years_experience": 0.0,
            "relevant_experience_years": 0.0,
            "key_achievements_summary": [],
            "seniority_indicators": [],
            "technologies_used": [],
            "problem_solving_evidence": "No specific evidence found."
        }


# ─────────────────────────────────────────────────
# SCORING, LOGGING & UI
# ─────────────────────────────────────────────────
def fuzzy_match(skill: str, meta_skills: list[str]) -> bool:
    """Performs a fuzzy match between a skill and a list of metadata skills."""
    return any(SequenceMatcher(None, skill.lower(), ms.lower()).ratio() > FUZZY_THRESHOLD for ms in meta_skills)

async def explain_match_gpt_async(client, query: str, evidence: str, detailed_profile: dict) -> str:
    achievements_text = "\n".join([f"- {item}" for item in detailed_profile.get("key_achievements_summary", []) if item]) or "None specified."
    seniority_text = ", ".join([item for item in detailed_profile.get("seniority_indicators", []) if item]) or "None specified."
    technologies_text = ", ".join([item for item in detailed_profile.get("technologies_used", []) if item]) or "None specified."
    problem_solving_text = detailed_profile.get('problem_solving_evidence', 'No specific evidence found.')

    prompt = f"""
You are a fair and neutral hiring analyst.

Based on the REQUIREMENT sentence, the provided EVIDENCE, and the CANDIDATE PROFILE details:

**REQUIREMENT:** "{query}"

**EVIDENCE (Directly from CV):**
{evidence}

**CANDIDATE PROFILE HIGHLIGHTS:**
- Total Years Experience: {detailed_profile.get('total_years_experience', 'N/A')} years
- Relevant Experience Years: {detailed_profile.get('relevant_experience_years', 'N/A')} years
- Key Achievements:
{achievements_text}
- Seniority Indicators: {seniority_text}
- Technologies Used: {technologies_text}
- Problem-Solving Evidence: {problem_solving_text}

Your task:
- Write **3 concise, factual bullet points** explaining why this person fits the requirement.
- **Strictly use ONLY the provided EVIDENCE and CANDIDATE PROFILE HIGHLIGHTS**. Do NOT invent or assume any information not explicitly shown.
- Directly address aspects of the REQUIREMENT.
- Avoid subjective praise like "excellent", "strong", "clearly".
- Use neutral, objective language.

Format:
- Bullet 1
- Bullet 2
- Bullet 3
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate detailed explanation due to an error: {e}"


def semantic_skill_match(skill: str, meta_skills: list[str], threshold: float = 0.7) -> bool:
    """Performs a semantic similarity check between a skill and a list of metadata skills."""
    if not skill or not meta_skills:
        return False
    try:
        skill_emb = embedder.encode(skill, convert_to_tensor=True)
        valid_meta_skills = [s for s in meta_skills if isinstance(s, str) and s.strip()]
        if not valid_meta_skills:
            return False
        
        meta_embs = embedder.encode(valid_meta_skills, convert_to_tensor=True)
        sims = util.cos_sim(skill_emb, meta_embs)[0]
        return any(s.item() > threshold for s in sims)
    except Exception:
        return False

def calculate_quality_score(
    profile: dict,
    req_skills: list[str],
    query: str,
    pine_score: float,
    cross_score: float,
    evidence_src: str,
    meta_match: bool
) -> float:
    """
    Calculates a comprehensive quality score. This version is recalibrated to
    better reward experienced candidates and tangible achievements with robust fallbacks.
    """
    score = 0.0

    # 1. Core Relevance (Pinecone + Cross-Encoder)
    score += pine_score * PINE_WEIGHT_FINAL_SCORE_CONTRIBUTION
    # Scale cross_score from [0,1] to contribute fully to MAX_POINTS after MIN_CONTRIBUTION
    cross_encoder_scaled_contribution = (cross_score - CROSS_ENCODER_MIN_CONTRIBUTION) / (1.0 - CROSS_ENCODER_MIN_CONTRIBUTION) if cross_score >= CROSS_ENCODER_MIN_CONTRIBUTION else 0
    score += max(0.0, cross_encoder_scaled_contribution) * CROSS_ENCODER_MAX_POINTS

    # 2. Skill Coverage (Increased impact)
    techs = [t.lower() for t in profile.get("technologies_used", []) if isinstance(t, str)]
    if req_skills and techs:
        found_count = 0
        for req_skill in req_skills:
            # Check for direct keyword match or semantic match
            if any(re.search(r'\b' + re.escape(req_skill.lower()) + r'\b', tech) for tech in techs) or \
               semantic_skill_match(req_skill, techs):
                found_count += 1
        skill_coverage_ratio = found_count / len(req_skills) if len(req_skills) > 0 else 0
        score += skill_coverage_ratio * SKILL_COVERAGE_MAX_POINTS

    # 3. Relevant Experience (Heavily weighted with robust fallbacks and better scaling)
    exp_points = 0.0
    relevant_years = float(profile.get("relevant_experience_years", 0.0))
    total_years = float(profile.get("total_years_experience", 0.0))

    if relevant_years > 0:
        # More granular and aggressive scaling for relevant years
        if relevant_years >= 15: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS
        elif relevant_years >= 10: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.95 # Increased from 0.9
        elif relevant_years >= 7: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.8  # Increased from 0.75
        elif relevant_years >= 4: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.65 # Increased from 0.6
        elif relevant_years >= 1: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.4  # Increased from 0.35
    elif total_years > 0: # Fallback to total years if relevant is zero/not found
        # Apply a reduced weight for total years, as it's less specific
        if total_years >= 20: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.8
        elif total_years >= 12: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.7
        elif total_years >= 7: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.6
        elif total_years >= 4: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.4
        else: exp_points = RELEVANT_EXPERIENCE_MAX_POINTS * 0.2
    
    score += exp_points

    # 4. Achievements & Impact (Increased impact)
    achievements = [a for a in profile.get("key_achievements_summary", []) if a]
    if achievements:
        # Reward more for multiple strong achievements
        score += min(len(achievements) * (ACHIEVEMENT_MAX_POINTS / 3), ACHIEVEMENT_MAX_POINTS)

    # 5. Seniority Alignment (Stronger reward)
    seniority_kws_in_query = [sk for sk in ["lead", "senior", "manager", "director", "head", "architect", "principal", "vp", "chief"] if sk in query.lower()]
    if seniority_kws_in_query and profile.get("seniority_indicators"):
        # Check for direct match of query seniority keywords in extracted titles
        if any(any(kw in s.lower() for kw in seniority_kws_in_query) for s in profile.get("seniority_indicators", [])):
            score += SENIORITY_MAX_POINTS
        # Also give a smaller bonus if general senior titles are present, even if not explicitly in query
        elif any(s.lower() for s in profile.get("seniority_indicators", [])): # any senior title
             score += SENIORITY_MAX_POINTS * 0.5 # Smaller bonus

    # 6. Problem Solving & Bonuses
    if profile.get("problem_solving_evidence") and "no specific evidence" not in profile.get("problem_solving_evidence", "").lower():
        score += PROBLEM_SOLVING_MAX_POINTS
    
    if meta_match: score += METADATA_SKILL_BONUS
    
    if evidence_src == "query": score += EVIDENCE_SOURCE_BONUS_QUERY
    elif evidence_src == "skills": score += EVIDENCE_SOURCE_BONUS_SKILLS
    elif evidence_src.startswith("none"): score += EVIDENCE_SOURCE_PENALTY_NONE
    
    return round(max(0.0, min(score, 10.0)), 2)


async def rerank_and_score(query: str, matches: list[dict], skills: list[str]) -> list[dict]:
    results_data = []
    
    cv_texts = [m["metadata"].get("text", "") for m in matches]
    cross_encoder_pairs = [(query, t[:2000]) for t in cv_texts]
    
    profile_extraction_tasks = [
        extract_detailed_candidate_profile(openai_client, text, skills) for text in cv_texts
    ]

    cross_probs = []
    if cross_encoder_pairs:
        try:
            cross_raw = cross_encoder.predict(cross_encoder_pairs)
            cross_probs = expit(cross_raw).tolist()
        except Exception:
            cross_probs = [0.0] * len(cross_encoder_pairs)
    
    detailed_profiles_batch = await asyncio.gather(*profile_extraction_tasks)

    explanation_prompts_to_run = []
    
    for i, m in enumerate(matches):
        cv_text = cv_texts[i]
        pine_score = m.get("score", 0)
        cross_score = float(cross_probs[i]) if i < len(cross_probs) else 0.0
        detailed_profile = detailed_profiles_batch[i]
        detailed_profile['name'] = m["metadata"].get("name", "Unnamed")

        evidence, source = extract_evidence(query, cv_text, skills)
        
        explanation_prompts_to_run.append((query, evidence, detailed_profile))

        meta_skills = m["metadata"].get("skills", [])
        # Check both fuzzy and semantic for meta match bonus
        matched_fuzzy = any(fuzzy_match(skill, meta_skills) for skill in skills)
        matched_semantic = any(semantic_skill_match(skill, meta_skills) for skill in skills)
        matched_meta = matched_fuzzy or matched_semantic

        final_score = calculate_quality_score(
            detailed_profile, skills, query, pine_score, cross_score, source, matched_meta
        )
        results_data.append({
            "name": m["metadata"].get("name", "Unnamed"),
            "cv_link": m["metadata"].get("cv_link", ""),
            "score": final_score,
            "evidence": evidence,
            "evidence_source": source,
            "detailed_profile": detailed_profile,
            "explanation_query": (query, evidence, detailed_profile)
        })

    explanation_tasks = [
        explain_match_gpt_async(openai_client, *r.pop("explanation_query")) for r in results_data
    ]
    explanations_batch = await asyncio.gather(*explanation_tasks)
    
    for i, r in enumerate(results_data):
        r["explanation"] = explanations_batch[i]
        r["score_label"] = score_label(r["score"])

    return sorted(results_data, key=lambda x: x["score"], reverse=True)


def score_label(score):
    """Assigns a human-readable label to the score."""
    if score >= 9.0: return "✅ Excellent Match"
    elif score >= 7.5: return "🟢 Strong Fit"
    elif score >= 6.0: return "🟡 Moderate Fit"
    return "🔴 Weak Fit"


# ─────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────
st.set_page_config(page_title="CV Matcher", layout="centered")
st.title("CV Matcher")

user_input = st.text_input("Enter your target (e.g., 'data scientist', 'FinTech', 'www.example.com'):", placeholder="e.g. data scientist, FinTech, www.example.com")
if not user_input:
    st.info("Please enter a target to start matching.")
    st.stop()

# Initialize session state variables if they don't exist
if 'manual_sector_input' not in st.session_state:
    st.session_state.manual_sector_input = ""
if 'use_manual_sector' not in st.session_state:
    st.session_state.use_manual_sector = False
if 'inferred_sector_info' not in st.session_state:
    st.session_state.inferred_sector_info = {"sector": [], "description": "", "confidence": 0.0, "rationale": ""}
if 'processed_about_text' not in st.session_state:
    st.session_state.processed_about_text = ""
if 'sector_determined' not in st.session_state:
    st.session_state.sector_determined = False
# New state variable for input type choice
if 'input_type_choice' not in st.session_state:
    st.session_state.input_type_choice = None


async def main():
    st.markdown("---")
    st.markdown("## ⚙️ Target Analysis")

    user_input_normalized = user_input.strip()

    is_url_input = is_probable_domain(user_input_normalized)
    is_direct_sector_input = user_input_normalized in ALLOWED_SECTORS

    if is_direct_sector_input:
        st.session_state.inferred_sector_info = {
            "sector": [user_input_normalized],
            "description": f"Directly provided sector: {user_input_normalized}",
            "confidence": 1.0,
            "rationale": ""
        }
        st.session_state.use_manual_sector = True
        st.session_state.sector_determined = True
        st.success(f"**Direct Sector Input:** `{user_input_normalized}` - Proceeding with this sector.")
        st.markdown("---") # Add a separator

    elif is_url_input:
        st.markdown("### 🌐 Website Analysis")
        domain = extract_domain(user_input_normalized)
        st.info(f"Analyzing website: `{domain}`")

        with st.spinner("Fetching and processing website content..."):
            st.session_state.processed_about_text = await fetch_and_process_website_content(domain)

        if st.session_state.processed_about_text:
            with st.expander("View Processed Website Content"):
                st.code(st.session_state.processed_about_text)

            with st.spinner("Inferring company sector from website content..."):
                st.session_state.inferred_sector_info = await infer_sector_from_text(st.session_state.processed_about_text)

            st.markdown("**Inferred Company Sector:**")
            st.json(st.session_state.inferred_sector_info)

            if not st.session_state.inferred_sector_info.get("sector") or \
               st.session_state.inferred_sector_info.get("confidence", 0.0) < LLM_SECTOR_CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Sector inference confidence is low or no specific sector was found. Consider overriding.")
                if st.session_state.inferred_sector_info.get("rationale"):
                    st.info(f"**Reason for low confidence:** {st.session_state.inferred_sector_info['rationale']}")
                
                # Manual override checkbox for URL/Company Description
                st.session_state.use_manual_sector = st.checkbox("Override with manual sector input?", value=st.session_state.use_manual_sector)
                
                if st.session_state.use_manual_sector:
                    st.session_state.manual_sector_input = st.text_input("Enter the correct sector(s) (comma-separated):", value=st.session_state.manual_sector_input)
                    if st.session_state.manual_sector_input:
                        st.session_state.inferred_sector_info["sector"] = [s.strip() for s in st.session_state.manual_sector_input.split(',')]
                        st.session_state.inferred_sector_info["description"] = f"Manually provided sector: {st.session_state.manual_sector_input}"
                        st.session_state.inferred_sector_info["confidence"] = 1.0
                        st.session_state.sector_determined = True
                    else:
                        st.session_state.sector_determined = False # Requires manual input
                else:
                    st.session_state.sector_determined = False # Inference was low/bad, and no manual override chosen
                    st.warning("No confident sector determined from URL, and no manual override. Cannot proceed with sector filtering.")
            else:
                st.session_state.sector_determined = True # Sector confidently inferred
            
            st.markdown("---") # Add a separator

        else:
            st.error("Could not extract enough meaningful content from the website. Please provide a more descriptive input or manually define the sector.")
            st.session_state.sector_determined = False # Cannot proceed without content/sector
            st.session_state.use_manual_sector = st.checkbox("Enter sector manually to proceed?", value=True)
            if st.session_state.use_manual_sector:
                st.session_state.manual_sector_input = st.text_input("Enter the correct sector(s) (comma-separated):", value=st.session_state.manual_sector_input)
                if st.session_state.manual_sector_input:
                    st.session_state.inferred_sector_info["sector"] = [s.strip() for s in st.session_state.manual_sector_input.split(',')]
                    st.session_state.inferred_sector_info["description"] = f"Manually provided sector: {st.session_state.manual_sector_input}"
                    st.session_state.inferred_sector_info["confidence"] = 1.0
                    st.session_state.sector_determined = True
                else:
                    st.session_state.sector_determined = False # Still requires input
            st.markdown("---") # Add a separator

    else: # Other text input (could be skill/role or company description)
        st.markdown("### 📝 Text Input Analysis")
        # Ask user to clarify intent for generic text input
        st.session_state.input_type_choice = st.radio(
            "What does this text represent?",
            ("Skill/Role Requirement (e.g., 'Machine Learning Engineer')", "Company Description for Sector Inference (e.g., 'A FinTech startup focusing on blockchain')"),
            key="input_type_radio"
        )
        
        if st.session_state.input_type_choice == "Skill/Role Requirement (e.g., 'Machine Learning Engineer')":
            st.session_state.inferred_sector_info = {
                "sector": [],
                "description": "User specified skill/role requirement, no sector inference needed.",
                "confidence": 0.0, # Indicate no sector confidence as it's not a company
                "rationale": "Input treated as direct skill/role query."
            }
            st.session_state.sector_determined = True # Can proceed, no sector filtering needed
            st.info("Proceeding with direct skill/role matching. Sector filtering will be skipped.")
            st.markdown("---") # Add a separator
        
        elif st.session_state.input_type_choice == "Company Description for Sector Inference (e.g., 'A FinTech startup focusing on blockchain')":
            st.session_state.processed_about_text = user_input_normalized # Use user_input directly for inference
            with st.spinner("Inferring company sector from text input..."):
                st.session_state.inferred_sector_info = await infer_sector_from_text(st.session_state.processed_about_text)

            st.markdown("**Inferred Company Sector:**")
            st.json(st.session_state.inferred_sector_info)

            if not st.session_state.inferred_sector_info.get("sector") or \
               st.session_state.inferred_sector_info.get("confidence", 0.0) < LLM_SECTOR_CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Sector inference confidence is low or no specific sector was found. Consider overriding.")
                if st.session_state.inferred_sector_info.get("rationale"):
                    st.info(f"**Reason for low confidence:** {st.session_state.inferred_sector_info['rationale']}")
                
                # Manual override checkbox for Company Description
                st.session_state.use_manual_sector = st.checkbox("Enter sector manually?", value=True) # Default to true as inference was low
                
                if st.session_state.use_manual_sector:
                    st.session_state.manual_sector_input = st.text_input("Enter the correct sector(s) (comma-separated):", value=st.session_state.manual_sector_input)
                    if st.session_state.manual_sector_input:
                        st.session_state.inferred_sector_info["sector"] = [s.strip() for s in st.session_state.manual_sector_input.split(',')]
                        st.session_state.inferred_sector_info["description"] = f"Manually provided sector: {st.session_state.manual_sector_input}"
                        st.session_state.inferred_sector_info["confidence"] = 1.0
                        st.session_state.sector_determined = True
                    else:
                        st.session_state.sector_determined = False # Requires manual input
                else:
                    st.session_state.sector_determined = False # Inference was low/bad, and no manual override chosen
                    st.warning("No confident sector determined from text, and no manual override. Cannot proceed with sector filtering.")
            else:
                st.session_state.sector_determined = True # Sector confidently inferred
            st.markdown("---") # Add a separator
        
        else: # If radio button is not yet selected
            st.session_state.sector_determined = False
            st.info("Please select how to interpret your text input to proceed.")
            st.stop()


    # Final check before proceeding to search
    # If the user chose "Skill/Role Requirement", sector_determined is True and no sector filtering is applied.
    # If URL or Company Description, sector_determined must be True with a valid sector.
    if not st.session_state.sector_determined:
        st.warning("A sector could not be reliably determined or provided. Please ensure a valid URL, direct sector, or manual sector input is set to proceed with optimal filtering. If this is a pure skill query, select 'Skill/Role Requirement'.")
        st.stop() # This stops if required sector info is missing for sector-dependent paths

    current_sector = st.session_state.inferred_sector_info.get("sector", [])
    current_company_desc = st.session_state.inferred_sector_info.get("description", "")

    st.markdown("## 🔍 Search & Match")

    with st.spinner("Rewriting query and extracting skills..."):
        try:
            # If it's a skill-only input, the query generation should focus on that
            if st.session_state.input_type_choice == "Skill/Role Requirement (e.g., 'Machine Learning Engineer')":
                query = clean_query(user_input_normalized) # Use original user input as query directly
                skills_extracted = await extract_skills_from_query(query)
                st.info("Using original input as query for direct skill matching.")
            else:
                query = await generate_query(user_input_normalized, current_sector, current_company_desc)
                skills_extracted = await extract_skills_from_query(query)
        except Exception as e:
            st.error(f"Query generation or skill extraction error: {e}")
            query = clean_query(user_input_normalized)
            skills_extracted = []

    st.markdown(f"**Search Query:** `{query}`")
    st.markdown(f"**Skills extracted:** `{', '.join(skills_extracted) if skills_extracted else 'None found. (This may impact scoring for skill-based roles)'}`")

    query_vec = get_cached_embedding(query)
    skills_vec = get_cached_embedding(", ".join(skills_extracted)) if skills_extracted else None

    if skills_vec is not None and len(skills_extracted) > 0:
        q_vec = 0.7 * query_vec + 0.3 * skills_vec
    else:
        q_vec = query_vec

    with st.spinner("Searching Pinecone database for candidates..."):
        try:
            base_filter = {"tier": {"$eq": "A"}}
            
            # Apply sector filter only if input type is NOT "Skill/Role Requirement" AND a confident sector exists
            if (st.session_state.input_type_choice != "Skill/Role Requirement (e.g., 'Machine Learning Engineer')" and
                current_sector and
                (st.session_state.inferred_sector_info.get("confidence", 0.0) >= LLM_SECTOR_CONFIDENCE_THRESHOLD or st.session_state.use_manual_sector)):
                
                base_filter["sectors"] = {"$in": current_sector}
                st.info(f"**Filtering by sector(s):** `{', '.join(current_sector)}`")
            elif st.session_state.input_type_choice == "Skill/Role Requirement (e.g., 'Machine Learning Engineer')":
                st.info("Skipping sector filter as input is a skill/role requirement.")
            else:
                st.warning("⚠️ No confident sector detected or manually provided for filtering. Running unfiltered Tier-A match (may be less precise).")
            
            resp = index.query(
                vector=q_vec.tolist(),
                top_k=TOP_K,
                include_metadata=True,
                filter=base_filter
            )

            matches = resp.get("matches", [])
            for m in matches:
                # Normalizing Pinecone score from [-1, 1] to [0, 1]
                m["score"] = max(0.0, min((m.get("score", 0.0) + 1) / 2, 1.0))

        except Exception as e:
            st.error(f"Pinecone search error: {e}. Please check API keys and index status.")
            st.stop()

    if not matches:
        st.warning("No relevant candidates found in the database matching your criteria. Try a different query or adjust filters.")
        st.stop()

    st.markdown(f"Found {len(matches)} potential candidates. Reranking for accuracy...")
    results = await rerank_and_score(query, matches, skills_extracted)

    st.markdown("---")
    st.markdown("## 📊 Match Results")

    if not results:
        st.info("No matches to display after reranking.")
    else:
        for i, r in enumerate(results, 1):
            st.markdown(f"### Match {i}: {r['name']} — {r['score']}/10  {r['score_label']}")
            st.write(f"**CV**: [View]({r['cv_link']})")
            
            with st.container(border=True):
                st.markdown("**Why This Matches (GPT‑4o):**")
                st.markdown(r["explanation"])
            
            with st.expander("Show Extracted Evidence & Details"):
                st.markdown("**Raw Evidence Extracted:**")
                st.markdown(r["evidence"])
                st.markdown(f"**Evidence Source:** `{r['evidence_source']}`")
                st.markdown("**Detailed Candidate Profile (LLM Extraction):**")
                st.json(r["detailed_profile"])
            st.markdown("---")


if __name__ == "__main__":
    if user_input:
        asyncio.run(main())