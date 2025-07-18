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

# --- SCORING WEIGHTS RE-CALIBRATED FOR HIGHER SCORES ---
PINE_WEIGHT_FINAL_SCORE_CONTRIBUTION = 0.20  # Max 2.0 points
CROSS_ENCODER_MAX_POINTS = 1.5             # Max 1.5 points - Increased
SKILL_COVERAGE_MAX_POINTS = 1.5            # Max 1.5 points
RELEVANT_EXPERIENCE_MAX_POINTS = 3.5       # Max 3.5 points - HEAVILY INCREASED
ACHIEVEMENT_MAX_POINTS = 1.0               # Max 1.0 point
SENIORITY_MAX_POINTS = 0.5                 # Max 0.5 points
PROBLEM_SOLVING_MAX_POINTS = 0.5           # Max 0.5 points
METADATA_SKILL_BONUS = 0.25
EVIDENCE_SOURCE_BONUS_QUERY = 0.1
EVIDENCE_SOURCE_BONUS_SKILLS = 0.05
EVIDENCE_SOURCE_PENALTY_NONE = -0.2

CROSS_ENCODER_MIN_CONTRIBUTION = 0.5
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
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', download_dir="nltk_data")
nltk.data.path.append("nltk_data")
from nltk.tokenize import sent_tokenize


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
    ALLOWED_SECTORS = [
        "Marketing Agency", "Market Research", "FinTech", "Retail", "HealthTech", "EdTech", "LegalTech",
        "SaaS", "Consulting", "Analytics", "AI", "Telecommunications", "Logistics", "Cybersecurity",
        "HRTech", "Manufacturing", "Finance", "Media", "Nonprofit", "Public Sector", "Biotechnology",
        "Automotive", "Energy", "Real Estate", "Aerospace", "Defence", "Entertainment", "Hospitality",
        "Agriculture", "Mining", "Construction", "Education", "Healthcare", "Government",
        "Property Management", "Facilities Management", "Utilities", "Infrastructure Development",
        "Environmental Services", "Waste Management", "Security Services",
        "Professional Services",
        "Staffing & Recruitment"
    ]

    if not text_content or len(text_content.strip()) < 50:
        return {"sector": [], "description": "Insufficient content to classify.", "confidence": 0.0, "rationale": "Not enough textual content."}

    prompt = f"""
You are a neutral, highly precise, and cautious business analyst.

Your task is to classify the company based ONLY on its actual core business function as described in the collected website text. Consider if the company's primary model is providing expertise (e.g., consulting, staffing) or manufacturing/selling a product.

ALLOWED SECTORS:
{json.dumps(ALLOWED_SECTORS)}

⚠️ Important Rules:
- Only use information explicitly present in the text.
- Focus on the company's direct, primary activities.
- Identify a closely related **secondary sector** ONLY if it's explicitly and strongly implied, distinct, and represents a significant part of their business beyond their primary function.
    - Example: If a "Construction" company clearly describes extensive "Facilities Management" services for buildings they construct, list both. If they just build, only "Construction".
- For service-based companies, determine if their core business is providing expertise (e.g., "Consulting", "Professional Services") or facilitating talent placement ("Staffing & Recruitment").
- DO NOT infer or assume missing details or apply overly broad categories if a more specific one applies.
- If the text is vague, ambiguous, or if no clear sector fits from ALLOWED_SECTORS, return an empty sector list and a specific explanation.
- Provide a confidence score (0.0 to 1.0) for your classification. If confidence is below {LLM_SECTOR_CONFIDENCE_THRESHOLD}, explain why it's uncertain in the 'rationale' field.

Return a JSON object with:
• sector: list of up to 2 most specific sectors (from ALLOWED_SECTORS only)
• description: 1 neutral sentence stating what the company does, focusing on its core services.
• confidence: float (0.0 to 1.0) indicating certainty of classification.
• rationale (optional): If confidence is low, a brief explanation for the uncertainty.

Format:
{{
  "sector": ["Sector1", "Sector2"],
  "description": "One-sentence summary of the company’s function.",
  "confidence": 0.95,
  "rationale": "If confidence is low, explain why."
}}

WEBSITE CONTENT:
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
    return filtered, embedder.encode(filtered, convert_to_tensor=True)


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
You are an expert HR analyst. Your task is to extract a structured profile from the candidate's CV text based on the requested skills.
Be precise, neutral, and only use information explicitly stated or strongly implied by the text.
Quantify achievements where possible. If information is not found, use "N/A" for text fields or "0" for numerical values. For lists, use "None found.".

**Required Skills (focus extraction around these):** {', '.join(required_skills) if required_skills else "General relevant skills"}

**Candidate CV Text:**
\"\"\"{cv_text_truncated}\"\"\"

Extract the following in JSON format:
{{
    "total_years_experience": "Estimate overall professional experience in years (e.g., '15 years', '5 years', 'N/A').",
    "relevant_experience_years": "Estimate years of experience directly related to the Required Skills or similar domains (e.g., '10 years', 'N/A').",
    "key_achievements_summary": ["List 2-3 concise, quantifiable accomplishments or significant impacts related to the Required Skills. Focus on results. If none, state 'None found.'"],
    "seniority_indicators": ["List keywords indicating seniority/level (e.g., 'Lead', 'Senior', 'Manager', 'Director', 'Principal', 'CTO'). If none, state 'None found.'"],
    "technologies_used": ["List 3-5 specific technologies, tools, or programming languages relevant to the Required Skills (e.g., 'Python', 'AWS', 'TensorFlow', 'SQL', 'Salesforce'). If none, state 'None found.'"],
    "problem_solving_evidence": "One concise sentence describing how the candidate demonstrated problem-solving, innovation, or overcame a significant challenge. Focus on the action and outcome. If none, state 'None found.'"
}}
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        parsed_profile = json.loads(content)

        for key in ["key_achievements_summary", "seniority_indicators", "technologies_used"]:
            if not isinstance(parsed_profile.get(key), list):
                if parsed_profile.get(key) == "None found." or parsed_profile.get(key) is None or parsed_profile.get(key) == []:
                    parsed_profile[key] = []
                elif isinstance(parsed_profile.get(key), str):
                    parsed_profile[key] = [item.strip() for item in parsed_profile[key].split(',') if item.strip() != "None found."]
                else:
                    parsed_profile[key] = []

        for key in ["total_years_experience", "relevant_experience_years", "problem_solving_evidence"]:
            if parsed_profile.get(key) is None or parsed_profile.get(key) == "None found." or parsed_profile.get(key) == []:
                parsed_profile[key] = "N/A" if key in ["total_years_experience", "relevant_experience_years"] else "None found."
            parsed_profile[key] = str(parsed_profile[key])

        return parsed_profile
    except Exception:
        return {
            "total_years_experience": "N/A",
            "relevant_experience_years": "N/A",
            "key_achievements_summary": [],
            "seniority_indicators": [],
            "technologies_used": [],
            "problem_solving_evidence": "None found."
        }


# ─────────────────────────────────────────────────
# SCORING, LOGGING & UI
# ─────────────────────────────────────────────────
def fuzzy_match(skill: str, meta_skills: list[str]) -> bool:
    """Performs a fuzzy match between a skill and a list of metadata skills."""
    return any(SequenceMatcher(None, skill.lower(), ms.lower()).ratio() > FUZZY_THRESHOLD for ms in meta_skills)

async def explain_match_gpt_async(client, query: str, evidence: str, detailed_profile: dict) -> str:
    achievements_text = "\n".join([f"- {item}" for item in detailed_profile.get("key_achievements_summary", []) if item and item != "None found."]) or "None found."
    seniority_text = ", ".join([item for item in detailed_profile.get("seniority_indicators", []) if item and item != "None found."]) or "None found."
    technologies_text = ", ".join([item for item in detailed_profile.get("technologies_used", []) if item and item != "None found."]) or "None found."
    problem_solving_text = detailed_profile.get('problem_solving_evidence', 'None found.')

    prompt = f"""
You are a fair and neutral hiring analyst.

Based on the REQUIREMENT sentence, the provided EVIDENCE, and the CANDIDATE PROFILE details:

REQUIREMENT: "{query}"

EVIDENCE (Directly from CV):
{evidence}

CANDIDATE PROFILE HIGHLIGHTS:
- Total Years Experience: {detailed_profile.get('total_years_experience', 'N/A')}
- Relevant Experience Years: {detailed_profile.get('relevant_experience_years', 'N/A')}
- Key Achievements: {achievements_text}
- Seniority Indicators: {seniority_text}
- Technologies Used: {technologies_text}
- Problem-Solving Evidence: {problem_solving_text}

Your task:
- Write **3 concise, factual bullet points** explaining why this person fits the requirement.
- Use **ONLY the provided EVIDENCE and CANDIDATE PROFILE HIGHLIGHTS** – do **not assume** anything not shown.
- Directly address aspects of the REQUIREMENT.
- Avoid vague praise like "excellent", "strong", "clearly".
- Each bullet should highlight a distinct point of alignment or key strength.
- Use neutral, objective language and avoid repetition.
- If an aspect of the requirement is not well-supported by evidence, state that directly but neutrally (e.g., "Limited evidence for specific X was found, focusing on Y instead.").

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
    Calculates a comprehensive quality score. This version is robust against failed experience extraction
    by using seniority indicators as a fallback to infer experience level.
    """
    score = 0.0

    # 1. Core Relevance (Pinecone + Cross-Encoder): Max 3.5 points
    score += pine_score * (PINE_WEIGHT_FINAL_SCORE_CONTRIBUTION * 10)
    cross_encoder_contribution = (cross_score - CROSS_ENCODER_MIN_CONTRIBUTION) / (1.0 - CROSS_ENCODER_MIN_CONTRIBUTION) if cross_score >= CROSS_ENCODER_MIN_CONTRIBUTION else 0
    score += max(0.0, cross_encoder_contribution) * CROSS_ENCODER_MAX_POINTS

    # 2. Skill Coverage: Max 1.5 points
    techs = [t.lower() for t in profile.get("technologies_used", []) if isinstance(t, str)]
    if req_skills and techs:
        found_count = sum(1 for req_skill in req_skills if any(req_skill.lower() in tech for tech in techs) or semantic_skill_match(req_skill, techs))
        skill_coverage_ratio = found_count / len(req_skills) if len(req_skills) > 0 else 0
        score += skill_coverage_ratio * SKILL_COVERAGE_MAX_POINTS

    # 3. Relevant Experience (Tiered with Seniority Fallback): Max 3.5 points
    exp_points = 0.0
    years_str = re.findall(r'\d+\.?\d*', str(profile.get("relevant_experience_years", "0")))
    relevant_years = float(years_str[0]) if years_str else 0

    if relevant_years > 0:
        if relevant_years >= 10: exp_points = 3.5
        elif relevant_years >= 7: exp_points = 3.0
        elif relevant_years >= 4: exp_points = 2.0
        elif relevant_years >= 1: exp_points = 1.0
    else:
        # --- MORE GENEROUS FALLBACK LOGIC ---
        seniority_indicators = [s.lower() for s in profile.get("seniority_indicators", []) if isinstance(s, str)]
        if any(kw in str(seniority_indicators) for kw in ["director", "head", "chief", "vp", "principal", "c-level"]):
            exp_points = 3.25 # Award very high score for top-tier roles
        elif any(kw in str(seniority_indicators) for kw in ["senior", "manager", "lead", "architect"]):
            exp_points = 2.25 # Award strong score for mid-tier roles
    score += exp_points

    # 4. Achievements & Impact: Max 1.0 points
    achievements = [a for a in profile.get("key_achievements_summary", []) if a and a != "None found."]
    achievement_points = 0.0
    if achievements:
        achievement_points = min(len(achievements) * (ACHIEVEMENT_MAX_POINTS / 2.0), ACHIEVEMENT_MAX_POINTS)
        if relevant_years == 0:
            achievement_points = min(achievement_points + 0.5, ACHIEVEMENT_MAX_POINTS)
    score += achievement_points

    # 5. Seniority Alignment: Max 0.5 points
    seniority_kws_in_query = [sk for sk in ["lead", "senior", "manager", "director", "head", "architect", "principal", "vp", "chief"] if sk in query.lower()]
    if seniority_kws_in_query and profile.get("seniority_indicators"):
        if any(any(kw in s.lower() for kw in seniority_kws_in_query) for s in profile.get("seniority_indicators", [])):
            score += SENIORITY_MAX_POINTS
    
    # 6. Problem Solving & Bonuses
    if profile.get("problem_solving_evidence") and profile.get("problem_solving_evidence") != "None found.":
        score += PROBLEM_SOLVING_MAX_POINTS
    if meta_match: score += METADATA_SKILL_BONUS
    if evidence_src == "query": score += EVIDENCE_SOURCE_BONUS_QUERY
    elif evidence_src == "skills": score += EVIDENCE_SOURCE_BONUS_SKILLS
    elif evidence_src.startswith("none"): score += EVIDENCE_SOURCE_PENALTY_NONE
    
    # Final score floor for highly relevant candidates who were penalized by other factors
    if cross_score > 0.9 and score < 5.0:
        score = 5.0

    return round(max(1.0, min(score, 10.0)), 2)


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
        matched_fuzzy = any(fuzzy_match(skill, meta_skills) for skill in skills)
        matched_semantic = any(semantic_skill_match(skill, meta_skills) for skill in skills)
        matched_meta = matched_fuzzy or matched_semantic

        final_score = calculate_quality_score(
            detailed_profile, skills, query, pine_score, cross_score, source, matched_meta
        )
        final_score = final_score +1.5
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
    if score >= 8.5: return "🟢 Perfect Match"
    elif score >= 7.0: return "🟡 Strong Fit"
    elif score >= 5.0: return "🟠 Moderate Fit"
    return "🔴 Weak Fit"


# ─────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────
st.set_page_config(page_title="CV Matcher", layout="centered")
st.title("CV Matcher")

user_input = st.text_input("Target (sector, skill, or URL):", placeholder="e.g. Howdens, LLMs, www.example.com")
if not user_input:
    st.info("Please enter a target to start matching.")
    st.stop()

if 'manual_sector_input' not in st.session_state:
    st.session_state.manual_sector_input = ""
if 'use_manual_sector' not in st.session_state:
    st.session_state.use_manual_sector = False
if 'inferred_sector_info' not in st.session_state:
    st.session_state.inferred_sector_info = {"sector": [], "description": "", "confidence": 0.0, "rationale": ""}
if 'processed_about_text' not in st.session_state:
    st.session_state.processed_about_text = ""

async def main():
    st.markdown("---")
    st.markdown("## ⚙️ Target Analysis")

    is_domain = is_probable_domain(user_input)
    if is_domain:
        domain = extract_domain(user_input)
        st.markdown("### 🧪 Website Analysis")
        st.markdown(f"**Domain:** `{domain}`")

        with st.spinner("Fetching and processing website content... This may take a moment."):
            st.session_state.processed_about_text = await fetch_and_process_website_content(domain)

        if st.session_state.processed_about_text:
            with st.expander("View Processed Website Content"):
                st.code(st.session_state.processed_about_text)

            with st.spinner("Inferring company sector from processed content..."):
                st.session_state.inferred_sector_info = await infer_sector_from_text(st.session_state.processed_about_text)

            st.markdown("**Inferred Company Sector:**")
            st.json(st.session_state.inferred_sector_info)

            if not st.session_state.inferred_sector_info.get("sector") or \
               st.session_state.inferred_sector_info.get("confidence", 0.0) < LLM_SECTOR_CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Sector inference confidence is low or no specific sector was found. Consider overriding.")
                if st.session_state.inferred_sector_info.get("rationale"):
                    st.info(f"**Reason for low confidence:** {st.session_state.inferred_sector_info['rationale']}")
                st.session_state.use_manual_sector = st.checkbox("Override with manual sector input?", value=st.session_state.use_manual_sector)
        else:
            st.warning("Could not extract enough meaningful content from the website. Please provide a more descriptive input or manually define the sector.")
            st.session_state.use_manual_sector = st.checkbox("Enter sector manually?", value=True)

        if st.session_state.use_manual_sector:
            st.session_state.manual_sector_input = st.text_input("Enter the correct sector(s) (comma-separated):", value=st.session_state.manual_sector_input)
            if st.session_state.manual_sector_input:
                st.session_state.inferred_sector_info["sector"] = [s.strip() for s in st.session_state.manual_sector_input.split(',')]
                st.session_state.inferred_sector_info["description"] = f"Manually provided sector: {st.session_state.manual_sector_input}"
                st.session_state.inferred_sector_info["confidence"] = 1.0

    else: # Not a domain, treat as direct text input for sector inference
        st.markdown("### 📝 Text Input Analysis")
        st.session_state.processed_about_text = user_input
        with st.spinner("Inferring company sector from text input..."):
            st.session_state.inferred_sector_info = await infer_sector_from_text(st.session_state.processed_about_text)

        st.markdown("**Inferred Company Sector:**")
        st.json(st.session_state.inferred_sector_info)

        if not st.session_state.inferred_sector_info.get("sector") or \
           st.session_state.inferred_sector_info.get("confidence", 0.0) < LLM_SECTOR_CONFIDENCE_THRESHOLD:
            st.warning("⚠️ Sector inference confidence is low or no specific sector was found. Consider overriding.")
            if st.session_state.inferred_sector_info.get("rationale"):
                st.info(f"**Reason for low confidence:** {st.session_state.inferred_sector_info['rationale']}")
            st.session_state.use_manual_sector = st.checkbox("Override with manual sector input?", value=True)
            if st.session_state.use_manual_sector:
                st.session_state.manual_sector_input = st.text_input("Enter the correct sector(s) (comma-separated):", value=st.session_state.manual_sector_input)
                if st.session_state.manual_sector_input:
                    st.session_state.inferred_sector_info["sector"] = [s.strip() for s in st.session_state.manual_sector_input.split(',')]
                    st.session_state.inferred_sector_info["description"] = f"Manually provided sector: {st.session_state.manual_sector_input}"
                    st.session_state.inferred_sector_info["confidence"] = 1.0

    current_sector = st.session_state.inferred_sector_info.get("sector", [])
    current_company_desc = st.session_state.inferred_sector_info.get("description", "")

    st.markdown("---")
    st.markdown("## 🔍 Search & Match")

    with st.spinner("Rewriting query and extracting skills..."):
        try:
            query = await generate_query(user_input, current_sector, current_company_desc)
            skills_extracted = await extract_skills_from_query(query)
        except Exception as e:
            st.error(f"Query generation or skill extraction error: {e}")
            query = clean_query(user_input)
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
            
            if current_sector and (st.session_state.inferred_sector_info.get("confidence", 0.0) >= LLM_SECTOR_CONFIDENCE_THRESHOLD or st.session_state.use_manual_sector):
                if current_sector:
                    base_filter["sectors"] = {"$in": current_sector}
                    st.markdown(f"**Filtering by sector(s):** `{', '.join(current_sector)}`")
            else:
                st.warning("⚠️ No reliable sector detected or manually provided — running unfiltered Tier-A match (may be less precise).")
            
            resp = index.query(
                vector=q_vec.tolist(),
                top_k=TOP_K,
                include_metadata=True,
                filter=base_filter
            )

            matches = resp.get("matches", [])
            for m in matches:
                m["score"] = max(0.0, min((m.get("score", 0.0) + 1) / 2, 1.0))

        except Exception as e:
            st.error(f"Pinecone search error: {e}. Please check API keys and index status.")
            st.stop()

    if not matches:
        st.warning("No relevant candidates found in the database matching your criteria. Try a different query or adjust sector filters.")
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
                st.markdown("**Why This Matches (GPT‑4):**")
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