
import os
import re
import json
import openai
import streamlit as st
import numpy as np
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
from scipy.special import expit
from dotenv import load_dotenv
from functools import lru_cache
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import requests
import asyncio
import openai
from openai import AsyncOpenAI

import nltk
nltk.data.path.append("nltk_data")
from nltk.tokenize import sent_tokenize
# ─────────────────────────────────────────────────
# CONFIGURATION & HYPERPARAMETERS
# ─────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = "us-east-1"
PINECONE_INDEX     = "head-of-ai"
TOP_K              = 10
PINE_WEIGHT        = 0.6
CROSS_WEIGHT       = 0.4
FINAL_SCALE        = 10.0
MIN_SIM_THRESHOLD  = 0.55
MAX_EVIDENCE       = 3
FUZZY_THRESHOLD    = 0.85

openai.api_key = OPENAI_API_KEY
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def load_models():
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
    transformer = models.Transformer("BAAI/bge-large-en-v1.5")
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    embedder = SentenceTransformer(modules=[transformer, pooling]) 
    embedder.to("cpu")
    return cross_encoder, embedder

cross_encoder, embedder = load_models()

# ─────────────────────────────────────
# DOMAIN DETECTION + SECTOR INFERENCE
# ─────────────────────────────────────
def is_probable_domain(text):
    print("it is in is_probable_domain")
    match = re.search(r"(https?://)?([a-zA-Z0-9.-]+\.[a-z]{2,})(/[\S]*)?", text)
    return bool(match)

def extract_domain(text):
    match = re.search(r"(https?://)?([a-zA-Z0-9.-]+\.[a-z]{2,})(/[\S]*)?", text)
    if match:
        return match.group(2).lower()
    return ""


def fetch_about_text(domain):
    domain = domain.strip().lower()
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]
    domain = f"https://{domain}"

    fallback_paths = ["", "/about", "/about-us", "/company", "/who-we-are", "/our-story"]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CVMatcherBot/1.0)"
    }

    for path in fallback_paths:
        try:
            url = domain.rstrip("/") + path
            print(f"[Trying] {url}")
            res = requests.get(url, timeout=6, headers=headers)

            if res.status_code != 200:
                print(f"[{res.status_code}] Skipped {url}")
                continue

            if "text/html" not in res.headers.get("Content-Type", ""):
                print(f"[Not HTML] {url}")
                continue

            soup = BeautifulSoup(res.text, 'html.parser')

            # Try meta description
            desc = soup.find("meta", attrs={"name": "description"})
            if desc and desc.get("content"):
                print(f"[Meta Description Found] {url}")
                return desc["content"][:1000]

            # Try sections with keywords
            keywords = ["about", "mission", "vision", "what we do", "who we are"]
            blocks = soup.find_all(['p', 'div', 'section', 'article'], limit=50)
            selected = []
            for blk in blocks:
                txt = blk.get_text(strip=True)
                if any(k in txt.lower() for k in keywords) and len(txt) > 60:
                    selected.append(txt)

            if selected:
                print(f"[Keyword Match Found] {url}")
                return " ".join(selected)[:1500]

            # Fallback: full page text
            full_text = soup.get_text(separator=" ", strip=True)
            if len(full_text) > 300:
                print(f"[Fallback Full Text Used] {url}")
                return full_text[:1500]

        except Exception as e:
            print(f"[Fetch Error] {url}: {e}")
            continue


    print(f"[Fetch Fail] No valid content at {domain}")
    return ""

def infer_sector_from_text(about_text: str) -> dict:
    # Extended, distinct, and semantically disambiguated list of sectors
    ALLOWED_SECTORS = [
        "Marketing Agency", "Market Research", "FinTech", "Retail", "HealthTech", "EdTech", "LegalTech",
        "SaaS", "Consulting", "Analytics", "AI", "Telecommunications", "Logistics", "Cybersecurity",
        "HRTech", "Manufacturing", "Finance", "Media", "Nonprofit", "Public Sector"
    ]

    if not about_text or len(about_text.strip()) < 30:
        return {"sector": [], "description": "No meaningful content found to classify."}

    prompt = f"""
You are a neutral and highly precise business analyst.

Your task is to classify the company based ONLY on its actual business function as described in the ABOUT section below.

ALLOWED SECTORS:
{json.dumps(ALLOWED_SECTORS)}

⚠️ Important Rules:
- Only use the information in the ABOUT section.
- Focus on the company's own core activities (e.g. "runs advertising campaigns", "provides analytics services").
- DO NOT label based on who the company helps (e.g. “we help retailers” ≠ Retail).
- DO NOT infer or assume missing details.
- If the description is vague or unclear, return an empty sector list with explanation.

🧠 Disambiguation Example:
- “We create and manage advertising campaigns for B2B firms” → ["Marketing Agency"]
- “We conduct surveys and generate consumer insights for B2B” → ["Market Research"]

Return:
- A JSON object with:
  • sector: list of up to 2 sectors (from ALLOWED_SECTORS only)
  • description: 1 neutral sentence stating what the company does

Format:
{{
  "sector": ["Sector1", "Sector2"],
  "description": "One-sentence summary of the company’s function."
}}

ABOUT:
\"\"\"{about_text.strip()}\"\"\"
""".strip()

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()

        # Strict JSON extraction
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            print("[Sector Parse Error] No JSON object found.")
            return {"sector": [], "description": "Invalid GPT output format."}

        parsed = json.loads(match.group(0))

        sector = parsed.get("sector", [])
        if isinstance(sector, str):
            sector = [sector]
        if not isinstance(sector, list):
            sector = []

        # Enforce validity and prevent hallucinated sectors
        valid_sector = [s for s in sector if s in ALLOWED_SECTORS]
        return {
            "sector": valid_sector,
            "description": parsed.get("description", "").strip()
        }

    except Exception as e:
        print(f"[Sector GPT Error] {e}")
        return {"sector": [], "description": "Error during GPT-based classification."}

# ─────────────────────────────────────
# QUERY + SKILL EXTRACTION
# ─────────────────────────────────────
def extract_skills_from_query(query: str) -> list[str]:
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
        resp = openai.chat.completions.create(
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

    except Exception as e:
        print(f"[Skill Extraction Error] {e}")
        return []


def clean_query(q: str) -> str:
    q = re.sub(r"\b(professionals?|candidates?|individuals?|leaders?|background(?: in)?|experience(?: in)?)\b", "", q, flags=re.I)
    q = re.sub(r"[\"'“”‘’,.]+", "", q)
    return re.sub(r'\s+', ' ', q.strip())

def generate_query(user_input: str, sector_hint: str = "", company_desc: str = "") -> str:
    hints = []
    if sector_hint: 
        hints.append(f"Sector: {', '.join(sector_hint) if isinstance(sector_hint, list) else sector_hint}.")
    if company_desc: 
        hints.append(f"Company focus: {company_desc}")
    hint_txt = " ".join(hints)

    prompt = f"""
You are an expert recruiter.

Rewrite the following user input into **one precise sentence** describing an ideal candidate’s:
- Sector/domain (if known),
- Core responsibilities or competencies,
- Measurable qualifications or tools.

✅ Use tangible skill nouns only, like "financial modelling", "Python development", "strategic planning".

Hints: {hint_txt}

Input:
\"\"\"{user_input}\"\"\"
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        rewritten = resp.choices[0].message.content.strip()
        return clean_query(rewritten or user_input)
    except Exception as e:
        print(f"[Query Rewrite Error] {e}")
        return clean_query(user_input)

# ─────────────────────────────────────
# EMBEDDING + EVIDENCE
# ─────────────────────────────────────
@lru_cache(maxsize=512)
def get_cached_embedding(text: str):
    return embedder.encode(text, convert_to_tensor=True)

@lru_cache(maxsize=256)
def get_sentence_embeddings(text):
    sentences = sent_tokenize(re.sub(r'\s+', ' ', text))
    filtered = [s for s in sentences if 30 <= len(s) <= 250 and not s.isupper()]
    return filtered, embedder.encode(filtered, convert_to_tensor=True)

def extract_evidence(query: str, cv_text: str, skills: list[str]) -> tuple[str, str]:
    sentences, sent_embs = get_sentence_embeddings(cv_text)
    if not sentences:
        return "No strong evidence found.", "none"

    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, sent_embs)[0].cpu().numpy().tolist()

    # Level 1: Query match
    candidates = [(sc, s) for sc, s in zip(scores, sentences) if sc >= MIN_SIM_THRESHOLD]
    if candidates:
        source = "query"
        top = sorted(candidates, key=lambda x: x[0], reverse=True)
        deduped = list(dict.fromkeys([s for _, s in top]))  # preserve order, remove duplicates
        bullets = [f"• {s if s.endswith('.') else s + '.'}" for s in deduped[:MAX_EVIDENCE]]

        return "\n".join(bullets), source

    # Level 2: Skill-wise keyword matching
    skill_matches = []
    for skill in skills:
        keywords = re.findall(r'\w+', skill.lower())
        matched = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        for m in matched:
            skill_matches.append((skill, m))

    if skill_matches:
        source = "skills"
        seen = set()
        bullets = []
        for skill, sent in skill_matches:
            if len(bullets) >= MAX_EVIDENCE:
                break
            if sent not in seen:
                bullets.append(f"• ({skill}) {sent.strip()}")
                seen.add(sent)
        return "\n".join(bullets), source

    # Level 3: No evidence
    if not sentences:
        fallback = cv_text.strip()[:400]
        return f"No long sentence evidence found. Fallback preview:\n• {fallback}", "none"



# ─────────────────────────────────────
# SCORING, LOGGING & UI
# ─────────────────────────────────────
def fuzzy_match(skill: str, meta_skills: list[str]) -> bool:
    return any(SequenceMatcher(None, skill, ms.lower()).ratio() > FUZZY_THRESHOLD for ms in meta_skills)

async def explain_match_gpt_async(client, query: str, evidence: str) -> str:
    prompt = f"""
You are a fair and neutral hiring analyst.

Only using the EVIDENCE below, and the REQUIREMENT sentence:

REQUIREMENT: "{query}"

EVIDENCE:
{evidence}

Your task:
- Write **3 concise, factual bullet points** explaining why this person fits the requirement.
- Use **only the evidence** – do **not assume** anything not shown.
- Avoid vague praise like "excellent", "strong", "clearly".
- Each bullet should match one point from the requirement.
- Use neutral language and avoid repetition.
- If evidence is unclear, say so.

Format:
- Bullet 1
- Bullet 2
- Bullet 3
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation error: {e}"



def semantic_skill_match(skill: str, meta_skills: list[str], threshold: float = 0.7) -> bool:
    if not skill or not meta_skills:
        return False
    try:
        skill_emb = embedder.encode(skill, convert_to_tensor=True)
        meta_embs = embedder.encode(meta_skills, convert_to_tensor=True)
        sims = util.cos_sim(skill_emb, meta_embs)[0]
        return any(s.item() > threshold for s in sims)
    except Exception as e:
        print(f"[Semantic Skill Match Error] {e}")
        return False

def rerank_and_score(query: str, matches: list[dict], skills: list[str]) -> list[dict]:
    raw_scores, logs, evidences, sources, explanations = [], [], [], [], []
    texts = [m["metadata"].get("text", "") for m in matches]
    pairs = [(query, t[:2000]) for t in texts]
    cross_raw = cross_encoder.predict(pairs)
    cross_probs = expit(cross_raw)

    async def generate_explanations_batch():
        tasks = []
        for i, m in enumerate(matches):
            pine = m.get("score", 0)
            cross = cross_probs[i]
            raw = PINE_WEIGHT * pine + CROSS_WEIGHT * cross
            evidence, source = extract_evidence(query, texts[i], skills)

            if source == "none":
                raw *= 0.7
            elif source == "skills":
                raw += 0.03

            meta_skills = m["metadata"].get("skills", [])
            matched_fuzzy = any(fuzzy_match(skill, meta_skills) for skill in skills)
            matched_semantic = any(semantic_skill_match(skill, meta_skills) for skill in skills)
            matched_meta = matched_fuzzy or matched_semantic

            if matched_meta:
                raw += 0.05

            evidences.append(evidence)
            sources.append(source)
            logs.append((pine, cross, raw, source, matched_meta))

            tasks.append(explain_match_gpt_async(openai_client, query, evidence))

        return await asyncio.gather(*tasks)

    # Run all explanation requests in parallel
    explanations = asyncio.run(generate_explanations_batch())

    # Final score computation
    raw_scores = [r[2] for r in logs]
    mean_raw = np.mean(raw_scores)
    std_raw = np.std(raw_scores) or 1e-6
    z_scores = [(r - mean_raw) / std_raw for r in raw_scores]
    final_scores = [round(max(4.5, min(7.5 + z * 2.5, 10.0)), 2) for z in z_scores]

    def score_label(score):
        if score >= 9.0: return "\U0001F7E2 Perfect Match"
        elif score >= 7.5: return "\U0001F7E1 Strong Fit"
        elif score >= 5.0: return "\U0001F7E0 Moderate Fit"
        return "\U0001F534 Weak Fit"

    results = []
    for i, (m, ev, src, expl, log, score) in enumerate(zip(matches, evidences, sources, explanations, logs, final_scores), 1):
        pine, cross, raw, _, matched_meta = log
        log_scaled = {
            "pinecone_score": float(np.log1p(pine)),
            "cross_score": float(np.log1p(cross)),
            "combined_raw_score": float(np.log1p(raw))
        }
        results.append({
            "name": m["metadata"].get("name", "Unnamed"),
            "cv_link": m["metadata"].get("cv_link", ""),
            "score": score,
            "score_label": score_label(score),
            "evidence": ev,
            "explanation": expl,
            "evidence_source": src,
            "log": {
                "pinecone_score": pine,
                "cross_score": cross,
                "matched_metadata": matched_meta,
                "raw_score": raw,
                "log_scaled": log_scaled
            }
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


    def score_label(score):
        if score >= 9.0: return "\U0001F7E2 Perfect Match"
        elif score >= 7.5: return "\U0001F7E1 Strong Fit"
        elif score >= 5.0: return "\U0001F7E0 Moderate Fit"
        return "\U0001F534 Weak Fit"

    results = []
    for i, (m, ev, src, expl, log, score) in enumerate(zip(matches, evidences, sources, explanations, logs, final_scores), 1):
        pine, cross, raw, _, matched_meta = log
        log_scaled = {
            "pinecone_score": float(np.log1p(pine)),
            "cross_score": float(np.log1p(cross)),
            "combined_raw_score": float(np.log1p(raw))
        }
        results.append({
            "name": m["metadata"].get("name", "Unnamed"),
            "cv_link": m["metadata"].get("cv_link", ""),
            "score": score,
            "score_label": score_label(score),
            "evidence": ev,
            "explanation": expl,
            "evidence_source": src,
            "log": {
                "pinecone_score": pine,
                "cross_score": cross,
                "matched_metadata": matched_meta,
                "raw_score": raw,
                "log_scaled": log_scaled
            }
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)

# ─────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────
st.title(" CV Matcher")

user_input = st.text_input("Target (sector, skill, or URL):", placeholder="e.g. Howdens, LLMs")
if not user_input: st.stop()

sector_info = {"sector": "", "description": ""}
about_text = ""

if is_probable_domain(user_input):
    domain = extract_domain(user_input)
    st.markdown("### 🧪 Website Analysis")
    st.markdown(f"**Domain:** `{domain}`")
    about_text = fetch_about_text(domain)
    if about_text:
        sector_info = infer_sector_from_text(about_text)
else:
    about_text = user_input
    sector_info = infer_sector_from_text(about_text)

# ✅ This part shows about_text and sector info for both cases
if about_text:
    st.markdown("**Extracted About Text (preview):**")
    st.code(about_text[:500])

if sector_info.get("sector"):
    st.markdown("**Inferred Company Sector:**")
    st.json(sector_info)
else:
    st.warning("Sector could not be inferred from text.")


with st.spinner("Rewriting query..."):
    try:
        query = generate_query(user_input, sector_info.get("sector", ""))
    except Exception as e:
        st.error(f"Query error: {e}")
        query = user_input

st.markdown(f"**Search Query:** `{query}`")
skills_extracted = extract_skills_from_query(query)
st.markdown(f"**Skills extracted:** {skills_extracted}")

query_vec = get_cached_embedding(query)
skills_vec = get_cached_embedding(", ".join(skills_extracted)) if skills_extracted else None

# Adaptive blending
if skills_vec is not None:
    q_vec = 0.75 * query_vec + 0.25 * skills_vec
else:
    q_vec = query_vec

with st.spinner("Searching..."):
    try:
        # Clean and validate sector info
        raw_sectors = sector_info.get("sector", [])
        sectors = [s.strip() for s in raw_sectors if isinstance(s, str) and s.strip()]
        
        # Build safe filter
        base_filter = {"tier": {"$eq": "A"}}
        if sectors:
            base_filter["sectors"] = {"$in": sectors}
            st.markdown(f"**Filtering by inferred sector(s):** `{', '.join(sectors)}`")
        else:
            st.warning("⚠️ No reliable sector detected — running unfiltered Tier-A match.")

        resp = index.query(
            vector=q_vec.tolist(),
            top_k=TOP_K,
            include_metadata=True,
            filter=base_filter
        )

        matches = resp.get("matches", [])

        # ✅ Normalise cosine similarity from [-1, 1] to [0, 1]
        for m in matches:
            raw_score = m.get("score", 0.0)
            m["score"] = max(0.0, min((raw_score + 1) / 2, 1.0))  # Clamp to [0, 1]


    except Exception as e:
        st.error(f"Pinecone error: {e}")
        st.stop()

    matches = resp.get("matches", [])
    if not matches:
        st.warning("No Tier-A matches found.")
        st.stop()

    results = rerank_and_score(query, matches, skills_extracted)

for i, r in enumerate(results, 1):
    st.markdown(f"### Match {i}: {r['name']} — {r['score']}/10  {r['score_label']}")
    st.write(f"**CV**: [View]({r['cv_link']})")
    st.markdown("**Why This Matches (GPT‑4):**")
    st.markdown(r["explanation"])
    st.markdown("**Raw Evidence Extracted:**")
    st.markdown(r["evidence"])
    st.markdown(f"**Evidence Source:** {r['evidence_source']}")
    with st.expander("Debug Log"):
        st.json(r["log"])
    st.markdown("---")