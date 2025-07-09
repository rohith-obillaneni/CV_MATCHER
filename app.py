
import os
import re
import json
import openai
import streamlit as st
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
from scipy.special import expit
from dotenv import load_dotenv
from functools import lru_cache
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import requests
import nltk
nltk.download('punkt')
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
MIN_SIM_THRESHOLD  = 0.6
MAX_EVIDENCE       = 3
FUZZY_THRESHOLD    = 0.85

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

@st.cache_resource
def load_models():
    cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base", device="cpu")
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
    if not about_text or len(about_text.strip()) < 50:
        print("[Sector Inference] Not enough text.")
        return {"sector": "", "description": ""}

    prompt = f"""
The following is an About Us section of a company:
\"\"\"{about_text}\"\"\"

Identify the company’s primary industry sector (e.g. FinTech, HealthTech, Retail, Logistics, EdTech), and give a 1-sentence summary of what they do. Return JSON like:
{{
  "sector": "FinTech",
  "description": "Provides global payment infrastructure for cross-border businesses."
}}
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        print("[GPT Output]", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[GPT Error] {e}")
        return {"sector": "", "description": ""}

# ─────────────────────────────────────
# QUERY + SKILL EXTRACTION
# ─────────────────────────────────────
def clean_query(q: str) -> str:
    q = q.strip().strip("\"'")
    return re.sub(r'\b(seeking|tier a|best|candidates|professionals)\b', '', q, flags=re.I).strip()

def generate_query(user_input: str, sector_hint: str = "", company_desc: str = "") -> str:
    hints = []
    if sector_hint: hints.append(f"Sector: {sector_hint}.")
    if company_desc: hints.append(f"Company focus: {company_desc}")
    hint_txt = " ".join(hints)
    prompt = f"""
You are an expert recruiter. Rewrite this into one precise sentence describing an ideal candidate's skills, sector, achievements, and experience. Be direct and concise. Fetch the sector from the query if it is url or company name and include it in the output{hint_txt}

Input: "{user_input}"
"""
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    print(hint_txt)
    return clean_query(resp.choices[0].message.content)

def extract_skills_from_query(query: str) -> list[str]:
    prompt = f"""Extract the key skills from this sentence as a JSON list: \"{query}\""""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        skills = json.loads(resp.choices[0].message.content)
        return [s.lower() for s in skills if isinstance(s, str)]
    except Exception:
        return []

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
    scores = util.cos_sim(query_emb, sent_embs)[0]
    scores = scores.cpu().numpy().tolist()
    candidates = [(sc, s) for sc, s in zip(scores, sentences) if sc >= MIN_SIM_THRESHOLD]
    source = "query" if candidates else "none"
    if not candidates and skills:
        keywords = [kw for skill in skills for kw in re.findall(r'\w+', skill)]
        candidates = [(sc, s) for sc, s in zip(scores, sentences)
                      if any(kw in s.lower() for kw in keywords)]
        source = "skills" if candidates else "none"
    if not candidates:
        return "No strong evidence found.", "none"
    top_evidence = sorted(candidates, key=lambda x: x[0], reverse=True)[:MAX_EVIDENCE]
    bullets = [f"\u2022 {s if s.endswith('.') else s + '.'}" for _, s in top_evidence]
    return "\n".join(bullets), source

# ─────────────────────────────────────
# SCORING, LOGGING & UI
# ─────────────────────────────────────
def fuzzy_match(skill: str, meta_skills: list[str]) -> bool:
    return any(SequenceMatcher(None, skill, ms.lower()).ratio() > FUZZY_THRESHOLD for ms in meta_skills)

def explain_match_gpt(query, evidence):
    prompt = f"""You're a hiring manager. ONLY using this evidence:

{evidence}

and this requirement: \"{query}\",

Explain in 3 bullet points why this person is a strong match.
Be specific and DO NOT include any information not present in the evidence.
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Explanation error: {e}"

def rerank_and_score(query: str, matches: list[dict], skills: list[str]) -> list[dict]:
    raw_scores, logs, evidences, sources, explanations = [], [], [], [], []
    texts = [m["metadata"].get("text", "") for m in matches]
    pairs = [(query, t[:2000]) for t in texts]
    cross_raw = cross_encoder.predict(pairs)
    cross_probs = expit(cross_raw)

    for i, m in enumerate(matches):
        pine = m.get("score", 0)
        cross = cross_probs[i]
        raw = PINE_WEIGHT * pine + CROSS_WEIGHT * cross
        evidence, source = extract_evidence(query, texts[i], skills)
        if source == "none": raw *= 0.7
        elif source == "skills": raw += 0.03
        meta_skills = m["metadata"].get("skills", [])
        matched_meta = any(fuzzy_match(skill, meta_skills) for skill in skills)
        if matched_meta: raw += 0.05
        gpt_expl = explain_match_gpt(query, evidence)
        evidences.append(evidence)
        sources.append(source)
        explanations.append(gpt_expl)
        raw_scores.append(raw)
        logs.append((pine, cross, raw, source, matched_meta))

    min_raw = min(raw_scores)
    max_raw = max(raw_scores)
    range_raw = max_raw - min_raw if max_raw != min_raw else 1e-6
    final_norm = [(r - min_raw) / range_raw for r in raw_scores]
    final_scores = [round(5 + s * 5, 2) for s in final_norm]

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

    if not about_text:
        print("[DEBUG] No content extracted from website.")
    if about_text:
        st.markdown("**Extracted About Text (preview):**")
        st.code(about_text[:500])
        sector_info = infer_sector_from_text(about_text)

        if sector_info.get("sector"):
            st.markdown("**Inferred Company Sector:**")
            st.json(sector_info)
        else:
            st.warning("Sector could not be inferred from website text.")
    else:
        st.error("No usable content found on the website.")


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
skills_vec = get_cached_embedding(", ".join(skills_extracted))
q_vec = 0.7 * query_vec + 0.3 * skills_vec

with st.spinner("Searching..."):
    try:
        resp = index.query(
            vector=q_vec.tolist(),
            top_k=TOP_K,
            include_metadata=True,
            filter={"tier": {"$eq": "A"}}
        )
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