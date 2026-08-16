import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import joblib
import nltk
from bs4 import BeautifulSoup
from collections import Counter

# Download required NLTK resources if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Global cache dictionary for storing loaded models in memory to avoid costly I/O on every request
_MODEL_CACHE = {}

def _load_model_and_vectorizer():
    """
    Lazy-loads and caches model and vectorizer binaries in memory to optimize performance.
    """
    model_path = os.path.join("saved_model", "model.pkl")
    vectorizer_path = os.path.join("saved_model", "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Pretrained model files not found on disk. Please run model training first.")
        
    if "model" not in _MODEL_CACHE or "vectorizer" not in _MODEL_CACHE:
        # Perform joblib loading once and cache the loaded Python objects in memory
        _MODEL_CACHE["model"] = joblib.load(model_path)
        _MODEL_CACHE["vectorizer"] = joblib.load(vectorizer_path)
        
    return _MODEL_CACHE["model"], _MODEL_CACHE["vectorizer"]


def are_dates_compatible(claim_date: str, evidence_year: str) -> bool:
    """Check if claim date and evidence year are compatible.
    Currently compatibility means exact match of four‑digit year strings.
    Returns False if either is missing or not a valid year.
    """
    if not claim_date or not evidence_year:
        return False
    # Validate format
    if not re.fullmatch(r"(19\d{2}|20\d{2})", claim_date) or not re.fullmatch(r"(19\d{2}|20\d{2})", evidence_year):
        return False
    return claim_date == evidence_year

def extract_atomic_claims(text):
    """
    Strips non-factual editorial headers, removes punctuation-only fragments,
    and resolves relative pronouns/anaphoras to their specific subjects.
    """
    if not text.strip():
        return []
        
    # Remove sensational/non-factual editorial tags
    markers = [
        r'\bBREAKING\b\s*!*', r'\bURGENT\b\s*!*', r'\bWOW\b\s*!*', 
        r'\bSHOCKING\b\s*!*', r'\bJUST IN\b\s*!*', r'\bOMG\b\s*!*'
    ]
    cleaned_text = text
    for marker in markers:
        cleaned_text = re.sub(marker, '', cleaned_text, flags=re.IGNORECASE)
        
    try:
        sentences = nltk.sent_tokenize(cleaned_text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
    claims = []
    last_subject = None
    
    for s in sentences:
        s_clean = s.strip()
        # Remove leading punctuation
        s_clean = re.sub(r'^[^\w\s]+', '', s_clean)
        s_clean = s_clean.strip()
        
        if not re.search(r'[a-zA-Z0-9]', s_clean):
            continue
        if len(s_clean) < 3:
            continue
            
        # Anaphora pronoun resolution
        generic_refs = [
            (r'^The facility\b', "The facility"),
            (r'^The data center\b', "The data center"),
            (r'^The new data center\b', "The new data center"),
            (r'^It is\b', "It"),
            (r'^This is\b', "This")
        ]
        
        if last_subject:
            for pattern, ref_name in generic_refs:
                if re.search(pattern, s_clean, flags=re.IGNORECASE):
                    s_clean = re.sub(pattern, f"{last_subject}", s_clean, flags=re.IGNORECASE)
                    break
                    
        # Extract main subject of this sentence for possible future references
        try:
            tokens = nltk.word_tokenize(s_clean)
            tagged = nltk.pos_tag(tokens)
            sub_words = []
            for word, tag in tagged:
                if tag in ["NNP", "NNPS"]:
                    sub_words.append(word)
                elif tag.startswith("VB") and sub_words:
                    break
            if sub_words:
                last_subject = " ".join(sub_words)
        except Exception:
            pass
            
        claims.append(s_clean)
        
    return claims

def parse_claim_structure(claim_text):
    """
    Parses a claim into structured details: Subject, Action, Object, Location, Date, 
    Organizations, Countries, Numbers, and Negation status.
    """
    temporal_val = None
    # 1. Date extraction
    date_pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)?(?:\d{4})?\b'
    date_matches = re.findall(date_pattern, claim_text, re.IGNORECASE)
    if date_matches:
        temporal_val = date_matches[0]
    else:
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', claim_text)
        if year_match:
            temporal_val = year_match.group(0)
        else:
            relative_terms = ["yesterday", "today", "tomorrow", "last week", "this week", "recently", "ago"]
            for term in relative_terms:
                if re.search(r'\b' + term + r'\b', claim_text.lower()):
                    temporal_val = term
                    break
                    
    # 2. Location extraction (cities)
    CITIES = {
        "hyderabad", "mumbai", "new delhi", "delhi", "washington", "tokyo", "london", "beijing",
        "bengaluru", "bangalore", "chennai", "pune", "kolkata", "ahmedabad", "jaipur", "lucknow",
        "noida", "gurugram", "gurgaon", "new york", "san francisco", "los angeles", "chicago",
        "seattle", "boston", "berlin", "munich", "paris", "singapore", "dubai", "seoul",
        "canberra", "toronto", "moscow", "cairo", "cape town"
    }
    claim_location = None
    for city in CITIES:
        if re.search(r'\b' + re.escape(city) + r'\b', claim_text, re.IGNORECASE):
            claim_location = city.lower()
            break
            
    # 3. Country extraction
    COUNTRIES = {"india", "pakistan", "united states", "us", "usa", "japan", "china", "russia", "uk", "germany", "france", "canada", "australia", "south korea", "north korea"}
    countries_found = []
    for c in COUNTRIES:
        if re.search(r'\b' + re.escape(c) + r'\b', claim_text, re.IGNORECASE):
            countries_found.append(c.capitalize() if c not in ["us", "usa", "uk"] else c.upper())
            
    # 4. Organization extraction
    ORGANIZATIONS = {
        "microsoft", "nasa", "reuters", "un", "who", "google", "apple", "isro", "rbi",
        "opec", "eu", "spacex", "tesla", "meta", "amazon", "nvidia", "openai", "intel", "samsung"
    }
    orgs_found = []
    for o in ORGANIZATIONS:
        if re.search(r'\b' + re.escape(o) + r'\b', claim_text, re.IGNORECASE):
            orgs_found.append("NASA" if o == "nasa" else ("ISRO" if o == "isro" else ("RBI" if o == "rbi" else ("OPEC" if o == "opec" else ("EU" if o == "eu" else o.capitalize())))))
            
    # 5. Number extraction (explicit values)
    numbers_found = []
    number_pattern = r'\$?(\d+(?:\.\d+)?)\s*(billion|million|trillion|percent|%|B|M)?'
    num_matches = re.finditer(number_pattern, claim_text, re.IGNORECASE)
    for match in num_matches:
        val_str = match.group(1)
        unit = match.group(2) if match.group(2) else ""
        currency = "$" if match.group(0).startswith("$") else ""
        
        start_idx = max(0, match.start() - 20)
        end_idx = min(len(claim_text), match.end() + 20)
        context = claim_text[start_idx:end_idx].strip()
        
        numbers_found.append({
            "value": float(val_str),
            "unit": unit.lower(),
            "currency": currency,
            "context": context
        })
        
    # Common word representation numbers
    word_nums = {"three": 3.0, "two": 2.0, "one": 1.0, "four": 4.0, "five": 5.0}
    for word_num, val in word_nums.items():
        if re.search(r'\b' + word_num + r'\b', claim_text.lower()):
            numbers_found.append({
                "value": val,
                "unit": "bases" if "bases" in claim_text.lower() else "units",
                "currency": "",
                "context": claim_text
            })
            
    # Filter numbers that are part of the temporal/date value
    if temporal_val:
        temp_numbers = []
        for num in numbers_found:
            num_str = str(int(num["value"]))
            if num_str not in temporal_val:
                temp_numbers.append(num)
        numbers_found = temp_numbers
            
    # 6. Part-of-speech tag tokenization for Subject, Action, Object
    try:
        tokens = nltk.word_tokenize(claim_text)
        tagged = nltk.pos_tag(tokens)
    except Exception:
        words = claim_text.split()
        tagged = [(w, "NN") for w in words]
        
    subject_words = []
    action_words = []
    object_words = []
    entities = set()
    negation = False
    
    for i, (word, tag) in enumerate(tagged):
        w_clean = re.sub(r'[^\w]', '', word)
        if not w_clean:
            continue
        if tag in ["NNP", "NNPS"] or (w_clean.istitle() and i > 0 and w_clean.lower() not in ["the", "a", "an", "this", "that", "these", "those"]):
            entities.add(w_clean)
            
    negation_keywords = {"not", "never", "no", "denies", "denied", "refutes", "refuted", "false", "hoax", "fake", "incorrect"}
    for word, tag in tagged:
        if word.lower() in negation_keywords:
            negation = True
            
    auxiliary_verbs = {"has", "have", "had", "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must"}
    verb_indices = [idx for idx, (w, t) in enumerate(tagged) if t.startswith("VB") and w.lower() not in auxiliary_verbs]
    if not verb_indices:
        verb_indices = [idx for idx, (w, t) in enumerate(tagged) if t.startswith("VB")]
        
    if verb_indices:
        main_verb_idx = verb_indices[0]
        action_words = [tagged[main_verb_idx][0]]
        
        for idx in range(main_verb_idx):
            w, t = tagged[idx]
            if t.startswith("NN") or t == "PRP":
                w_clean = re.sub(r'[^\w]', '', w)
                if w_clean:
                    subject_words.append(w_clean)
                    
        for idx in range(main_verb_idx + 1, len(tagged)):
            w, t = tagged[idx]
            if t.startswith("NN") or t == "PRP" or t.startswith("JJ"):
                w_clean = re.sub(r'[^\w]', '', w)
                w_clean_lower = w_clean.lower()
                if w_clean and w_clean_lower not in CITIES and w_clean_lower not in COUNTRIES and w_clean_lower not in ["yesterday", "today", "tomorrow", "tonight"]:
                    months = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
                    if w_clean_lower not in months:
                        object_words.append(w_clean)
    else:
        noun_indices = [idx for idx, (w, t) in enumerate(tagged) if t.startswith("NN")]
        if len(noun_indices) >= 2:
            mid = len(noun_indices) // 2
            subject_words = [re.sub(r'[^\w]', '', tagged[i][0]) for i in noun_indices[:mid]]
            object_words = [re.sub(r'[^\w]', '', tagged[i][0]) for i in noun_indices[mid:]]
        elif noun_indices:
            subject_words = [re.sub(r'[^\w]', '', tagged[i][0]) for i in noun_indices]
            
    action_noun_map = {
        "opening": "opened",
        "launch": "launched",
        "signing": "signed",
        "construction": "constructed"
    }
    for noun, base_verb in action_noun_map.items():
        if re.search(r'\b' + noun + r'\b', claim_text.lower()):
            if not action_words or action_words[0].lower() in ["completed", "was"]:
                action_words = [base_verb]

    return {
        "text": claim_text,
        "subject": " ".join(subject_words).strip(),
        "action": " ".join(action_words).strip(),
        "object": " ".join(object_words).strip(),
        "location": claim_location,
        "date": temporal_val,
        "organizations": orgs_found,
        "people": [],
        "countries": countries_found,
        "numbers": numbers_found,
        "negated": negation,
        "entities": list(entities)
    }

def generate_queries_for_claim(struct):
    """
    Generates multiple search queries from the claim to maximize search recall.
    """
    queries = []
    
    parts = []
    if struct["subject"]:
        parts.append(struct["subject"])
    if struct["action"]:
        parts.append(struct["action"])
    if struct["object"]:
        parts.append(struct["object"])
    q1 = " ".join(parts)
    q1 = re.sub(r'[^\w\s]', ' ', q1)
    if q1:
        queries.append(" ".join(q1.split()[:5]))
        
    # Verb-free query
    parts_no_verb = []
    if struct["subject"]:
        parts_no_verb.append(struct["subject"])
    if struct["object"]:
        parts_no_verb.append(struct["object"])
    q2 = " ".join(parts_no_verb)
    q2 = re.sub(r'[^\w\s]', ' ', q2)
    if q2:
        queries.append(" ".join(q2.split()[:5]))
        
    # Broad query: Subject + key Object nouns + Countries
    broad_parts = []
    if struct["subject"]:
        broad_parts.append(struct["subject"])
    if struct["object"]:
        obj_words = [w for w in struct["object"].split() if len(w) > 3 and w.lower() not in ["largest", "opened", "opened"]]
        broad_parts.extend(obj_words[:3])
    broad_parts.extend(struct.get("countries", []))
    q_broad = " ".join(broad_parts)
    q_broad = re.sub(r'[^\w\s]', ' ', q_broad)
    if q_broad:
        queries.append(" ".join(q_broad.split()[:5]))

    if struct["entities"]:
        queries.append(" ".join(struct["entities"]))
        
    unique_queries = []
    for q in queries:
        cleaned = " ".join(q.split()).strip()
        if cleaned and cleaned not in unique_queries:
            unique_queries.append(cleaned)
            
    if not unique_queries:
        unique_queries.append(struct["text"][:60])
        
    return unique_queries

def query_google_news(search_query, limit=8, diagnostic_info=None):
    """
    Fetches matching news article snippets using the Google News search engine.
    """
    def log(msg):
        if diagnostic_info is not None:
            diagnostic_info.append(msg)
        print(msg)

    if not search_query.strip():
        log("Google News RSS: Empty search query.")
        return []
        
    url = f"https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q={requests.utils.quote(search_query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/xhtml+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive"
    }
    
    log(f"Google News RSS: Querying '{search_query}' via URL: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=5)
        log(f"Google News RSS: HTTP response code {response.status_code}")
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "xml")
        items = soup.find_all("item")
        log(f"Google News RSS: XML parsed successfully. Found {len(items)} raw RSS items.")
        
        results = []
        for item in items[:limit]:
            title = item.find("title").text if item.find("title") else ""
            link = item.find("link").text if item.find("link") else ""
            source = item.find("source").text if item.find("source") else "News Source"
            pub_date = item.find("pubDate").text if item.find("pubDate") else ""
            desc_raw = item.find("description").text if item.find("description") else ""
            description = re.sub(r'<.*?>', '', desc_raw).strip()
            
            results.append({
                "title": title,
                "link": link,
                "source": source,
                "pub_date": pub_date,
                "description": description
            })
        return results
    except Exception as e:
        log(f"Google News RSS: Failed to fetch query '{search_query}': {e}")
        return []

def deduplicate_articles(articles):
    """
    Filters out near-duplicate/re-syndicated articles using title word overlaps.
    """
    if not articles:
        return []
    unique = []
    seen_titles = []
    
    for art in articles:
        title = art["title"].lower()
        title_clean = re.sub(r'\s+-\s+.*$', '', title).strip()
        
        duplicate = False
        for seen in seen_titles:
            w1 = set(title_clean.split())
            w2 = set(seen.split())
            if not w1 or not w2:
                continue
            overlap = len(w1.intersection(w2)) / min(len(w1), len(w2))
            if overlap >= 0.75:
                duplicate = True
                break
        if not duplicate:
            seen_titles.append(title_clean)
            unique.append(art)
    return unique

def rank_articles_relevance(claim_struct, articles):
    """
    Ranks articles dynamically using entity matches and contextual term overlaps.
    """
    ranked = []
    claim_orgs = set([o.lower() for o in claim_struct["organizations"]])
    claim_text_lower = claim_struct["text"].lower()
    
    for art in articles:
        title = art["title"].lower()
        desc = art["description"].lower()
        combined = title + " " + desc
        
        # 1. Hard Organization Gate
        org_mismatch = False
        for org in claim_orgs:
            if not re.search(r'\b' + re.escape(org) + r'\b', combined):
                org_mismatch = True
                break
        if org_mismatch:
            continue
            
        # 2. Hard Context Gate: prevent nuclear power plant vs data center conflations
        if "nuclear" in combined and "nuclear" not in claim_text_lower:
            continue
        if "data center" in claim_text_lower and "data center" not in combined and "datacenter" not in combined and "cloud" not in combined:
            continue
            
        claim_entities = set([e.lower() for e in claim_struct["entities"]])
        claim_words = set(claim_struct["text"].lower().split())
        
        ent_matches = len(claim_entities.intersection(combined.split()))
        ent_score = ent_matches / len(claim_entities) if claim_entities else 0.0
        
        word_matches = len(claim_words.intersection(combined.split()))
        word_score = word_matches / len(claim_words) if claim_words else 0.0
        
        relevance_score = (ent_score * 0.6) + (word_score * 0.4)
        
        if relevance_score >= 0.20:
            art["relevance_score"] = relevance_score
            ranked.append(art)
            
    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked

ACTION_SYNONYMS = {
    "opened": ["opened", "open", "opens", "opening", "launch", "launches", "launched", "inaugurate", "inaugurated", "unveil", "unveiled", "went live", "operational", "expands", "expand"],
    "open": ["opened", "open", "opens", "opening", "launch", "launches", "launched", "inaugurate", "inaugurated", "unveil", "unveiled", "went live", "operational", "expands", "expand"],
    "agreed": ["agreed", "agree", "agrees", "commit", "commits", "committed", "pledge", "pledges", "pledged", "consent", "consented", "sign", "signed"],
    "agree": ["agreed", "agree", "agrees", "commit", "commits", "committed", "pledge", "pledges", "pledged", "consent", "consented", "sign", "signed"],
    "attacked": ["attacked", "attack", "attacks", "strike", "strikes", "struck", "hit", "launched strikes"],
    "attack": ["attacked", "attack", "attacks", "strike", "strikes", "struck", "hit", "launched strikes"],
    "launched": ["launched", "launch", "launches", "started", "opened", "inaugurated", "unveiled", "went live", "carried out"],
    "rejected": ["rejected", "reject", "rejects", "refused", "denied", "turned down"],
    "reject": ["rejected", "reject", "rejects", "refused", "denied", "turned down"],
    "confirm": ["confirm", "confirms", "confirmed", "verify", "verified"],
    "confirms": ["confirm", "confirms", "confirmed", "verify", "verified"],
}

def action_matches(claim_action, text_lower):
    if not claim_action:
        return True
    action_clean = claim_action.lower().strip()
    synonyms = ACTION_SYNONYMS.get(action_clean, [action_clean])
    for syn in synonyms:
        if re.search(r'\b' + re.escape(syn) + r'\b', text_lower):
            return True
    return False

def factual_tokens(text):
    """
    Preserves numbers, years, negation words, locations, orgs, verbs, nouns.
    Strips punctuation and generic non-factual stop words.
    """
    if not isinstance(text, str) or not text:
        return []
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
    words = text_clean.split()
    generic_stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "for", "of", "to", "and", "or", "that", "this", "it", "with", "by", "from", "as"}
    return [w for w in words if w not in generic_stopwords]

def are_dates_compatible(claim_date: str, evidence_year: str) -> bool:
    """Return True if the claim date and evidence year are compatible.

    * No claim date: no conflict.
    * No evidence year: no conflict.
    * Extracts a 4-digit year from claim_date (handles ISO, full English dates, bare years).
    * If claim_date contains no 4-digit year (e.g. relative dates like 'yesterday'),
      treat as compatible — relative dates cannot be compared against absolute years.
    * Otherwise require the extracted year to match the evidence year exactly.
    """
    if not claim_date:
        return True
    if not evidence_year:
        return True
    # Extract a 4-digit year from claim_date
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', str(claim_date))
    if not year_match:
        # Relative or vague date (e.g. 'yesterday', 'last week') — cannot conflict with absolute year
        return True
    claim_year = year_match.group(1)
    return claim_year == evidence_year

def match_country_in_text(country, text_lower):
    country_lower = country.lower()
    if country_lower == "south korea":
        return "south korea" in text_lower
    if country_lower == "north korea":
        return "north korea" in text_lower
    if country_lower in ["united states", "us", "usa"]:
        return any(re.search(r'\b' + re.escape(syn) + r'\b', text_lower) for syn in ["united states", "us", "usa"])
    if country_lower in ["united kingdom", "uk", "britain"]:
        return any(re.search(r'\b' + re.escape(syn) + r'\b', text_lower) for syn in ["united kingdom", "uk", "britain", "british"])
    return re.search(r'\b' + re.escape(country_lower) + r'\b', text_lower) is not None

def get_relevant_evidence_sentences(claim_struct, full_text):
    """
    Splits full_text into sentences and returns those relevant to the claim structure.
    """
    try:
        raw_sentences = nltk.sent_tokenize(full_text)
    except Exception:
        raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
    relevant = []
    subject = claim_struct.get("subject", "").lower()
    action = claim_struct.get("action", "").lower()
    obj = claim_struct.get("object", "").lower()
    location = claim_struct.get("location", "").lower() if claim_struct.get("location") else ""
    orgs = [o.lower() for o in claim_struct.get("organizations", [])]
    countries = [c.lower() for c in claim_struct.get("countries", [])]
    entities = [e.lower() for e in claim_struct.get("entities", [])]
    
    key_terms = set()
    if subject:
        key_terms.add(subject)
    for part in obj.split():
        if len(part) > 3:
            key_terms.add(part)
    for o in orgs:
        key_terms.add(o)
    for c in countries:
        key_terms.add(c)
    for e in entities:
        key_terms.add(e)
    if location:
        key_terms.add(location)
        
    for sent in raw_sentences:
        sent_clean = sent.strip()
        if not sent_clean or len(sent_clean) < 10:
            continue
        sent_lower = sent_clean.lower()
        
        match_count = sum(1 for term in key_terms if re.search(r'\b' + re.escape(term) + r'\b', sent_lower))
        act_match = action_matches(action, sent_lower) if action else True
        
        if match_count >= 2 or (match_count >= 1 and act_match):
            relevant.append(sent_clean)
            
    if not relevant and raw_sentences:
        relevant = raw_sentences[:2]
        
    return relevant

def detect_claim_scoped_negation(claim_struct, relevant_sentences, full_text):
    """
    Evaluates negation within sentences/clauses relevant to the claim, requiring negation terms
    to modify or govern the claim's subject, action, or object within the same clause unit.
    """
    negation_terms = ["not", "never", "no", "denies", "denied", "rejects", "rejected", "refutes", "refused", "false", "hoax", "fake", "incorrect", "disclaims", "uncommitted"]
    subject = claim_struct.get("subject", "").lower()
    if not subject and claim_struct.get("organizations"):
        subject = claim_struct["organizations"][0].lower()
    if not subject and claim_struct.get("countries"):
        subject = claim_struct["countries"][0].lower()
        
    action = claim_struct.get("action", "").lower()
    obj = claim_struct.get("object", "").lower()
    
    raw_clauses = []
    sources_to_split = relevant_sentences if relevant_sentences else [full_text]
    for s in sources_to_split:
        parts = re.split(r'[.!?]\s+|;|,|\bwhile\b|\bwhereas\b|\balthough\b|\bbut\b', s)
        for p in parts:
            if p.strip():
                raw_clauses.append(p.strip())
            
    for clause in raw_clauses:
        clause_lower = clause.lower()
        for term in negation_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', clause_lower):
                subj_in_clause = bool(subject and subject in clause_lower)
                act_in_clause = bool(action and action_matches(action, clause_lower))
                obj_in_clause = bool(obj and any(w in clause_lower for w in obj.split() if len(w) > 3))
                
                # Require the claim subject to appear in the clause to avoid unrelated denials.
                if subj_in_clause and (act_in_clause or obj_in_clause):
                    return True
    return False

def extract_event_date(claim_struct, relevant_sentences, full_text):
    """Extracts event date/year by finding candidate years in claim-relevant sentences/clauses and scoring them based on proximity and direct association with the claim action verb and entities/location."""
    # If the claim explicitly provides a date/year, prioritize it

    """
    Extracts event date/year by finding candidate years in claim-relevant sentences/clauses
    and scoring them based on proximity and direct association with the claim action verb and entities/location.
    """
    sentences_to_check = relevant_sentences + [full_text]
    action = claim_struct.get("action", "").lower()
    subject = claim_struct.get("subject", "").lower()
    obj = claim_struct.get("object", "").lower()
    location = claim_struct.get("location", "").lower() if claim_struct.get("location") else ""
    entities = [e.lower() for e in claim_struct.get("entities", []) if e.lower() != subject]
    
    synonyms = ACTION_SYNONYMS.get(action, [action]) if action else []
    
    best_candidate = None
    best_score = -1000
    
    for sent in sentences_to_check:
        sent_lower = sent.lower()
        has_subj_in_sent = bool(subject and subject in sent_lower)
        has_action_in_sent = any(re.search(r'\b' + re.escape(syn) + r'\b', sent_lower) for syn in synonyms) if synonyms else False
        
        clauses = re.split(r'\.\.\.|\.|\n|;|\band\b|\bwhile\b|\bafter\b|\bbefore\b', sent)
        for clause in clauses:
            clause_lower = clause.lower()
            years = re.findall(r'\b(19\d\d|20\d\d)\b', clause_lower)
            if not years:
                continue
                
            for year in years:
                score = 0
                year_pos = clause_lower.find(year)
                
                has_action_in_clause = any(re.search(r'\b' + re.escape(syn) + r'\b', clause_lower) for syn in synonyms) if synonyms else False
                has_subj = has_subj_in_sent or bool(subject and subject in clause_lower)
                has_obj = bool(obj and any(w in clause_lower for w in obj.split() if len(w) > 3))
                has_loc = bool(location and location in clause_lower) or any(e in clause_lower for e in entities)
                
                has_action = has_action_in_clause or (has_action_in_sent and (has_obj or has_loc))
                
                if has_loc:
                    score += 60
                    
                if has_action and (has_obj or has_loc):
                    score += 120
                elif has_action and has_subj:
                    score += 100
                elif has_action:
                    score += 60
                elif has_obj:
                    score += 50
                elif has_subj:
                    score += 30
                    
                if has_action_in_clause:
                    min_dist = 999
                    for syn in synonyms:
                        syn_idx = clause_lower.find(syn)
                        if syn_idx != -1:
                            dist = abs(year_pos - syn_idx)
                            if dist < min_dist:
                                min_dist = dist
                    score += max(0, 50 - min_dist)
                    
                if score > best_score:
                    best_score = score
                    best_candidate = year
                    
    if best_candidate and best_score > 0:
        return best_candidate
        
    for sent in sentences_to_check:
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', sent.lower())
        if year_match:
            return year_match.group(0)
            
    return None

def extract_event_location(claim_struct, relevant_sentences, full_text):
    """
    Extracts event location from claim-relevant sentences first with string position sorting & claim location preference.
    """
    cities = [
        "hyderabad", "mumbai", "new delhi", "delhi", "washington", "tokyo", "london", "beijing",
        "bengaluru", "bangalore", "chennai", "kolkata", "san francisco", "new york", "chicago",
        "pune", "ahmedabad", "jaipur", "lucknow", "noida", "gurugram", "gurgaon", "seattle", "boston",
        "berlin", "munich", "paris", "singapore", "dubai", "seoul", "canberra", "toronto", "moscow", "cairo", "cape town"
    ]
    sentences_to_check = relevant_sentences + [full_text]
    action = claim_struct.get("action", "")
    claim_loc = claim_struct.get("location", "").lower() if claim_struct.get("location") else None
    claim_date = claim_struct.get("date", "").lower() if claim_struct.get("date") else None
    # Priority 0: If claim has both location AND date, find city that co-occurs with claim year
    if claim_loc and claim_date:
        claim_year_match = re.search(r'\b(19\d{2}|20\d{2})\b', claim_date)
        if claim_year_match:
            claim_year = claim_year_match.group(1)
            for sent in sentences_to_check:
                # Use full sentence without splitting on 'and' to preserve multi-event context
                clauses = re.split(r'\.\.\.|\.|\n|;|,|(?<=\d)\s+and\s+(?=\w)', sent, flags=re.IGNORECASE)
                for clause in clauses:
                    clause_lower = clause.lower()
                    if claim_year in clause_lower and claim_loc in clause_lower:
                        return claim_loc  # claim city co-occurs with claim year in same clause

    # Priority 1: Sentence that contains action/synonym AND a city
    if action:
        best_city = None
        best_dist = 9999
        for sent in sentences_to_check:
            # Do NOT split on 'and' here: splitting breaks "Mumbai in 2020 and Hyderabad in 2026"
            clauses = re.split(r'\.\.\.|\.|\n|;|,|(?<!\d)\band\b(?!\s*\d)', sent)
            for clause in clauses:
                clause_lower = clause.lower()
                # Use action synonyms for matching
                action_synonyms = ACTION_SYNONYMS.get(action, [action])
                if any(re.search(r'\b' + re.escape(syn) + r'\b', clause_lower) for syn in action_synonyms):
                    act_match = next((re.search(r'\b' + re.escape(syn) + r'\b', clause_lower) for syn in action_synonyms if re.search(r'\b' + re.escape(syn) + r'\b', clause_lower)), None)
                    act_pos = act_match.start() if act_match else 0
                    
                    for city in cities:
                        m = re.search(r'\b' + re.escape(city) + r'\b', clause_lower)
                        if m:
                            dist = m.start() - act_pos
                            penalty = 0 if dist >= 0 else 1000
                            tot_dist = penalty + abs(dist)
                            if claim_loc and are_locations_equivalent(claim_loc, city):
                                return city
                            if tot_dist < best_dist:
                                best_dist = tot_dist
                                best_city = city
        if best_city:
            return best_city
                        
    # Priority 2: Clause with subject + object/topic AND a city
    subject = claim_struct.get("subject", "").lower()
    obj = claim_struct.get("object", "").lower()
    for sent in sentences_to_check:
        clauses = re.split(r'\.\.\.|\.|\n|;|\band\b|\bwhile\b|\bafter\b|\bbefore\b', sent)
        for clause in clauses:
            clause_lower = clause.lower()
            if (subject and subject in clause_lower) or (obj and any(w in clause_lower for w in obj.split() if len(w) > 3)):
                found_cities = []
                for city in cities:
                    m = re.search(r'\b' + re.escape(city) + r'\b', clause_lower)
                    if m:
                        found_cities.append((m.start(), city))
                if found_cities:
                    found_city_names = [c[1] for c in found_cities]
                    if claim_loc:
                        for f_city in found_city_names:
                            if are_locations_equivalent(claim_loc, f_city):
                                return f_city
                    found_cities.sort()
                    return found_cities[0][1]
                    
    # Priority 3: Fallback to any sentence with subject + city
    for sent in sentences_to_check:
        sent_lower = sent.lower()
        if subject and subject in sent_lower:
            found_cities = []
            for city in cities:
                m = re.search(r'\b' + re.escape(city) + r'\b', sent_lower)
                if m:
                    found_cities.append((m.start(), city))
            if found_cities:
                found_city_names = [c[1] for c in found_cities]
                if claim_loc:
                    for f_city in found_city_names:
                        if are_locations_equivalent(claim_loc, f_city):
                            return f_city
                found_cities.sort()
                return found_cities[0][1]

    return None


def detect_role_reversal(claim_struct, relevant_sentences, full_text):
    """
    Detects role reversal within claim-relevant sentences.
    """
    subject = claim_struct.get("subject", "").lower()
    action = claim_struct.get("action", "").lower()
    obj = claim_struct.get("object", "").lower()
    countries = [c.lower() for c in claim_struct.get("countries", [])]
    entities = [e.lower() for e in claim_struct.get("entities", [])]
    
    target_obj = obj
    if not target_obj:
        other_countries = [c for c in countries if c != subject]
        if other_countries:
            target_obj = other_countries[0]
        else:
            other_entities = [e for e in entities if e != subject]
            if other_entities:
                target_obj = other_entities[0]
                
    if not subject or not target_obj:
        return False
        
    candidates = relevant_sentences if relevant_sentences else [full_text]
    # Sentence-tokenize to avoid positional false-positives from concatenated strings.
    # First normalize double-period artifacts (e.g. "title.. scraped body") that NLTK
    # does not split, then sentence-tokenize each clean chunk.
    sentences_to_check = []
    for chunk in candidates:
        # Replace .. or .  . patterns with a single sentence boundary
        normalized = re.sub(r'\.{2,}\s*', '. ', chunk).strip()
        try:
            sentences_to_check.extend(nltk.sent_tokenize(normalized))
        except Exception:
            sentences_to_check.extend(re.split(r'(?<=[.!?])\s+', normalized))

    for sent in sentences_to_check:
        sent_lower = sent.lower()
        if subject in sent_lower and target_obj in sent_lower:
            obj_idx = sent_lower.find(target_obj)
            sub_idx = sent_lower.find(subject)
            if obj_idx < sub_idx:
                if action and action_matches(action, sent_lower):
                    return True
                if any(verb in sent_lower for verb in ["attacked", "struck", "hit", "launched"]):
                    return True
    return False

def are_locations_equivalent(loc1, loc2):
    if not loc1 or not loc2:
        return False
    l1 = loc1.lower().strip()
    l2 = loc2.lower().strip()
    if l1 == l2:
        return True
    bengaluru_syns = {"bengaluru", "bangalore"}
    if l1 in bengaluru_syns and l2 in bengaluru_syns:
        return True
    delhi_syns = {"delhi", "new delhi"}
    if l1 in delhi_syns and l2 in delhi_syns:
        return True
    gurgaon_syns = {"gurgaon", "gurugram"}
    if l1 in gurgaon_syns and l2 in gurgaon_syns:
        return True
    return False

COMPLETED_VERBS = {
    "opened", "open", "opens", "launch", "launches", "launched", "inaugurate", "inaugurated",
    "unveil", "unveiled", "went live", "became operational", "expands", "expand", "agreed",
    "agree", "agrees", "commit", "commits", "committed", "pledge", "pledges", "pledged",
    "signed", "attacked", "attack", "attacks", "strike", "strikes", "struck", "hit"
}

ANNOUNCEMENT_VERBS = {"announced", "said", "revealed", "stated"}
FUTURE_INTENT_VERBS = {"plans", "will", "expects", "aims", "intends", "scheduled", "planning"}
PROPOSAL_VERBS = {"proposed", "considering", "considered", "discussing", "discuss"}

def is_aspect_compatible(claim_action, text_lower):
    if not claim_action:
        return True
    action_clean = claim_action.lower().strip()
    
    if action_clean in COMPLETED_VERBS:
        future_patterns = [
            r'\bplans\s+to\s+\w+\b', r'\bwill\s+\w+\b', r'\bintends?\s+to\s+\w+\b', 
            r'\bannounced\s+plans?\s+to\s+\w+\b', r'\bconsidering\s+\w+\b', r'\bproposed\s+to\s+\w+\b'
        ]
        has_future_or_proposal = any(re.search(pat, text_lower) for pat in future_patterns)
        
        masked_text = text_lower
        for pat in future_patterns:
            masked_text = re.sub(pat, ' ', masked_text)
            
        completed_past_verbs = {"opened", "launched", "inaugurated", "went live", "became operational", "agreed", "committed", "signed", "attacked", "struck"}
        has_completed_verb = any(re.search(r'\b' + re.escape(v) + r'\b', masked_text) for v in completed_past_verbs)
        
        if has_future_or_proposal and not has_completed_verb:
            return False
            
    return True

def parse_factual_numbers(text):
    """
    Parses structured numerical expressions: value, unit, currency, multiplier, context.
    Prevents matching $50B with 50 jobs, 50%, 50M, or 50 users.
    """
    numbers = []
    pattern = r'(?:\$|USD\s*|€|£)?\b(\d+(?:\.\d+)?)\s*(billion|million|trillion|percent|%|B|M|jobs|users|units)?\b'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    for m in matches:
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        full_match = m.group(0).lower()
        currency = "$" if ("$" in full_match or "usd" in full_match) else ("%" if "%" in full_match else "")
        if "percent" in full_match:
            currency = "%"
        
        scale = 1.0
        if unit in ["billion", "b"]:
            scale = 1e9
        elif unit in ["million", "m"]:
            scale = 1e6
        elif unit in ["trillion"]:
            scale = 1e12
            
        numbers.append({
            "value": val,
            "scale": scale,
            "normalized_val": val * scale,
            "unit": unit,
            "currency": currency,
            "context": full_match
        })
    return numbers

def numbers_match(claim_num, evidence_num):
    """
    Checks if a claim number concept matches an evidence number concept.
    """
    c_val = claim_num.get("value", 0.0)
    e_val = evidence_num.get("value", 0.0)
    
    c_unit = claim_num.get("unit", "").lower()
    e_unit = evidence_num.get("unit", "").lower()
    
    c_curr = claim_num.get("currency", "")
    e_curr = evidence_num.get("currency", "")
    
    # Currency vs percentage mismatch ($50B vs 50%)
    if c_curr == "$" and e_curr == "%":
        return False
    if c_curr == "%" and e_curr != "%":
        return False
        
    scale_b = {"billion", "b"}
    scale_m = {"million", "m"}
    if c_unit in scale_b and e_unit in scale_m:
        return False
    if c_unit in scale_m and e_unit in scale_b:
        return False
    if c_unit in scale_b and e_unit in ["jobs", "users", "units"]:
        return False
        
    c_norm = claim_num.get("normalized_val", c_val)
    e_norm = evidence_num.get("normalized_val", e_val)
    
    if abs(c_norm - e_norm) < 1e-3:
        return True
    if c_val == e_val and (c_unit == e_unit or (not c_unit and not e_unit)):
        return True
        
    return False

EVENT_STATES = {
    "UNDER_CONSTRUCTION": ["under construction", "constructing", "construction of", "building"],
    "CANCELLED": ["cancelled", "canceled", "abandoned", "scrapped"],
    "PLANNED": ["plans", "will", "expects", "aims", "intends", "scheduled", "planning"],
    "PROPOSAL": ["proposed", "considering", "considered", "discussing"],
    "ANNOUNCEMENT": ["announced", "said", "revealed", "stated"],
    "COMPLETED": ["opened", "open", "opens", "launched", "launched", "completed", "became operational", "inaugurated", "went live", "agreed", "signed"]
}

def classify_event_state(sentence):
    sent_lower = sentence.lower()
    for state, keywords in EVENT_STATES.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', sent_lower):
                return state
    return "UNKNOWN"

def are_event_states_compatible(claim_state, evidence_state, text_lower=""):
    if claim_state == "COMPLETED":
        completed_verbs = {"opened", "launched", "inaugurated", "went live", "became operational", "completed", "agreed", "signed"}
        has_completed_verb = any(re.search(r'\b' + re.escape(v) + r'\b', text_lower) for v in completed_verbs)
        
        if evidence_state in ["UNDER_CONSTRUCTION", "PLANNED", "PROPOSAL"] and not has_completed_verb:
            return False, "UNVERIFIED"
        if evidence_state == "CANCELLED":
            return False, "CONTRADICTED"
    return True, "COMPATIBLE"

def is_exact_entity_match(claim_entity, text):
    c_lower = claim_entity.lower().strip()
    t_lower = text.lower()
    
    if c_lower == "south korea":
        if "north korea" in t_lower and "south korea" not in t_lower:
            return False
    if c_lower == "north korea":
        if "south korea" in t_lower and "north korea" not in t_lower:
            return False
            
    if "apple inc" in c_lower or "apple" in c_lower:
        if "apple corps" in t_lower and "apple inc" not in t_lower:
            return False
            
    return True

ALLEGATION_KEYWORDS = {"claims", "claimed", "alleges", "alleged", "rumors", "rumored", "purported"}
DIRECT_REPORTERS = {"reuters", "associated press", "ap news", "bbc", "the hindu", "indian express", "official", "spokesperson"}

def classify_attribution(sentence, source_name=""):
    sent_lower = sentence.lower()
    src_lower = source_name.lower()
    
    is_direct_reporter = any(r in src_lower for r in DIRECT_REPORTERS)
    has_allegation = any(re.search(r'\b' + re.escape(kw) + r'\b', sent_lower) for kw in ALLEGATION_KEYWORDS)
    
    if has_allegation and not is_direct_reporter:
        return "ALLEGATION"
    return "DIRECT"

DOUBLE_NEGATION_PATTERNS = [
    r'\bdid\s+not\s+deny\b', r'\bdid\s+not\s+refuse\b', r'\bnever\s+denied\b', r'\bnot\s+opposed\b'
]

def check_double_negation(sent_lower):
    for pat in DOUBLE_NEGATION_PATTERNS:
        if re.search(pat, sent_lower):
            return True
    return False

def calculate_source_quality(source_name):
    """
    3-Tier Conservative Source Quality Weighting.
    Tier 1 (3.0x): Reuters, AP, BBC, The Hindu, Indian Express, Govt, NASA, ISRO, RBI.
    Tier 2 (1.5x): Established major news publications.
    Tier 3 (0.5x): Unknown / blog sources.
    """
    if not source_name:
        return 0.5
    src_lower = source_name.lower()
    
    tier1 = [
        "reuters", "associated press", "ap news", "bbc", "the hindu", "the indian express", 
        "official government", "official source", "nasa", "isro", "rbi", "opec", "who", "un"
    ]
    for t1 in tier1:
        if t1 in src_lower:
            return 3.0
            
    tier2 = [
        "economic times", "financial times", "cnbc", "bloomberg", "wsj", "wall street journal", 
        "new york times", "nyt", "microsoft", "times of india", "hindustan times", "ndtv", "afp", "quantum commodity"
    ]
    for t2 in tier2:
        if t2 in src_lower:
            return 1.5
            
    return 0.5

def is_syndicated_copy(art, processed_articles):
    title_lower = art.get("title", "").lower()
    desc_lower = art.get("description", "").lower()
    full = title_lower + " " + desc_lower
    
    if any(phrase in full for phrase in ["according to reuters", "reuters reports", "according to ap", "ap reports", "according to bbc"]):
        return True
        
    for prev in processed_articles:
        p_title = prev.get("title", "").lower()
        w1 = set(title_lower.split())
        w2 = set(p_title.split())
        if w1 and w2 and len(w1.intersection(w2)) / min(len(w1), len(w2)) >= 0.75:
            return True
    return False

def verify_claim_against_evidence(claim_struct, articles, diagnostic_info=None):
    """
    Evaluates evidence sentences/body texts against the claim structure.
    Determines SUPPORTED, CONTRADICTED, or UNVERIFIED statuses with claim-scoped reasoning.
    """
    def log(msg):
        if diagnostic_info is not None:
            diagnostic_info.append(msg)
        print(msg)

    log(f"verify_claim_against_evidence: Evaluating claim structure: {claim_struct}")
    log(f"verify_claim_against_evidence: Received {len(articles)} articles.")

    if not articles:
        log("verify_claim_against_evidence: No articles provided. Returning UNVERIFIED.")
        return {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "explanation": "No relevant and reliable online news coverage could be found to verify this claim.",
            "verdict_reason": "No news coverage found.",
            "attribute_breakdown": {},
            "sources": [],
            "diagnostics": diagnostic_info
        }
        
    support_count = 0
    contradiction_count = 0
    neutral_count = 0
    
    total_support_score = 0.0
    total_contradiction_score = 0.0
    
    supporting_sources = []
    contradicting_sources = []
    processed_articles = []
    
    claim_subject = claim_struct.get("subject", "").lower()
    claim_action = claim_struct.get("action", "").lower()
    claim_object = claim_struct.get("object", "").lower()
    claim_negation = claim_struct.get("negated", False)
    claim_location = claim_struct.get("location").lower() if claim_struct.get("location") else None
    claim_date = claim_struct.get("date").lower() if claim_struct.get("date") else None
    claim_numbers = claim_struct.get("numbers", [])
    
    conflict_details = {
        "role_reversal": None,
        "negation": None,
        "location": None,
        "date": None,
        "numbers": None
    }
    
    for idx, art in enumerate(articles[:6]):
        url = art["link"]
        scraped_text = ""
        log(f"Article [{idx+1}]: Source: '{art.get('source')}', Title: '{art.get('title')}'")
        log(f"  URL: {url}")
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            response = requests.get(url, headers=headers, timeout=5)
            log(f"  Scrape HTTP Status: {response.status_code}")
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
                scraped_text = " ".join([p for p in paragraphs if p])
        except Exception:
            pass
            
        evidence_corpus = art.get("title", "") + ". " + art.get("description", "")
        if scraped_text:
            evidence_corpus += ". " + scraped_text
            
        evidence_corpus_lower = evidence_corpus.lower()
        src_weight = calculate_source_quality(art.get("source", ""))
        
        # Check syndication status
        is_synd = is_syndicated_copy(art, processed_articles)
        if is_synd:
            src_weight *= 0.2  # Syndicated copies carry reduced weight
            
        processed_articles.append(art)
        
        # 1. Strict Entity Binding Match Check
        entity_missing = False
        gate_reason = ""
        
        for country in claim_struct.get("countries", []):
            if not match_country_in_text(country, evidence_corpus_lower):
                entity_missing = True
                gate_reason = f"Country '{country}' not found in text."
                break
                
        if not entity_missing:
            # Exact Entity Boundary Check
            for ent in claim_struct.get("entities", []):
                if not is_exact_entity_match(ent, evidence_corpus):
                    entity_missing = True
                    gate_reason = f"Entity '{ent}' not found in text."
                    break
                    
        if not entity_missing:
            # Organization Hard Gate: If claim has specific orgs (e.g. ISRO, RBI, SpaceX), evidence must mention at least one
            claim_orgs = [o.lower() for o in claim_struct.get("organizations", [])]
            if claim_orgs:
                if not any(re.search(r'\b' + re.escape(o) + r'\b', evidence_corpus_lower) for o in claim_orgs):
                    entity_missing = True
                    gate_reason = f"None of the required organizations {claim_orgs} found in text."
                    
        if not entity_missing:
            crucial_topics = ["nuclear", "ethanol", "data center", "datacenter", "aliens", "ufo", "mars", "water", "satellite"]
            for topic in crucial_topics:
                if topic in claim_struct.get("text", "").lower():
                    if topic in ["data center", "datacenter"]:
                        if "data center" not in evidence_corpus_lower and "datacenter" not in evidence_corpus_lower and "cloud" not in evidence_corpus_lower:
                            entity_missing = True
                            gate_reason = "Crucial topic 'data center' or 'cloud' not found."
                            break
                    else:
                        if topic not in evidence_corpus_lower:
                            entity_missing = True
                            gate_reason = f"Crucial topic '{topic}' not found."
                            break
                            
        if entity_missing:
            log(f"  -> Neutral: Entity gate failed. Reason: {gate_reason}")
            art["type"] = "Neutral"
            neutral_count += 1
            continue
        log("  -> Entity gate passed.")
            
        # Get claim-relevant sentences from evidence
        relevant_sentences = get_relevant_evidence_sentences(claim_struct, evidence_corpus)
        relevant_corpus = " ".join(relevant_sentences) if relevant_sentences else evidence_corpus
        relevant_corpus_lower = relevant_corpus.lower()
        log(f"  Extracted {len(relevant_sentences)} claim-relevant sentences.")
        
        # Attribution Check
        attr_class = classify_attribution(relevant_corpus, art.get("source", ""))
        if attr_class == "ALLEGATION":
            log("  -> Neutral: Attribution classified as ALLEGATION.")
            art["type"] = "Neutral"
            neutral_count += 1
            continue
            
        # Action Aspect & Event State Compatibility Check
        if not is_aspect_compatible(claim_action, relevant_corpus_lower):
            log(f"  -> Neutral: Action aspect compatibility failed for '{claim_action}'.")
            art["type"] = "Neutral"
            neutral_count += 1
            continue
            
        ev_state = classify_event_state(relevant_corpus)
        claim_state = classify_event_state(claim_struct["text"])
        state_compat, state_verdict = are_event_states_compatible(claim_state, ev_state, relevant_corpus_lower)
        if not state_compat:
            log(f"  -> Event state mismatch: claim={claim_state}, evidence={ev_state}. Verdict: {state_verdict}")
            if state_verdict == "CONTRADICTED":
                art["type"] = "Contradicting"
                contradiction_count += 1
                total_contradiction_score += 1.5 * src_weight
                contradicting_sources.append(art)
                continue
            else:
                art["type"] = "Neutral"
                neutral_count += 1
                continue
            
        # 2. Subject-Object Role Reversal Check (sentence-scoped)
        role_reversal = detect_role_reversal(claim_struct, relevant_sentences, evidence_corpus)
        if role_reversal:
            log("  -> Contradicting: Subject-Object Role Reversal detected.")
            art["type"] = "Contradicting"
            contradiction_count += 1
            total_contradiction_score += 1.5 * src_weight
            contradicting_sources.append(art)
            if not conflict_details["role_reversal"]:
                conflict_details["role_reversal"] = {
                    "claim_subject": claim_struct.get("subject"),
                    "claim_object": claim_struct.get("object"),
                    "evidence_subject": claim_struct.get("object"),
                    "evidence_object": claim_struct.get("subject")
                }
            continue
            
        # 3. Claim-Scoped Negation Check
        evidence_negated = detect_claim_scoped_negation(claim_struct, relevant_sentences, evidence_corpus)
        if claim_negation != evidence_negated:
            if check_double_negation(relevant_corpus_lower):
                # Double negation / ambiguity: fallback to neutral instead of false contradiction
                log("  -> Neutral: Negation mismatch but double-negation/ambiguity triggered fallback to Neutral.")
                art["type"] = "Neutral"
                neutral_count += 1
                continue
            log(f"  -> Contradicting: Negation mismatch (claim negated={claim_negation}, evidence negated={evidence_negated}).")
            art["type"] = "Contradicting"
            contradiction_count += 1
            total_contradiction_score += 1.5 * src_weight
            contradicting_sources.append(art)
            if not conflict_details["negation"]:
                conflict_details["negation"] = {
                    "claim": "Negated" if claim_negation else "Asserted",
                    "evidence": "Negated" if evidence_negated else "Asserted"
                }
            continue
            
        # 4. Event-Scoped Location Verification Check
        location_conflict = False
        found_evidence_city = None
        if claim_location:
            found_evidence_city = extract_event_location(claim_struct, relevant_sentences, evidence_corpus)
            log(f"  Claim location: '{claim_location}', extracted evidence location: '{found_evidence_city}'")
            if found_evidence_city and not are_locations_equivalent(found_evidence_city, claim_location):
                location_conflict = True
                
        if location_conflict:
            log("  -> Contradicting: Location mismatch.")
            art["type"] = "Contradicting"
            contradiction_count += 1
            total_contradiction_score += 1.5 * src_weight
            contradicting_sources.append(art)
            if not conflict_details["location"]:
                conflict_details["location"] = {
                    "claim": claim_struct.get("location"),
                    "evidence": found_evidence_city.capitalize() if found_evidence_city else "Unknown"
                }
            continue
            
        # 5. Event-Scoped Temporal / Date Verification Check
        date_conflict = False
        ev_year = None
        if claim_date:
            ev_year = extract_event_date(claim_struct, relevant_sentences, evidence_corpus)
            log(f"  Claim date: '{claim_date}', extracted evidence date: '{ev_year}'")
            if not ev_year:
                pub_year_match = re.search(r'\b(19\d\d|20\d\d)\b', art.get("pub_date", ""))
                if pub_year_match:
                    ev_year = pub_year_match.group(0)
                    log(f"  Extracted date from article publication date: '{ev_year}'")
                    
            
            # For relative dates (no 4-digit year), resolve against article pub_date
            claim_year_match = re.search(r'\b(19\d{2}|20\d{2})\b', str(claim_date)) if claim_date else None
            effective_claim_date = claim_date
            if claim_date and not claim_year_match:
                # Relative date: resolve using pub_date
                pub_year_match2 = re.search(r'\b(19\d{2}|20\d{2})\b', art.get("pub_date", ""))
                if pub_year_match2:
                    effective_claim_date = pub_year_match2.group(0)
                    log(f"  Resolved relative claim date to effective date: '{effective_claim_date}'")

            if ev_year and effective_claim_date and not are_dates_compatible(effective_claim_date, ev_year):
                date_conflict = True
                        
        if date_conflict:
            log(f"  -> Contradicting: Date mismatch (Claim date: {effective_claim_date}, Evidence date: {ev_year}).")
            art["type"] = "Contradicting"
            contradiction_count += 1
            total_contradiction_score += 1.5 * src_weight
            contradicting_sources.append(art)
            if not conflict_details["date"]:
                conflict_details["date"] = {
                    "claim": claim_struct.get("date"),
                    "evidence": ev_year if ev_year else "Unknown"
                }
            continue
            
        # 6. Structured Support Overlap comparison using factual_tokens()
        claim_toks = factual_tokens(claim_struct["text"])
        ev_toks = set(factual_tokens(relevant_corpus))
        
        user_words = set(claim_toks)
        overlap_ratio = len(user_words.intersection(ev_toks)) / len(user_words) if user_words else 0.0
        
        sub_ok = (not claim_subject) or (claim_subject in relevant_corpus_lower)
        act_ok = action_matches(claim_action, relevant_corpus_lower)
        loc_ok = (not claim_location) or (found_evidence_city and are_locations_equivalent(found_evidence_city, claim_location)) or (not found_evidence_city)
        
        log(f"  Overlap overlap_ratio: {overlap_ratio:.2%}, sub_ok: {sub_ok}, act_ok: {act_ok}, loc_ok: {loc_ok}")
        if (overlap_ratio >= 0.25 and sub_ok and act_ok and loc_ok) or (overlap_ratio >= 0.35 and act_ok):
            log("  -> Supporting article found.")
            art["type"] = "Supporting"
            support_count += 1
            total_support_score += 1.0 * src_weight
            supporting_sources.append(art)
        else:
            log("  -> Neutral: Overlap/compatibility checks failed.")
            art["type"] = "Neutral"
            neutral_count += 1
            
    total = support_count + contradiction_count + neutral_count
    
    # Construct breakdown
    breakdown = {}
    
    if claim_struct.get("subject"):
        breakdown["subject"] = {
            "name": "Subject",
            "claim": claim_struct["subject"],
            "evidence": claim_struct["subject"],
            "status": "MATCH"
        }
        if conflict_details["role_reversal"]:
            breakdown["subject"]["evidence"] = conflict_details["role_reversal"]["evidence_subject"]
            breakdown["subject"]["status"] = "CONFLICT"
            
    if claim_struct.get("action"):
        breakdown["action"] = {
            "name": "Action",
            "claim": claim_struct["action"],
            "evidence": claim_struct["action"],
            "status": "MATCH"
        }
        
    if claim_struct.get("object"):
        breakdown["object"] = {
            "name": "Object",
            "claim": claim_struct["object"],
            "evidence": claim_struct["object"],
            "status": "MATCH"
        }
        if conflict_details["role_reversal"]:
            breakdown["object"]["evidence"] = conflict_details["role_reversal"]["evidence_object"]
            breakdown["object"]["status"] = "CONFLICT"
            
    if claim_struct.get("location"):
        breakdown["location"] = {
            "name": "Location",
            "claim": claim_struct["location"],
            "evidence": claim_struct["location"],
            "status": "MATCH"
        }
        if conflict_details["location"]:
            breakdown["location"]["evidence"] = conflict_details["location"]["evidence"]
            if conflict_details["location"].get("status") == "UNSUPPORTED":
                breakdown["location"]["status"] = "UNSUPPORTED"
            else:
                breakdown["location"]["status"] = "CONFLICT"
                
    if claim_struct.get("date"):
        breakdown["date"] = {
            "name": "Date",
            "claim": claim_struct["date"],
            "evidence": claim_struct["date"],
            "status": "MATCH"
        }
        if conflict_details["date"]:
            breakdown["date"]["evidence"] = conflict_details["date"]["evidence"]
            breakdown["date"]["status"] = "CONFLICT"
            
    def _fmt_num(n):
        cur = n.get("currency", "")
        val = int(n["value"]) if n["value"] == int(n["value"]) else n["value"]
        unit = (" " + n["unit"].capitalize()) if n.get("unit") else ""
        return f"{cur}{val}{unit}"
    if claim_numbers:
        num_label = ", ".join([_fmt_num(n) for n in claim_numbers])
        breakdown["numbers"] = {
            "name": "Numbers",
            "claim": num_label,
            "evidence": num_label,
            "status": "MATCH"
        }
        if conflict_details["numbers"]:
            breakdown["numbers"]["evidence"] = conflict_details["numbers"]["evidence"]
            breakdown["numbers"]["status"] = "UNSUPPORTED"
            
    breakdown["negation"] = {
        "name": "Negation",
        "claim": "Negated" if claim_negation else "Asserted",
        "evidence": "Negated" if claim_negation else "Asserted",
        "status": "MATCH"
    }
    if conflict_details["negation"]:
        breakdown["negation"]["evidence"] = conflict_details["negation"]["evidence"]
        breakdown["negation"]["status"] = "CONFLICT"

    # Default values
    verdict = "UNVERIFIED"
    confidence = 0.0
    v_reason = "No relevant news coverage found."
    expl = "No relevant and reliable online news coverage could be found to verify this claim."
    sources = articles[:6]
    
    if total == 0:
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": expl,
            "verdict_reason": v_reason,
            "attribute_breakdown": breakdown,
            "sources": []
        }
        
    # Evidence-Aware Aggregation Rule:
    has_strong_contradiction = contradiction_count > 0 and (total_contradiction_score >= total_support_score or support_count == 0)
    
    if has_strong_contradiction:
        verdict = "CONTRADICTED"
        confidence = min(0.95, 0.50 + (contradiction_count / total * 0.45))
        sources = contradicting_sources
        
        if conflict_details["role_reversal"]:
            v_reason = "Subject-object role reversal."
            expl = f"Subject/object role reversal: Claimed '{conflict_details['role_reversal']['claim_subject']}' acted on '{conflict_details['role_reversal']['claim_object']}', but evidence shows role reversal."
        elif conflict_details["negation"]:
            v_reason = "Negation/action contradiction."
            expl = f"Negation mismatch: Claimed fact is {conflict_details['negation']['claim'].lower()}, but evidence is {conflict_details['negation']['evidence'].lower()}."
        elif conflict_details["location"]:
            v_reason = "Location contradiction."
            expl = f"Location mismatch: Claimed location '{conflict_details['location']['claim']}' contradicts the established location '{conflict_details['location']['evidence']}'."
        elif conflict_details["date"]:
            v_reason = "Date contradiction."
            expl = f"Date mismatch: Claimed event date '{conflict_details['date']['claim']}' contradicts the established date '{conflict_details['date']['evidence']}'."
        else:
            v_reason = "Factual inconsistency detected."
            expl = "Factual inconsistencies (e.g. location, date, negation, or role reversal) were detected in the news coverage."
            
    elif support_count > 0:
        has_claim_numbers = len(claim_numbers) > 0
        numbers_validated = False
        if has_claim_numbers:
            for art in supporting_sources:
                evidence_text = art["title"] + " " + art["description"]
                ev_numbers = parse_factual_numbers(evidence_text)
                all_nums_found = True
                for num in claim_numbers:
                    if not any(numbers_match(num, ev_num) for ev_num in ev_numbers):
                        all_nums_found = False
                        break
                if all_nums_found:
                    numbers_validated = True
                    break
                    
        if has_claim_numbers and not numbers_validated:
            verdict = "MIXED"
            confidence = 0.65
            sources = supporting_sources
            v_reason = "Numerical claim unsupported."
            expl = f"Core event is supported by news coverage, but the claimed numerical figures ({breakdown['numbers']['claim']}) are not independently verified."
            
            conflict_details["numbers"] = {
                "claim": breakdown["numbers"]["claim"],
                "evidence": "No matching figure found"
            }
            breakdown["numbers"]["evidence"] = "No matching figure found"
            breakdown["numbers"]["status"] = "UNSUPPORTED"
        else:
            verdict = "VERIFIED"
            confidence = min(0.98, 0.55 + (support_count / total * 0.44))
            sources = supporting_sources
            v_reason = "All attributes verified."
            expl = "Reliable online news coverage independently supports all factual attributes of the claim."
            
    else:
        verdict = "UNVERIFIED"
        confidence = 0.0
        sources = articles[:6]
        
        if conflict_details["location"] and conflict_details["location"].get("status") == "UNSUPPORTED":
            v_reason = "Location missing from headlines."
            expl = f"Relevant news coverage was found, but it does not confirm the claimed location '{claim_struct.get('location')}'."
        else:
            v_reason = "Insufficient evidence."
            expl = "There is insufficient reliable online news coverage to confidently verify or contradict this claim."
            
    log(f"Aggregation results: support={support_count}, contradiction={contradiction_count}, neutral={neutral_count}")
    log(f"Final Verdict: '{verdict}' (Confidence: {confidence:.2%}), Reason: {v_reason}")
    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": expl,
        "verdict_reason": v_reason,
        "attribute_breakdown": breakdown,
        "sources": sources,
        "diagnostics": diagnostic_info
    }

def predict_with_saved_model(text):
    try:
        model, vectorizer = _load_model_and_vectorizer()
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {e}"}
        
    try:
        from preprocessing import clean_text
        cleaned = clean_text(text)
        if not cleaned:
            return {"status": "empty", "message": "Input text is too short to analyze."}
            
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vectorized)[0][prediction]
        elif hasattr(model, "decision_function"):
            import math
            decision_val = model.decision_function(vectorized)[0]
            prob_real = 1 / (1 + math.exp(-decision_val))
            confidence = prob_real if prediction == 1 else (1.0 - prob_real)
            
        label = "Real" if prediction == 1 else "Fake"
        return {
            "status": "success",
            "label": label,
            "confidence": confidence
        }
    except Exception as e:
        return {"status": "error", "message": f"Inference execution failed: {e}"}

def verify_local_claim(article_title, csv_path="recent_claims.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
        
    if df.empty or "claim" not in df.columns or "verdict" not in df.columns:
        return None
        
    try:
        vectorizer = TfidfVectorizer()
        claims = df["claim"].tolist()
        all_texts = [article_title] + claims
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except ValueError:
        return None
    except Exception:
        return None
    
    if len(similarities) == 0:
        return None
        
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    row = df.iloc[best_idx]
    return {
        "claim": row["claim"],
        "verdict": row["verdict"],
        "source": row.get("source", "Local DB"),
        "score": best_score
    }

def compute_style_score(title, text):
    score = 0
    reasons = []
    
    exclamation_count = title.count("!") + text.count("!")
    if exclamation_count > 3:
        score += 20
        reasons.append("High frequency of exclamation marks")
        
    caps_words = re.findall(r'\b[A-Z]{3,}\b', title + " " + text)
    if len(caps_words) >= 3:
        score += 25
        reasons.append(f"Excessive capitalized words (e.g., {', '.join(caps_words[:3])})")
        
    clickbait_triggers = [
        "shocking", "unbelievable", "won't believe", "miracle cure",
        "secret leaked", "they don't want you to know", "viral claim",
        "100% working", "pass this on", "must watch", "exposed"
    ]
    
    combined_text = (title + " " + text).lower()
    found_triggers = [phrase for phrase in clickbait_triggers if phrase in combined_text]
    
    if found_triggers:
        score += min(15 * len(found_triggers), 45)
        reasons.append(f"Contains known clickbait indicators: {', '.join(found_triggers)}")
        
    return score, reasons
