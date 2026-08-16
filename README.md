# Evidence-Based News Verification System

An automated, real-time factual claims verification system that evaluates editorial text and statements against live online news sources and analyzes their writing style using an offline machine learning model.

---

## 📌 Problem Statement
In the modern digital information ecosystem, the rapid spread of misinformation, clickbait, and fake news poses a significant challenge to public trust and decision-making. Standard machine learning classifiers often predict whether an article is "real" or "fake" based purely on writing style (stylometry), which can be bypassed by sophisticated misinformation. To address this, this project implements a dual-approach verification system that combines an **offline style classifier** with a **real-time, rule-based verification engine** that crawls live news coverage to verify individual factual claims.

## 🎯 Project Objective
The objective of this system is to accept user-provided paragraphs or articles, parse them into distinct factual claims, dynamically retrieve live news evidence from reputable sources, cross-reference factual attributes (subject, action, object, location, date, negation, and numerical details), and output a structured verification verdict.

---

## 🏆 Key Features

* **Atomic Claim Extraction**: Automatically parses complex paragraphs into individual, testable factual assertions (atomic claims) using NLTK sentence tokenization and basic anaphora resolution.
* **Live Evidence Gathering**: Programmatically queries Google News RSS feeds to fetch the latest relevant news articles and coverage.
* **Robust Web Scraping**: Dynamically scrapes article content from retrieved search links using desktop user-agent emulation and request timeout protection to handle anti-bot or network errors gracefully.
* **Rule-Based Factual Cross-Referencing**:
  * **Entity & Organization Gates**: Ensures that the search evidence actually references the specific organizations, countries, and entities mentioned in the claim.
  * **Temporal & Date Check**: Extracts event dates and checks them against the claim year, supporting full ISO dates, English text dates, and relative expressions (like "yesterday").
  * **Location Matching**: Extracts and matches cities (with synonym mappings like Bangalore/Bengaluru and Delhi/New Delhi).
  * **Negation and Role-Reversal Analysis**: Scopes negation cues to claims and detects reversed agent-object interactions (e.g., "Company X acquired Company Y" vs "Company Y acquired Company X").
  * **Numerical & Scale Checks**: Compares quantities and scale modifiers (e.g., "$50 million" vs "$50 billion" or "50%" vs "50").
* **Offline Stylometric Classifier**: An auxiliary Linear Support Vector Machine (SVM) model that inspects the overall text block for clickbait, hyper-partisanship, and stylistic sensationalism.
* **Deterministic Verification Verdicts**: Produces clear, explainable statuses:
  * 🟢 **VERIFIED / SUPPORTED**: Core factual attributes match reliable live news reports.
  * 🔴 **CONTRADICTED**: Direct factual contradictions (such as conflicting locations, dates, or negation states) found in reports.
  * 🟠 **MIXED / PARTIALLY VERIFIED**: Core event matches but specific details (e.g., numerical figures) cannot be verified or differ.
  * 🟡 **UNVERIFIED**: Insufficient online coverage found to support or contradict the claim.

---

## ⚙️ Architecture & Pipeline

The system processes claims through the following automated verification pipeline:

```text
User Article / Text Input
          ↓
Atomic Claim Extraction (NLTK Tokenizer)
          ↓
NLP Claim Parsing (Entities, Organizations, Dates, Locations)
          ↓
Query Generation (Search Key Formulation)
          ↓
Google News RSS Retrieval (Public Search Feed)
          ↓
Article/Snippet Extraction (BS4 HTML parser)
          ↓
Entity & Organization Matching (Entity Gates)
          ↓
Event / Location / Date Matching (Temporal and Spatial Checks)
          ↓
Negation & Role Analysis (Claim-Scoped Negation & Positional Index checks)
          ↓
Numerical Verification (Scale and Value cross-checks)
          ↓
Source Quality Weighting (Credibility and Syndication checks)
          ↓
Evidence Aggregation (Score calculation)
          ↓
Final Verdict (VERIFIED, CONTRADICTED, MIXED, or UNVERIFIED)
          ↓
Streamlit UI Display (Color-coded verdicts with attribute breakdown tables)
```

---

## 🛠️ Technologies Used

* **Programming Language**: Python (tested on Python 3.9 - 3.14)
* **Web Dashboard**: Streamlit
* **Natural Language Processing**: NLTK (for sentence tokenization, stopwords, and word lemmatization)
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (TF-IDF Vectorization, Linear SVM Model)
* **Web Scraping & Parsing**: BeautifulSoup (bs4), Requests
* **Model Serialization**: Joblib

---

## 🔍 Verification Methodology

1. **Information Extraction**: The claim text is structured into semantic components: `subject`, `action`, `object`, `location`, `date`, `organizations`, `countries`, `numbers`, and `negated`.
2. **Entity Gating**: Before analyzing evidence, the verifier validates that the evidence source mentions the key entities. If a claim mentions specific organizations (e.g. "SpaceX"), any evidence snippet not mentioning the organization is skipped.
3. **Attribute Level Scoring**:
   * **Role Reversal**: Scans sentence structures to ensure that subject-object directions match the claim.
   * **Negation Scoping**: Ensures that negation terms are grammatically bound to the claim's action rather than unrelated topics.
   * **Date Compatibility**: Compares extracted years. Relative dates (e.g. "yesterday") are resolved dynamically against the article's publication date.
   * **Location Alignment**: Resolves locations using pre-defined city synonyms and clause-proximity scores.
4. **Source Weighting**: Outlets are weighted by credibility. Syndicated copies (exact duplicates) receive a reduced weight to prevent echo-chamber false positives.
5. **Aggregated Verdict**: Verifiably matching elements increment support scores. Factual conflicts set contradiction flags. The ratio of supporting to contradicting scores determines the final verdict.

> [!NOTE]
> **Understanding Confidence Scores**: The confidence rating displayed in the UI is an **evidence-confidence score** representing the density and alignment of matching factual attributes across scraped sources. It is **not** a guaranteed real-world mathematical accuracy percentage.

---

## 📊 Test Results

The verification engine and auxiliary ML classifier are fully validated using an automated unit and integration testing suite in `test_app.py`.

* **Total Tests**: 68
* **Passed**: 68 (100% success rate)
* **Failed / Errors**: 0
* **Execution Time**: ~15 seconds

---

## 🚀 Installation & Local Execution

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: NLTK resources (punkt, stopwords, wordnet, omw-1.4, etc.) are downloaded automatically by the application code on startup.*

### 4. Run the Dashboard
You can start the Streamlit server directly by executing:
```bash
python app.py
```
*(Running `python app.py` contains a boot hook that automatically redirects to the Streamlit runner).*

Alternatively, run:
```bash
streamlit run app.py
```
Open **`http://localhost:8501/`** in your web browser.

---

## 🌐 Streamlit Community Cloud Deployment

To deploy this project to the web:

1. **Push Code to GitHub**: Make sure the repository contains all application scripts, `requirements.txt`, `.gitignore`, `recent_claims.csv`, and the pre-trained `saved_model/` folder.
2. **Deploy on Streamlit**:
   * Go to [Streamlit Share](https://share.streamlit.io/).
   * Click **New app** and connect your GitHub repository.
   * Set **Main file path** to `app.py`.
   * Click **Deploy!**
3. **Secrets/Environment Variables**: This application accesses the Google News RSS search feed directly and requires no private API keys or configurations. Leave the secrets field blank.

---

## ⚠️ Limitations & Future Scope

### Limitations
* **Public Search Throttling**: Heavy querying of the Google News RSS feed can trigger temporary rate limits.
* **Anti-Bot Defenses**: Some publishers protect their pages with Cloudflare, Captchas, or paywalls, which prevents the scraper from extracting the full body text (the verifier falls back to RSS snippets in this case).
* **Rule-based Matching**: Factual alignment is check-based and does not leverage deep semantic context reasoning (e.g. synonyms not explicitly listed).

### Future Scope
* **Semantic Search & LLMs**: Incorporate private LLMs or zero-shot classifiers (e.g. via Hugging Face/Gemini API) to handle complex linguistic reformulations.
* **Claim-Extraction Models**: Replace standard sentence tokenization with an active Open Information Extraction (OpenIE) model.
* **Multilingual Support**: Translate foreign language evidence to check claims globally.
