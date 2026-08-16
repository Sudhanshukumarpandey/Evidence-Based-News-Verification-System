import streamlit as st
import re
import importlib

# Force reloads of helper modules on run to clear bytecode cache issues
import verifier
importlib.reload(verifier)

from scraper import scrape_article
from streamlit.runtime import exists as st_exists

def main():
    st.set_page_config(page_title="Evidence-Based news verification System", page_icon="🔍", layout="wide")

    st.title("🔍 Evidence-Based News Verification System")
    st.markdown("Verify the factual credibility of claims, paragraphs, or article excerpts in real-time against live online news sources.")

    st.markdown("### Paste Paragraph or Editorial Text")
    pasted_text = st.text_area("Article text:", placeholder="Paste your paragraph or claims here...", height=200, key="text_input")
    
    if st.button("Verify Claim(s)", type="primary"):
        if not pasted_text.strip():
            st.warning("Please enter some text to verify.")
        else:
            with st.spinner("Analyzing claims and searching for live evidence..."):
                # 1. Extract atomic claims
                atomic_claims = verifier.extract_atomic_claims(pasted_text)
                
                if not atomic_claims:
                    st.warning("No valid sentences could be identified as claims.")
                    return
                    
                st.markdown(f"Detected **{len(atomic_claims)}** atomic claim(s) to verify:")
                
                claim_results = []
                
                # 2. Verify each atomic claim
                for i, claim_text in enumerate(atomic_claims):
                    struct = verifier.parse_claim_structure(claim_text)
                    
                    # Generate search queries and fetch news articles
                    queries = verifier.generate_queries_for_claim(struct)
                    
                    diagnostics = []
                    diagnostics.append(f"🔍 Analyzing Claim: '{claim_text}'")
                    diagnostics.append(f"Parsed claim structure: {struct}")
                    diagnostics.append(f"Generated search queries: {queries}")
                    
                    all_articles = []
                    seen_links = set()
                    
                    for q in queries[:4]:
                        fetched = verifier.query_wikipedia(q, limit=5, diagnostic_info=diagnostics)
                        for art in fetched:
                            if art["link"] not in seen_links:
                                seen_links.add(art["link"])
                                all_articles.append(art)
                                
                    diagnostics.append(f"Retrieved {len(all_articles)} unique articles from Wikipedia search queries.")
                    
                    # Deduplicate and rank relevance
                    deduped = verifier.deduplicate_articles(all_articles)
                    ranked = verifier.rank_articles_relevance(struct, deduped)
                    
                    diagnostics.append(f"After deduplication and relevance ranking, selected top {len(ranked)} articles.")
                    
                    # Verify claim against evidence
                    verdict_data = verifier.verify_claim_against_evidence(struct, ranked, diagnostic_info=diagnostics)
                    claim_results.append({
                        "claim": claim_text,
                        "structure": struct,
                        "verdict_data": verdict_data
                    })
                    
                # 3. Aggregate Paragraph-Level Verdict
                verdicts = [res["verdict_data"]["verdict"] for res in claim_results]
                
                st.markdown("---")
                st.markdown("## 🏆 Overall Verification Verdict")
                
                if "CONTRADICTED" in verdicts:
                    st.error("🔴 **Verdict: CONTRADICTED**")
                    st.info("System has found reliable news coverage that contradicts one or more claims in the text.")
                elif all(v == "VERIFIED" for v in verdicts):
                    st.success("🟢 **Verdict: VERIFIED / SUPPORTED**")
                    st.info("All claims in your text are supported by reliable live news coverage.")
                elif "VERIFIED" in verdicts or "MIXED" in verdicts:
                    st.warning("🟠 **Verdict: MIXED / PARTIALLY VERIFIED**")
                    st.info("Some claims are verified, but others remain unverified or have unsupported details.")
                else:
                    st.warning("🟡 **Verdict: UNVERIFIED**")
                    st.info("There is insufficient reliable online news coverage to verify or contradict the claims.")
                    
                st.markdown("---")
                
                # 4. Display Individual Claims Detailed Analysis
                st.markdown("### 🔍 Sentence-Level Evidence Analysis")
                for i, res in enumerate(claim_results):
                    c_text = res["claim"]
                    v_data = res["verdict_data"]
                    verdict = v_data["verdict"]
                    
                    # Render claim card container
                    with st.container():
                        st.markdown(f"#### Claim {i+1}: *\"{c_text}\"*")
                        
                        # Render Badge
                        # Render Badge
                        conf_label = f"Evidence Confidence: {v_data['confidence']:.0%}"
                        if verdict == "VERIFIED":
                            st.success(f"🟢 **VERIFIED / SUPPORTED** ({conf_label})")
                        elif verdict == "CONTRADICTED":
                            st.error(f"🔴 **CONTRADICTED** ({conf_label})")
                        elif verdict == "MIXED":
                            st.warning(f"🟠 **MIXED / PARTIALLY VERIFIED** ({conf_label})")
                        else:
                            st.warning("🟡 **UNVERIFIED** (Insufficient Evidence)")
                            
                        # Show exact reason and detailed explanation
                        if "verdict_reason" in v_data:
                            st.write(f"**Reason**: {v_data['verdict_reason']}")
                        st.write(f"**Explanation**: {v_data['explanation']}")
                        
                        # Render supports/mixed/unverified highlights
                        s_info = res["structure"]
                        if verdict == "VERIFIED":
                            st.markdown("**Evidence supports:**")
                            if s_info["subject"]: st.markdown(f"- **Subject**: {s_info['subject']}")
                            if s_info["action"]: st.markdown(f"- **Action**: {s_info['action']}")
                            if s_info["location"]: st.markdown(f"- **Location**: {s_info['location']}")
                            if s_info["date"]: st.markdown(f"- **Event Date**: {s_info['date']}")
                        elif verdict == "MIXED":
                            st.markdown("**Factual Breakdown:**")
                            supported_core = []
                            if s_info["subject"]: supported_core.append(s_info["subject"])
                            if s_info["action"]: supported_core.append(s_info["action"])
                            if s_info["location"]: supported_core.append(f"in {s_info['location']}")
                            if s_info["date"]: supported_core.append(f"on {s_info['date']}")
                            
                            st.markdown(f"- **Supported Core Event**: {' '.join(supported_core)}")
                            st.markdown(f"- **Not Independently Verified**: {v_data.get('attribute_breakdown', {}).get('numbers', {}).get('claim', 'Numerical figures')}")
                        elif verdict == "UNVERIFIED":
                            st.markdown("**Verification Status:**")
                            st.markdown(f"- {v_data.get('explanation', 'Insufficient matching evidence.')}")
                            
                        # Show Attribute Matching Breakdown Table
                        if "attribute_breakdown" in v_data and v_data["attribute_breakdown"]:
                            with st.expander("📊 View Factual Attribute Match Breakdown"):
                                table_rows = []
                                for attr_key, attr_info in v_data["attribute_breakdown"].items():
                                    status = attr_info["status"]
                                    status_icon = "🟢 MATCH"
                                    if status == "CONFLICT":
                                        status_icon = "🔴 CONTRADICTION"
                                    elif status == "UNSUPPORTED":
                                        status_icon = "🟠 UNSUPPORTED"
                                    elif status == "NOT_APPLICABLE":
                                        status_icon = "⚪ NOT APPLICABLE"
                                        
                                    table_rows.append(f"| **{attr_info['name']}** | {attr_info['claim']} | {attr_info['evidence']} | {status_icon} |")
                                    
                                if table_rows:
                                    st.markdown("| Attribute | Claimed Attribute | Evidence Attribute | Status |\n| :--- | :--- | :--- | :--- |\n" + "\n".join(table_rows))
                        
                        # Show parsed NLP structure for clarity
                        with st.expander("🛠️ View Parsed Factual NLP Attributes"):
                            st.json({
                                "Subject": s_info["subject"],
                                "Action": s_info["action"],
                                "Object": s_info["object"],
                                "Location": s_info["location"],
                                "Date": s_info["date"],
                                "Organizations": s_info["organizations"],
                                "Countries": s_info["countries"],
                                "Numbers": s_info["numbers"],
                                "Negated": s_info["negated"],
                                "Named Entities": s_info["entities"]
                            })
                            
                        # Render supporting/contradicting sources
                        if v_data["sources"]:
                            st.write("**Identified News Sources:**")
                            for art in v_data["sources"][:5]:
                                src_type = art.get("type", "Neutral")
                                type_badge = "⚪ Neutral"
                                if src_type == "Supporting":
                                    type_badge = "🟢 Supporting"
                                elif src_type == "Contradicting":
                                    type_badge = "🔴 Contradicting"
                                    
                                st.markdown(f"- **[{art['source']}]({art['link']})**: {art['title']} *(Published: {art['pub_date']})* — **{type_badge}**")
                        else:
                            st.write("No matching news sources found for this statement.")
                            
                        if "diagnostics" in v_data and v_data["diagnostics"]:
                            with st.expander("⚙️ View Deployment Diagnostic Info"):
                                for d_msg in v_data["diagnostics"]:
                                    st.write(d_msg)
                                    
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                st.markdown("---")
                
                # 5. Optional Offline Writing Style Check (Auxiliary Only)
                st.markdown("### 🤖 Style & Clickbait Analysis (Auxiliary Signal)")
                ml_result = verifier.predict_with_saved_model(pasted_text)
                if ml_result and ml_result["status"] == "success":
                    st.markdown("The offline ML model analyzed the writing style characteristics of the entire text block:")
                    if ml_result["label"] == "Real":
                        st.success(f"🟢 **Writing Style: Professional / Neutral** (Confidence: {ml_result['confidence']:.1%})")
                    else:
                        st.error(f"🔴 **Writing Style: Sensational / Clickbait Risk** (Confidence: {ml_result['confidence']:.1%})")
                else:
                    st.info("Style prediction not available.")

if __name__ == '__main__':
    if st_exists():
        main()
    else:
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
