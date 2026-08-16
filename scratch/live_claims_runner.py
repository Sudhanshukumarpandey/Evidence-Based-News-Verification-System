import sys
import os

# Set PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

import verifier

print("--- Testing Live RSS and Verification Pipeline ---")

def run_live_claim_test(claim):
    print(f"\nClaim: '{claim}'")
    try:
        struct = verifier.parse_claim_structure(claim)
        print(f"Parsed Structure: {struct}")
        
        queries = verifier.generate_queries_for_claim(struct)
        print(f"Generated Queries: {queries}")
        
        if not queries:
            print("No queries generated.")
            return
            
        # Test query news RSS
        diagnostics = []
        all_articles = []
        seen_links = set()
        
        for q in queries[:4]:
            fetched = verifier.query_google_news(q, limit=5, diagnostic_info=diagnostics)
            for art in fetched:
                if art["link"] not in seen_links:
                    seen_links.add(art["link"])
                    all_articles.append(art)
                    
        deduped = verifier.deduplicate_articles(all_articles)
        ranked = verifier.rank_articles_relevance(struct, deduped)
        
        res = verifier.verify_claim_against_evidence(struct, ranked, diagnostic_info=diagnostics)
        print(f"VERDICT: {res['verdict']}")
        print(f"CONFIDENCE: {res['confidence']:.2%}")
        print(f"REASON: {res.get('verdict_reason')}")
        print(f"EXPLANATION: {res['explanation']}")
        print("--- Diagnostics Log ---")
        for log_msg in diagnostics:
            print(f"  [DIAG] {log_msg}")
    except Exception as e:
        print(f"ERROR OCCURRED: {e}")

# 1. Test VERIFIED claim
run_live_claim_test("Microsoft opened its data center in Hyderabad in 2026.")

# 2. Test CONTRADICTED claim
run_live_claim_test("Microsoft did not open its data center in Hyderabad in 2026.")

# 3. Test UNVERIFIED claim
run_live_claim_test("Zylophonia Corporation announced a bubble gum factory on Pluto in 2099.")
