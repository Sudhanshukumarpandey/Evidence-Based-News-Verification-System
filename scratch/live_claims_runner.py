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
        fetched = verifier.query_google_news(queries[0], limit=3)
        print(f"Fetched RSS Articles Count: {len(fetched)}")
        if fetched:
            print(f"Sample Article: {fetched[0]['title']} ({fetched[0]['source']})")
        else:
            print("No articles fetched for query.")
            
        deduped = verifier.deduplicate_articles(fetched)
        ranked = verifier.rank_articles_relevance(struct, deduped)
        
        res = verifier.verify_claim_against_evidence(struct, ranked)
        print(f"VERDICT: {res['verdict']}")
        print(f"CONFIDENCE: {res['confidence']:.2%}")
        print(f"REASON: {res.get('verdict_reason')}")
        print(f"EXPLANATION: {res['explanation']}")
    except Exception as e:
        print(f"ERROR OCCURRED: {e}")

# 1. Test a standard real live news claim (should be VERIFIED or MIXED depending on recent news)
run_live_claim_test("Microsoft opened a new data center in Hyderabad.")

# 2. Test a contradicted claim (wrong date for the event)
run_live_claim_test("Microsoft opened the Hyderabad data center in 2005.")

# 3. Test an unverified claim (random words with no news coverage)
run_live_claim_test("Zylophonia corporation announced a bubble gum factory on Pluto in 2099.")

# 4. Test an RSS failure fallback
print("\n--- Testing RSS Fallback / Network Error Graceful Handling ---")
original_url = "https://news.google.com/rss/search"
try:
    # Temporarily corrupt URL to simulate network failure
    import verifier
    # We can patch requests.get or corrupt the URL temporarily
    old_query = verifier.query_google_news
    def dummy_query(q, limit=5):
        raise RuntimeError("Simulated network timeout")
    
    verifier.query_google_news = dummy_query
    
    struct = verifier.parse_claim_structure("Microsoft opened a data center in Hyderabad.")
    res = verifier.verify_claim_against_evidence(struct, [])
    print(f"Fallback Verdict: {res['verdict']}")
    print(f"Fallback Explanation: {res['explanation']}")
finally:
    pass
