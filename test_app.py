import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import verifier
from verifier import (
    verify_local_claim, 
    compute_style_score, 
    predict_with_saved_model,
    extract_atomic_claims,
    parse_claim_structure,
    generate_queries_for_claim,
    verify_claim_against_evidence
)
from scraper import scrape_article

class NewsVerificationTestCase(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary claims CSV to test cosine-similarity match calculations
        self.temp_csv = "temp_test_claims.csv"
        df = pd.DataFrame([
            {"claim": "Liquid water exists on Mars", "verdict": "Real", "source": "NASA"},
            {"claim": "Celery juice cures all diseases", "verdict": "Fake", "source": "WHO"}
        ])
        df.to_csv(self.temp_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.temp_csv):
            try:
                os.remove(self.temp_csv)
            except OSError:
                pass

    def test_compute_style_score(self):
        # Verify clickbait markers (all-caps, exclamations, triggers) increment score
        title = "SHOCKING secrets about the economy!"
        text = "This is unbelievable!!! Pass this on now!!!"
        score, reasons = compute_style_score(title, text)
        self.assertGreater(score, 0)
        self.assertTrue(any("exclamation" in r.lower() for r in reasons))
        
        # Verify clean editorial styles score 0
        title_clean = "Federal Reserve keeps interest rates steady"
        text_clean = "The central bank decided to hold benchmark rates unchanged in their meeting."
        score_clean, reasons_clean = compute_style_score(title_clean, text_clean)
        self.assertEqual(score_clean, 0)
        self.assertEqual(len(reasons_clean), 0)

    def test_verify_local_claim_exact_match(self):
        res = verify_local_claim("Liquid water exists on Mars", csv_path=self.temp_csv)
        self.assertIsNotNone(res)
        self.assertEqual(res["verdict"], "Real")
        self.assertEqual(res["source"], "NASA")
        self.assertGreaterEqual(res["score"], 0.65)

    def test_verify_local_claim_no_match(self):
        res = verify_local_claim("Unrelated cooking recipe about pasta and cheese", csv_path=self.temp_csv)
        self.assertIsNotNone(res)
        self.assertLess(res["score"], 0.65)

    @patch("requests.get")
    def test_scrape_article_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body><h1>Scraped Article Title</h1><p>Body paragraph one.</p><p>Body paragraph two.</p></body></html>"
        mock_get.return_value = mock_response
        
        title, text = scrape_article("https://example.com/article")
        self.assertEqual(title, "Scraped Article Title")
        self.assertIn("Body paragraph one.", text)
        self.assertIn("Body paragraph two.", text)

    @patch("os.path.exists")
    def test_predict_with_saved_model_empty_input(self, mock_exists):
        mock_exists.return_value = True
        import verifier
        verifier._MODEL_CACHE.clear()
        
        mock_vec = MagicMock()
        mock_model = MagicMock()
        
        with patch("joblib.load", side_effect=[mock_model, mock_vec]):
            res = predict_with_saved_model("")
            self.assertEqual(res["status"], "empty")

    @patch("os.path.exists")
    @patch("joblib.load")
    def test_predict_with_saved_model_success(self, mock_load, mock_exists):
        mock_exists.return_value = True
        import verifier
        verifier._MODEL_CACHE.clear()
        
        mock_vec = MagicMock()
        mock_vec.transform.return_value = "mock_vectorized"
        
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.1, 0.9]]
        mock_model.predict.return_value = [1]
        mock_model.decision_function.return_value = [1.5]
        
        mock_load.side_effect = [mock_model, mock_vec]
        
        res = predict_with_saved_model("some real text")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["label"], "Real")
        self.assertGreater(res["confidence"], 0.5)

    def test_extract_atomic_claims(self):
        text = "BREAKING!!! India launched a military operation. The facility is India's largest nuclear power plant."
        # The facility should be resolved to the subject of the previous sentence (India)
        claims = extract_atomic_claims(text)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0], "India launched a military operation.")
        self.assertIn("India", claims[1])

    def test_parse_claim_structure(self):
        claim = "India launched a military attack on Pakistan yesterday."
        struct = parse_claim_structure(claim)
        self.assertEqual(struct["subject"], "India")
        self.assertEqual(struct["action"], "launched")
        self.assertEqual(struct["object"], "military attack")
        self.assertFalse(struct["negated"])
        self.assertEqual(struct["date"], "yesterday")
        self.assertIn("India", struct["countries"])
        self.assertIn("Pakistan", struct["countries"])

    def test_generate_queries_for_claim(self):
        struct = {
            "text": "India launched an attack on Pakistan.",
            "subject": "India",
            "action": "launched",
            "object": "attack",
            "entities": ["India", "Pakistan"],
            "negated": False,
            "date": None,
            "organizations": []
        }
        queries = generate_queries_for_claim(struct)
        self.assertGreaterEqual(len(queries), 1)
        self.assertIn("India launched attack", queries)

    @patch("requests.get")
    def test_verify_claim_against_evidence_supported(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>India launched a military attack on Pakistan today to secure the borders.</p></body></html>"
        mock_get.return_value = mock_response
        
        struct = {
            "text": "India launched a military attack on Pakistan.",
            "subject": "India",
            "action": "launched",
            "object": "military attack",
            "entities": ["India", "Pakistan"],
            "negated": False,
            "date": None,
            "location": None,
            "organizations": [],
            "numbers": [],
            "countries": ["India", "Pakistan"]
        }
        
        mock_articles = [{
            "title": "India launched attack on Pakistan",
            "link": "https://example.com/news1",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Reports confirm India launched attack on Pakistan."
        }]
        
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")
        self.assertGreater(res["confidence"], 0.5)

    @patch("requests.get")
    def test_verify_claim_against_evidence_contradicted(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>India denied launching any military attack on Pakistan.</p></body></html>"
        mock_get.return_value = mock_response
        
        struct = {
            "text": "India launched an attack on Pakistan.",
            "subject": "India",
            "action": "launched",
            "object": "attack",
            "entities": ["India", "Pakistan"],
            "negated": False,
            "date": None,
            "location": None,
            "organizations": [],
            "numbers": [],
            "countries": ["India", "Pakistan"]
        }
        
        mock_articles = [{
            "title": "India denies launching attack on Pakistan",
            "link": "https://example.com/news1",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "India denied reports of attack."
        }]
        
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")
        self.assertGreater(res["confidence"], 0.5)

    @patch("requests.get")
    def test_accuracy_1_support(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its largest India data center hub in Hyderabad on August 6, 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its largest India data center hub in Hyderabad on August 6, 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened data center in Hyderabad",
            "link": "https://example.com/1",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened its largest India data center hub in Hyderabad on August 6, 2026."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_accuracy_2_location_contradiction(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its largest India data center hub in Hyderabad on August 6, 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its largest India data center hub in Mumbai on August 6, 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft launched data center in Hyderabad",
            "link": "https://example.com/2",
            "source": "Economic Times",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft data center opened in Hyderabad."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_accuracy_3_date_contradiction(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its largest India data center hub in Hyderabad in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its largest India data center hub in Hyderabad in 2020."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft launched data center in Hyderabad in 2026",
            "link": "https://example.com/3",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft data center opened in 2026."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_accuracy_4_number_mixed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its Hyderabad data center as part of its AI infrastructure expansion.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its Hyderabad data center as part of a $50 billion investment."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened Hyderabad data center",
            "link": "https://example.com/4",
            "source": "Economic Times",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened its Hyderabad data center as part of AI expansion."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "MIXED")

    @patch("requests.get")
    def test_accuracy_5_negation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India says it is not committed to importing ethanol from the United States.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "India has agreed to import ethanol from the United States."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "India denies importing ethanol from US",
            "link": "https://example.com/5",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "India says it is not committed to importing ethanol from the United States."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_accuracy_6_unrelated_negation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its Hyderabad data center. Pakistan denied involvement in attacks against India.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened Hyderabad data center",
            "link": "https://example.com/6",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened its Hyderabad data center."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_accuracy_7_role_reversal(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Pakistan attacked India after tensions escalated.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "India attacked Pakistan."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Pakistan attacked India",
            "link": "https://example.com/7",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Pakistan attacked India across the border."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_accuracy_8_multiple_years(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced plans in 2020 and opened the facility in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the facility in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened facility in 2026",
            "link": "https://example.com/8",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft announced plans in 2020 and opened the facility in 2026."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_accuracy_9_multiple_locations(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft has offices in Mumbai and Bengaluru. Its new data center region opened in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened data center in Hyderabad",
            "link": "https://example.com/9",
            "source": "Economic Times",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft has offices in Mumbai and Bengaluru. Its new data center region opened in Hyderabad."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_accuracy_10_unrelated_article(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India's largest nuclear power plant is located elsewhere.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened a data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Nuclear power plant details",
            "link": "https://example.com/10",
            "source": "Generic News",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "India's largest nuclear power plant is located elsewhere."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "UNVERIFIED")

    @patch("requests.get")
    def test_accuracy_11_multiple_supporting_sources(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Hyderabad data center in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the Hyderabad data center in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [
            {"title": "Microsoft opened Hyderabad data center in 2026", "link": "https://example.com/11a", "source": "Reuters", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Microsoft opened the Hyderabad data center in 2026."},
            {"title": "Microsoft launches datacenter region in Hyderabad 2026", "link": "https://example.com/11b", "source": "BBC", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Microsoft launches datacenter region in Hyderabad 2026."},
            {"title": "Hyderabad cloud data center opened by Microsoft in 2026", "link": "https://example.com/11c", "source": "Economic Times", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Hyderabad cloud data center opened by Microsoft in 2026."},
            {"title": "Unrelated news headline", "link": "https://example.com/11d", "source": "Blog", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Pakistan denied involvement in regional incident."}
        ]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_accuracy_12_false_historical_claim(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Hyderabad data center in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the Hyderabad data center in 2020."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened Hyderabad data center in 2026",
            "link": "https://example.com/12",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened the Hyderabad data center in 2026."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_new_A_unlisted_city(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its new data center in Bengaluru.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened its data center in Bengaluru."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened data center in Bengaluru",
            "link": "https://example.com/newA",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened its new data center in Bengaluru."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_B_organization(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>ISRO successfully launched the satellite into orbit.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "ISRO launched the satellite."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "ISRO launched satellite",
            "link": "https://example.com/newB",
            "source": "The Hindu",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "ISRO successfully launched the satellite into orbit."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_C_wrong_organization(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>NASA launched the satellite from Cape Canaveral.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "ISRO launched the satellite."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "NASA launched satellite",
            "link": "https://example.com/newC",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "NASA launched the satellite from Cape Canaveral."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "UNVERIFIED")

    @patch("requests.get")
    def test_new_D_announcement_not_completion(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced plans to open the data center in Hyderabad next year.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft announced plans for data center",
            "link": "https://example.com/newD",
            "source": "Economic Times",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft announced plans to open the data center in Hyderabad."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_E_future_not_completion(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft will open the data center next year in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft will open data center next year",
            "link": "https://example.com/newE",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft will open the data center next year."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_F_event_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced plans in 2020 and opened the facility in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the facility in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened facility in 2026",
            "link": "https://example.com/newF",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft announced plans in 2020 and opened the facility in 2026."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_G_event_location(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft has offices in Mumbai and Bengaluru. Its new data center opened in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened data center in Hyderabad",
            "link": "https://example.com/newG",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft has offices in Mumbai and Bengaluru. Its new data center opened in Hyderabad."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_H_wrong_event_location(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft has offices in Mumbai, but its new data center opened in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the data center in Mumbai."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft data center opened in Hyderabad",
            "link": "https://example.com/newH",
            "source": "Economic Times",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft has offices in Mumbai, but its new data center opened in Hyderabad."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_new_I_number_context(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its Hyderabad data center. The company has discussed $50 billion in investments globally.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft invested $50 billion in the Hyderabad data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Microsoft opened Hyderabad data center",
            "link": "https://example.com/newI",
            "source": "Reuters",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Microsoft opened its Hyderabad data center. The company has discussed $50 billion in investments globally."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_J_syndication(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Hyderabad data center.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "Microsoft opened the Hyderabad data center."
        struct = parse_claim_structure(claim)
        mock_articles = [
            {"title": "Microsoft opened Hyderabad data center", "link": "https://example.com/j1", "source": "Reuters", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Microsoft opened the Hyderabad data center."},
            {"title": "Microsoft opened Hyderabad data center, Reuters reports", "link": "https://example.com/j2", "source": "Blog A", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Microsoft opened the Hyderabad data center, according to Reuters."},
            {"title": "Microsoft opened Hyderabad data center", "link": "https://example.com/j3", "source": "Blog B", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "Microsoft opened the Hyderabad data center."}
        ]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_new_K_low_quality_vs_high_quality(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India says it is not committed to importing ethanol from the US.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "India agreed to import ethanol from the United States."
        struct = parse_claim_structure(claim)
        mock_articles = [
            {"title": "India agrees to import ethanol", "link": "https://example.com/k1", "source": "Blog 1", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "India agrees to import ethanol from US."},
            {"title": "India agrees to import ethanol", "link": "https://example.com/k2", "source": "Blog 2", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "India agrees to import ethanol from US."},
            {"title": "India agrees to import ethanol", "link": "https://example.com/k3", "source": "Blog 3", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "India agrees to import ethanol from US."},
            {"title": "India says it is not committed to import ethanol", "link": "https://example.com/k4", "source": "Reuters", "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT", "description": "India says it is not committed to import ethanol from the United States."}
        ]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_new_L_reverse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Pakistan attacked India after tensions rose.</p></body></html>"
        mock_get.return_value = mock_resp
        
        claim = "India attacked Pakistan."
        struct = parse_claim_structure(claim)
        mock_articles = [{
            "title": "Pakistan attacked India",
            "link": "https://example.com/l",
            "source": "BBC",
            "pub_date": "Sun, 09 Aug 2026 10:00:00 GMT",
            "description": "Pakistan attacked India."
        }]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P1_number_unit_mismatch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the data center and created 50 jobs.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft invested $50 billion in the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened data center creating 50 jobs", "link": "https://p1", "source": "Reuters", "description": "Created 50 jobs"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P2_number_scale_mismatch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft invested $50 million in the data center.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft invested $50 billion in the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft invested $50 million", "link": "https://p2", "source": "Reuters", "description": "Invested $50 million"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P3_percentage_vs_currency(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft data center uses 50% renewable energy.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft invested $50 billion in the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Data center uses 50% renewable energy", "link": "https://p3", "source": "Reuters", "description": "50% renewable energy"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P4_number_synonym(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft invested 50 billion dollars in the data center.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft invested $50 billion in the data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft invested 50 billion dollars", "link": "https://p4", "source": "Reuters", "description": "Invested 50 billion dollars"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P5_construction_vs_opened(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft is currently constructing the data center in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft constructing data center", "link": "https://p5", "source": "Reuters", "description": "Currently constructing"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "UNVERIFIED")

    @patch("requests.get")
    def test_P6_cancelled_event(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft cancelled plans for the data center in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft cancelled data center plans", "link": "https://p6", "source": "Reuters", "description": "Cancelled plans"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P7_exact_entity(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>North Korea launched the satellite into orbit.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "South Korea launched the satellite."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "North Korea launched satellite", "link": "https://p7", "source": "Reuters", "description": "North Korea launched satellite"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P8_apple_entity(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Apple Corps acquired the music startup.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Apple Inc acquired the startup."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Apple Corps acquired startup", "link": "https://p8", "source": "Reuters", "description": "Apple Corps acquired startup"}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P9_attributed_claim(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>A competitor claims Microsoft opened an illegal data center.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened a data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Competitor claims Microsoft opened data center", "link": "https://p9", "source": "Blog X", "description": "Competitor claims..."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "UNVERIFIED")

    @patch("requests.get")
    def test_P10_direct_plus_attributed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the data center in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the data center in Hyderabad."
        struct = parse_claim_structure(claim)
        mock_articles = [
            {"title": "Microsoft opened data center in Hyderabad", "link": "https://p10a", "source": "Reuters", "description": "Microsoft opened the data center in Hyderabad."},
            {"title": "Competitor claims Microsoft opened data center", "link": "https://p10b", "source": "Blog Y", "description": "Competitor claims Microsoft opened data center."}
        ]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P11_double_negation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India did not reject the ethanol proposal.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "India rejected the proposal."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "India did not reject proposal", "link": "https://p11", "source": "Reuters", "description": "India did not reject proposal."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P12_negation_of_negation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India did not deny the proposal.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "India rejected the proposal."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "India did not deny proposal", "link": "https://p12", "source": "Reuters", "description": "India did not deny proposal."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P13_multi_event(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened a center in Mumbai in 2020. Microsoft opened another center in Hyderabad in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the center in Hyderabad in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened center in Hyderabad in 2026", "link": "https://p13", "source": "Reuters", "description": "Opened in Hyderabad in 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P14_multi_event_wrong_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its new center in Hyderabad in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the Hyderabad center in 2020."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened Hyderabad center in 2026", "link": "https://p14", "source": "Reuters", "description": "Opened in 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P15_multi_event_wrong_location(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened its 2026 data center in Hyderabad.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the center in Mumbai in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened 2026 center in Hyderabad", "link": "https://p15", "source": "Reuters", "description": "Opened in Hyderabad in 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P16_action_noun(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft's opening of the Hyderabad data center was completed in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the Hyderabad data center in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Opening of Hyderabad data center completed", "link": "https://p16", "source": "Reuters", "description": "Opening completed."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P17_relative_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the center on August 12, 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the center yesterday."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened center on August 12, 2026", "link": "https://p17", "source": "Reuters", "pub_date": "Thu, 13 Aug 2026 10:00:00 GMT", "description": "Opened on August 12, 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P18_publication_date_is_not_event_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the center in 2025.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the center yesterday."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened center in 2025", "link": "https://p18", "source": "Reuters", "pub_date": "Thu, 13 Aug 2026 10:00:00 GMT", "description": "Opened in 2025."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_P19_unrelated_number(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the center and created 50 jobs.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft invested $50 billion in the center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened center creating 50 jobs", "link": "https://p19", "source": "Reuters", "description": "Created 50 jobs."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertNotEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_P20_unrelated_negation_safe(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Hyderabad data center. Pakistan denied involvement in attacks against India.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the Hyderabad data center."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened Hyderabad data center", "link": "https://p20", "source": "Reuters", "description": "Opened Hyderabad data center. Pakistan denied involvement..."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    # PHASE 7 REGRESSION TESTS (DATE-1..5 & NEG-1..8)
    @patch("requests.get")
    def test_DATE1_historical_year_precedence(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced the project in 2020 and opened the facility in Hyderabad in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft announced in 2020 and opened facility in 2026", "link": "https://date1", "source": "Reuters", "description": "Announced in 2020 and opened in 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_DATE2_separate_sentence_historical_year(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced plans in 2020. The facility opened in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Announced in 2020. Facility opened in 2026", "link": "https://date2", "source": "Reuters", "description": "Announced 2020. Opened 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_DATE3_year_before_verb(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>On August 6, 2026, Microsoft opened the facility.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "On August 6, 2026, Microsoft opened facility", "link": "https://date3", "source": "Reuters", "description": "Opened on Aug 6, 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_DATE4_multi_event_location_year(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Mumbai office in 2020 and the Hyderabad data center in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the Hyderabad data center in 2026."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened Mumbai office in 2020 and Hyderabad data center in 2026", "link": "https://date4", "source": "Reuters", "description": "Mumbai office 2020, Hyderabad data center 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_DATE5_historical_year_conflict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft announced the project in 2020 and opened the facility in 2026.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility in 2020."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft announced in 2020 and opened in 2026", "link": "https://date5", "source": "Reuters", "description": "Opened in 2026."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_NEG1_direct_not_open(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft did not open the facility.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft did not open facility", "link": "https://neg1", "source": "Reuters", "description": "Did not open facility."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_NEG2_direct_denied_opening(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft denied opening the facility.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft denied opening facility", "link": "https://neg2", "source": "Reuters", "description": "Denied opening."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_NEG3_unrelated_denial_same_sentence(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the facility, while denying unrelated allegations concerning another project.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened the facility while denying unrelated allegations", "link": "https://neg3", "source": "Reuters", "description": "Opened the facility."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_NEG4_unrelated_denial_separate_sentence(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the facility. The company denied allegations about a separate project.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened the facility. Company denied separate allegations", "link": "https://neg4", "source": "Reuters", "description": "Opened the facility."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_NEG5_denied_agreed_import(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India denied that it had agreed to import ethanol.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "India agreed to import ethanol."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "India denied it agreed to import ethanol", "link": "https://neg5", "source": "Reuters", "description": "India denied agreement."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_NEG6_unrelated_party_denial(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft opened the Hyderabad facility. Pakistan denied involvement in a separate incident.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the Hyderabad facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft opened Hyderabad facility. Pakistan denied involvement", "link": "https://neg6", "source": "Reuters", "description": "Opened Hyderabad facility."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "VERIFIED")

    @patch("requests.get")
    def test_NEG7_did_not_reject(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>India did not reject the proposal.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "India rejected the proposal."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "India did not reject proposal", "link": "https://neg7", "source": "Reuters", "description": "Did not reject proposal."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "CONTRADICTED")

    @patch("requests.get")
    def test_NEG8_did_not_deny_opening(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body><p>Microsoft did not deny opening the facility.</p></body></html>"
        mock_get.return_value = mock_resp
        claim = "Microsoft opened the facility."
        struct = parse_claim_structure(claim)
        mock_articles = [{"title": "Microsoft did not deny opening facility", "link": "https://neg8", "source": "Reuters", "description": "Did not deny opening."}]
        res = verify_claim_against_evidence(struct, mock_articles)
        self.assertEqual(res["verdict"], "UNVERIFIED")

if __name__ == "__main__":
    unittest.main()
