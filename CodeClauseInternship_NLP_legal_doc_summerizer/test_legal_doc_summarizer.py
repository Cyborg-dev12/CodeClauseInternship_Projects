import unittest
from advanced_legal_doc_summarizer import preprocess_text, clean_text, extract_clauses, summarize_text, extract_named_entities

class TestLegalDocSummarizer(unittest.TestCase):
    
    def setUp(self):
        self.sample_text = """
        This agreement shall be governed by the laws of the State of California. The parties agree to the termination of services upon breach of contract.
        Confidentiality shall be maintained by both parties for all proprietary information.
        """
        self.keywords = ["liability", "termination", "confidentiality", "dispute", "governing law"]
        self.regex_patterns = [r"\bGoverning Law\b", r"\bLiability\b", r"\bTermination\b"]

    def test_preprocess_text(self):
        sentences = preprocess_text(self.sample_text)
        self.assertEqual(len(sentences), 3)

    def test_clean_text(self):
        cleaned_text = clean_text("Hereby, the parties agree to the termination.")
        self.assertNotIn("hereby", cleaned_text.lower())  

    def test_extract_clauses(self):
        clauses = extract_clauses(self.sample_text, self.keywords, self.regex_patterns)
        self.assertGreater(len(clauses), 0)
        self.assertIn("termination", clauses[0].lower())

    def test_summarize_text(self):
        summary = summarize_text("The confidentiality clause requires both parties to maintain secrecy of shared information.")
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    def test_extract_named_entities(self):
        entities = extract_named_entities("This agreement shall be governed by the laws of the State of California.")
        self.assertIn(("California", "GPE"), entities)  


if __name__ == "__main__":
    unittest.main()
