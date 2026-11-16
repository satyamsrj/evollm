"""
Unit tests for the EvoLLM paper generation and citation grounding engine.
"""
import os
import unittest
from evollm.paper.citations import CitationRegistry
from evollm.paper.generator import PaperGenerator


class TestPaperGenerator(unittest.TestCase):
    def setUp(self):
        self.registry = CitationRegistry()
        self.generator = PaperGenerator(output_dir="paper")

    def test_citation_registry_contains_foundational_papers(self):
        """Verifies core foundational papers exist with DOIs/arXiv IDs."""
        foundational_keys = [
            "hu2021lora", "wang2023olora", "slora2024subspace",
            "revive2026spectral", "meng2024simpo", "ricl2026feedback",
            "linc2021dialogue", "liska2022streamingqa", "vu2024freshqa",
            "lin2022truthfulqa", "hendrycks2021mmlu", "cobbe2021gsm8k",
            "zhao2024wildchat"
        ]
        for key in foundational_keys:
            self.assertTrue(self.registry.contains(key), f"Missing key {key}")
            entry = self.registry.get(key)
            self.assertIsNotNone(entry)
            self.assertTrue(entry.doi_or_arxiv, f"Missing DOI/arXiv for {key}")
            self.assertTrue(entry.bibtex.startswith("@"), f"Invalid BibTeX for {key}")

    def test_citation_verification_detects_missing_keys(self):
        """Verifies anti-hallucination detection flags fake or unverified citations."""
        sample_text = r"Previous work \cite{hu2021lora, fake_citation_2025_nonexistent} explored this."
        res = self.registry.verify_citations_in_text(sample_text)
        self.assertFalse(res["is_fully_grounded"])
        self.assertIn("fake_citation_2025_nonexistent", res["missing_keys"])
        self.assertEqual(res["valid_count"], 1)

    def test_paper_generation_and_grounding(self):
        """Verifies complete paper generation outputs valid files and 100% grounded citations."""
        res = self.generator.compile_all()
        
        self.assertTrue(os.path.exists(res["bibtex_file"]))
        self.assertTrue(os.path.exists(res["latex_file"]))
        self.assertTrue(os.path.exists(res["markdown_file"]))
        
        # Verify 100% grounded citations in generated LaTeX
        verification = res["verification"]
        self.assertTrue(
            verification["is_fully_grounded"],
            f"Unverified citations found in paper: {verification['missing_keys']}"
        )
        self.assertGreater(verification["valid_count"], 5)


if __name__ == "__main__":
    unittest.main()
