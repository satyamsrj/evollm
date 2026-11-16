"""
Paper generation and citation management module for EvoLLM.
"""
from evollm.paper.citations import CitationRegistry, CitationEntry
from evollm.paper.generator import PaperGenerator

__all__ = ["CitationRegistry", "CitationEntry", "PaperGenerator"]
