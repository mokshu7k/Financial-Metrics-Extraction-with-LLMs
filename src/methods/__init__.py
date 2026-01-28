"""Implementation of different extraction methods."""

from .zero_shot import ZeroShotExtractor
from .chain_of_thought import ChainOfThoughtExtractor
from .rag import RAGExtractor

__all__ = ['ZeroShotExtractor', 'ChainOfThoughtExtractor', 'RAGExtractor']
