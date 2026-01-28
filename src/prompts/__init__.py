"""Prompt templates for different extraction methods."""

from .zero_shot import ZeroShotPrompt
from .cot_prompt import ChainOfThoughtPrompt
from .rag_prompt import RAGPrompt

__all__ = ['ZeroShotPrompt', 'ChainOfThoughtPrompt', 'RAGPrompt']
