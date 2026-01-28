"""RAG prompt templates that incorporate retrieved context."""

from typing import Dict, List, Any

class RAGPrompt:
    """
    RAG-based prompts that use retrieved document chunks 
    for context-aware extraction.
    """
    
    @staticmethod
    def create_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        query: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Create RAG prompt with retrieved context chunks.
        
        Args:
            company: Company name
            quarter: Quarter
            year: Year
            query: User query/task
            retrieved_contexts: List of retrieved chunks with metadata
        
        Returns:
            Formatted RAG prompt
        """
        
        # Format retrieved contexts
        contexts_text = ""
        for i, context in enumerate(retrieved_contexts, 1):
            contexts_text += f"""
Context {i} (Relevance Score: {context.get('score', 0):.3f}):
Source: {context.get('document_type', 'Unknown')} - {context.get('section_type', 'General')}
Content:
{context.get('content', '')}

---
"""
        
        prompt = f"""You are a financial analyst extracting information from financial documents.

**Task:** {query}

**Company:** {company}
**Quarter:** {quarter}
**Year:** {year}

**Retrieved Relevant Contexts:**
{contexts_text}

**Instructions:**
1. Use ONLY the information provided in the contexts above
2. Extract the requested financial metrics accurately
3. If a metric is not found in any context, mark it as "Not Available"
4. Cite which context number you used for each metric
5. Maintain the original units and formatting

**Expected Metrics to Extract:**
- Revenue from operations
- Total Income
- Net profit for the period
- Basic Earnings per share
- Total assets
- Total liabilities
- Interim Equity Dividend
- Gross Margin
- Operating Margin
- Foods Business Revenue
- Premium Personal Care contribution to domestic revenue

**Output Format (JSON):**
{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "extraction_metadata": {{
        "num_contexts_used": <number>,
        "primary_source": "<context number>",
        "confidence": "<high/medium/low>"
    }},
    "metrics": {{
        "revenue_from_operations": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "total_income": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "net_profit": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "basic_eps": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "total_assets": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "total_liabilities": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "interim_equity_dividend": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "gross_margin": {{
            "value": <number>,
            "unit": "%",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "operating_margin": {{
            "value": <number>,
            "unit": "%",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "foods_business_revenue": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }},
        "premium_personal_care_contribution": {{
            "value": <number>,
            "unit": "<unit>",
            "source_context": <context_number>,
            "confidence": "<high/medium/low>"
        }}
    }}
}}

Extract the metrics now:"""
        
        return prompt
    
    @staticmethod
    def create_commentary_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        query: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Create RAG prompt for extracting commentary from transcripts.
        """
        
        # Format contexts
        contexts_text = ""
        for i, context in enumerate(retrieved_contexts, 1):
            section_type = context.get('section_type', 'Unknown')
            contexts_text += f"""
Context {i} (Section: {section_type}, Score: {context.get('score', 0):.3f}):
{context.get('content', '')}

---
"""
        
        prompt = f"""You are analyzing earnings call transcript excerpts.

**Task:** {query}

**Company:** {company}
**Quarter:** {quarter}
**Year:** {year}

**Retrieved Relevant Contexts:**
{contexts_text}

**Instructions:**
1. Analyze the provided transcript excerpts
2. Extract key management commentary on the requested topics
3. Cite which context contains each piece of information
4. Only use information explicitly stated in the contexts
5. Maintain factual accuracy - do not infer or speculate

**Topics to Extract:**
1. Revenue Performance and Drivers
2. Profitability and Margin Trends
3. Business Outlook and Guidance
4. Challenges and Headwinds
5. Strategic Initiatives
6. Market Conditions

**Output Format (JSON):**
{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "extraction_metadata": {{
        "num_contexts_analyzed": <number>,
        "section_types_covered": ["<type1>", "<type2>"],
        "confidence": "<high/medium/low>"
    }},
    "commentary": {{
        "revenue_performance": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }},
        "profitability": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }},
        "business_outlook": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }},
        "key_challenges": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }},
        "strategic_initiatives": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }},
        "market_conditions": {{
            "summary": "<concise summary>",
            "key_points": ["<point 1>", "<point 2>"],
            "source_contexts": [<context_numbers>]
        }}
    }}
}}

Extract the commentary now:"""
        
        return prompt
    
    @staticmethod
    def create_hybrid_prompt(
        company: str,
        quarter: str,
        year: int,
        query: str,
        financial_contexts: List[Dict[str, Any]],
        transcript_contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Create hybrid RAG prompt using both financial filings and transcripts.
        """
        
        # Format financial contexts
        financial_text = "**Financial Filing Contexts:**\n"
        for i, context in enumerate(financial_contexts, 1):
            financial_text += f"\nFinancial Context {i}:\n{context.get('content', '')}\n---\n"
        
        # Format transcript contexts
        transcript_text = "\n**Earnings Call Transcript Contexts:**\n"
        for i, context in enumerate(transcript_contexts, 1):
            transcript_text += f"\nTranscript Context {i}:\n{context.get('content', '')}\n---\n"
        
        prompt = f"""You are a comprehensive financial analyst with access to both quantitative and qualitative data.

**Task:** {query}

**Company:** {company}
**Quarter:** {quarter}
**Year:** {year}

{financial_text}

{transcript_text}

**Instructions:**
1. Combine insights from both financial filings and management commentary
2. Extract quantitative metrics from financial contexts
3. Extract qualitative insights from transcript contexts
4. Cross-reference information between sources when possible
5. Provide a comprehensive analysis

**Output:**
Provide both extracted metrics AND management commentary that contextualizes the numbers.

{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "metrics": {{
        // Financial metrics as before
    }},
    "commentary": {{
        // Management commentary as before
    }},
    "synthesis": {{
        "key_insights": ["<insight 1>", "<insight 2>"],
        "performance_narrative": "<how metrics and commentary align>",
        "notable_discrepancies": "<any gaps between numbers and narrative>"
    }}
}}

Provide your comprehensive analysis:"""
        
        return prompt