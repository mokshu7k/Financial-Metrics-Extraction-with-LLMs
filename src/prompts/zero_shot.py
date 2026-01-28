"""Zero-shot prompt templates for baseline extraction."""

from typing import Dict, List, Any, Optional

class ZeroShotPrompt:
    """
    Zero-shot prompt template for extracting financial metrics 
    without any context or examples.
    """
    
    # Metrics to extract based on your requirements
    METRICS = [
        "Revenue from operations",
        "Total Income",
        "Net profit for the period",
        "Basic Earnings per share (of Re. 1/-each)",
        "Total assets",
        "Total liabilities",
        "Interim Equity Dividend",
        "Gross Margin",
        "Operating Margin",
        "Foods Business Revenue",
        "Premium Personal Care contribution to domestic revenue"
    ]
    
    @staticmethod
    def create_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        document_text: str
    ) -> str:
        """
        Create a zero-shot prompt for metric extraction.
        
        Args:
            company: Company name
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            document_text: Full document text
        
        Returns:
            Formatted prompt string
        """
        
        metrics_list = "\n".join([f"- {metric}" for metric in ZeroShotPrompt.METRICS])
        
        prompt = f"""You are a financial analyst extracting key metrics from financial documents.

Extract the following metrics from the financial document for {company} for {quarter} {year}:

{metrics_list}

Document:
{document_text}

Instructions:
1. Extract each metric value if present in the document
2. If a metric is not found, return "Not Available"
3. Include the unit (e.g., crores, INR, percentage)
4. For ratios and margins, include the percentage sign
5. Be precise and only extract values explicitly stated in the document

Output format (JSON):
{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "metrics": {{
        "revenue_from_operations": {{"value": <number>, "unit": "<unit>"}},
        "total_income": {{"value": <number>, "unit": "<unit>"}},
        "net_profit": {{"value": <number>, "unit": "<unit>"}},
        "basic_eps": {{"value": <number>, "unit": "<unit>"}},
        "total_assets": {{"value": <number>, "unit": "<unit>"}},
        "total_liabilities": {{"value": <number>, "unit": "<unit>"}},
        "interim_equity_dividend": {{"value": <number>, "unit": "<unit>"}},
        "gross_margin": {{"value": <number>, "unit": "%"}},
        "operating_margin": {{"value": <number>, "unit": "%"}},
        "foods_business_revenue": {{"value": <number>, "unit": "<unit>"}},
        "premium_personal_care_contribution": {{"value": <number>, "unit": "<unit>"}}
    }}
}}

Extract the metrics now:"""
        
        return prompt
    
    @staticmethod
    def create_commentary_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        transcript_text: str
    ) -> str:
        """
        Create a zero-shot prompt for extracting management commentary.
        
        Args:
            company: Company name
            quarter: Quarter
            year: Year
            transcript_text: Earnings call transcript
        
        Returns:
            Formatted prompt
        """
        
        prompt = f"""You are analyzing an earnings call transcript for {company} for {quarter} {year}.

Transcript:
{transcript_text}

Extract and summarize the following key discussion points:

1. Revenue Performance: What did management say about revenue growth and drivers?
2. Profitability: Comments on margins, cost management, and profitability trends?
3. Business Outlook: Forward-looking statements and guidance?
4. Key Challenges: Any headwinds or challenges mentioned?
5. Strategic Initiatives: New products, expansions, or strategic priorities?
6. Market Conditions: Comments on market trends and competition?

Provide concise, factual summaries for each point based only on what was explicitly stated in the transcript.

Output format (JSON):
{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "commentary": {{
        "revenue_performance": "<summary>",
        "profitability": "<summary>",
        "business_outlook": "<summary>",
        "key_challenges": "<summary>",
        "strategic_initiatives": "<summary>",
        "market_conditions": "<summary>"
    }}
}}

Extract the commentary now:"""
        
        return prompt

