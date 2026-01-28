"""Chain-of-thought prompt templates for step-by-step reasoning."""

from typing import Dict, List, Any

class ChainOfThoughtPrompt:
    """
    Chain-of-thought prompts that guide the model through 
    step-by-step reasoning for more accurate extraction.
    """
    
    @staticmethod
    def create_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        document_text: str
    ) -> str:
        """
        Create a CoT prompt with reasoning steps for metric extraction.
        """
        
        prompt = f"""You are a financial analyst extracting metrics from a financial document.
Follow these steps carefully:

**Document Context:**
Company: {company}
Quarter: {quarter}
Year: {year}

**Document:**
{document_text}

**Step 1: Identify Document Structure**
First, identify what sections are present in this document:
- Is this a quarterly financial results statement?
- What tables or sections contain financial metrics?
- Where are the key metrics located?

**Step 2: Extract Core Financial Metrics**
For each metric below, explain where you found it:

1. Revenue from operations
   - Look for: "Revenue from operations", "Total revenue", "Net sales"
   - Location in document: [specify section/table]
   - Value found: [number with unit]

2. Net profit for the period
   - Look for: "Net profit", "Profit after tax", "PAT"
   - Location: [specify]
   - Value: [number with unit]

3. Basic Earnings per share
   - Look for: "Basic EPS", "Earnings per share"
   - Location: [specify]
   - Value: [number with unit]

4. Total assets
   - Look for: Balance sheet section, "Total assets"
   - Location: [specify]
   - Value: [number with unit]

5. Total liabilities
   - Look for: Balance sheet section, "Total liabilities"
   - Location: [specify]
   - Value: [number with unit]

6. Margins (if calculable or stated)
   - Gross Margin: [value or "Not Available"]
   - Operating Margin: [value or "Not Available"]

7. Business segment metrics (if available)
   - Foods Business Revenue: [value or "Not Available"]
   - Premium Personal Care contribution: [value or "Not Available"]

**Step 3: Verify and Cross-check**
- Do the numbers make logical sense?
- Are units consistent?
- Are there any obvious errors or misreadings?

**Step 4: Final Output**
Provide the extracted metrics in JSON format:

{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "reasoning": {{
        "document_type": "<identified type>",
        "key_sections_found": ["<section 1>", "<section 2>"],
        "confidence": "<high/medium/low>"
    }},
    "metrics": {{
        "revenue_from_operations": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "total_income": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "net_profit": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "basic_eps": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "total_assets": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "total_liabilities": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "interim_equity_dividend": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "gross_margin": {{"value": <number>, "unit": "%", "source": "<where found>"}},
        "operating_margin": {{"value": <number>, "unit": "%", "source": "<where found>"}},
        "foods_business_revenue": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}},
        "premium_personal_care_contribution": {{"value": <number>, "unit": "<unit>", "source": "<where found>"}}
    }}
}}

Now, work through each step and provide your analysis:"""
        
        return prompt
    
    @staticmethod
    def create_commentary_extraction_prompt(
        company: str,
        quarter: str,
        year: int,
        transcript_text: str
    ) -> str:
        """
        Create CoT prompt for extracting management commentary with reasoning.
        """
        
        prompt = f"""You are analyzing an earnings call transcript. Use step-by-step reasoning.

**Context:**
Company: {company}
Quarter: {quarter}
Year: {year}

**Transcript:**
{transcript_text}

**Step 1: Identify Speakers and Sections**
- Who are the key speakers? (CEO, CFO, etc.)
- What are the main sections? (prepared remarks, Q&A)

**Step 2: Extract Key Themes**
For each theme, identify specific quotes or statements:

1. **Revenue Performance**
   - What specific numbers or percentages were mentioned?
   - What drivers were cited (new products, market expansion, etc.)?
   - Your reasoning: [explain]

2. **Profitability Trends**
   - Were margins discussed?
   - Any cost reduction initiatives mentioned?
   - Your reasoning: [explain]

3. **Future Outlook**
   - Any guidance provided for next quarter/year?
   - Growth targets or expectations?
   - Your reasoning: [explain]

4. **Challenges Mentioned**
   - What headwinds or risks were discussed?
   - How is management addressing them?
   - Your reasoning: [explain]

5. **Strategic Initiatives**
   - New products or launches?
   - Expansion plans?
   - Your reasoning: [explain]

**Step 3: Synthesize Findings**
Combine the above into concise summaries.

**Step 4: Final Output**
{{
    "company": "{company}",
    "quarter": "{quarter}",
    "year": {year},
    "reasoning": {{
        "transcript_quality": "<assessment>",
        "key_speakers": ["<speaker 1>", "<speaker 2>"],
        "confidence": "<high/medium/low>"
    }},
    "commentary": {{
        "revenue_performance": {{"summary": "<text>", "key_quote": "<quote>"}},
        "profitability": {{"summary": "<text>", "key_quote": "<quote>"}},
        "business_outlook": {{"summary": "<text>", "key_quote": "<quote>"}},
        "key_challenges": {{"summary": "<text>", "key_quote": "<quote>"}},
        "strategic_initiatives": {{"summary": "<text>", "key_quote": "<quote>"}},
        "market_conditions": {{"summary": "<text>", "key_quote": "<quote>"}}
    }}
}}

Work through the steps now:"""
        
        return prompt
