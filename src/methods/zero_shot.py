"""Zero-shot baseline extraction pipeline."""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.prompts.zero_shot import ZeroShotPrompt
from src.utils.helpers import timer, retry_on_failure, calculate_cost, safe_json_save

logger = logging.getLogger(__name__)

class ZeroShotExtractor:
    """
    Baseline extraction method using zero-shot prompting.
    No context or examples provided to the model.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        api_client=None  # Groq client
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_client = api_client
        
        self.prompt_generator = ZeroShotPrompt()
        
        # Track metrics
        self.total_cost = 0.0
        self.total_tokens = 0
        self.num_requests = 0
    
    def _truncate_text(self, text: str, max_tokens: int = 5000) -> str:
        """Truncate text to fit within token limit (rough estimate: 4 chars = 1 token)"""
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
            return text[:max_chars] + "\n\n[... text truncated due to length ...]"
        return text

    @retry_on_failure(max_attempts=3, delay_seconds=2.0)
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Make API call to LLM.
        
        Args:
            prompt: Formatted prompt
        
        Returns:
            Dictionary with response and usage stats
        """
        start_time = datetime.now()
        
        try:
            if self.api_client and hasattr(self.api_client, 'chat'):
                # Assumes self.api_client is the groq.Groq instance
                
                # Check for required system prompt for Groq/OpenAI compatible chat
                messages = [
                    {"role": "system", "content": "You are a financial analyst expert at extracting metrics from documents. Respond only with the requested JSON object."},
                    {"role": "user", "content": prompt}
                ]
                
                # NOTE: Groq's response_format is an OpenAI v1 feature. 
                # If using the groq library, this should work as intended.
                response = self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
            else:
                # Original logic if self.api_client is not set or not Groq
                raise ValueError(f"Unsupported client or client not initialized for model: {self.model_name}")
            
            # if "gpt" in self.model_name.lower():
            #     # OpenAI API
            #     response = self.api_client.chat.completions.create(
            #         model=self.model_name,
            #         messages=[
            #             {"role": "system", "content": "You are a financial analyst expert at extracting metrics from documents."},
            #             {"role": "user", "content": prompt}
            #         ],
            #         temperature=self.temperature,
            #         max_tokens=self.max_tokens,
            #         response_format={"type": "json_object"}  # Force JSON output
            #     )
                
            #     content = response.choices[0].message.content
            #     prompt_tokens = response.usage.prompt_tokens
            #     completion_tokens = response.usage.completion_tokens
                
            # elif "claude" in self.model_name.lower():
            #     # Anthropic API
            #     response = self.api_client.messages.create(
            #         model=self.model_name,
            #         max_tokens=self.max_tokens,
            #         temperature=self.temperature,
            #         messages=[
            #             {"role": "user", "content": prompt}
            #         ]
            #     )
                
            #     content = response.content[0].text
            #     prompt_tokens = response.usage.input_tokens
            #     completion_tokens = response.usage.output_tokens
            
            # else:
            #     raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Calculate cost and latency
            cost = calculate_cost(prompt_tokens, completion_tokens, self.model_name)
            latency = (datetime.now() - start_time).total_seconds()
            
            # Update tracking
            self.total_cost += cost
            self.total_tokens += prompt_tokens + completion_tokens
            self.num_requests += 1
            
            logger.info(f"API call completed in {latency:.2f}s, cost: ${cost:.4f}")
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "latency_seconds": latency
            }
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def extract_metrics(
        self,
        company: str,
        quarter: str,
        year: int,
        document_text: str
    ) -> Dict[str, Any]:
        """
        Extract financial metrics using zero-shot prompting.
        
        Args:
            company: Company name
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year
            document_text: Full document text
        
        Returns:
            Extracted metrics with metadata
        """
        logger.info(f"Extracting metrics for {company} {quarter} {year} (Zero-shot)")
        
        document_text = self._truncate_text(document_text, max_tokens=5000)

        # Generate prompt
        with timer(f"Prompt generation for {company}", log_result=False):
            prompt = self.prompt_generator.create_extraction_prompt(
                company, quarter, year, document_text
            )
        
        # Call LLM
        with timer(f"LLM call for {company}"):
            response = self._call_llm(prompt)
        
        # Parse response
        try:
            extracted_data = json.loads(response["content"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": response["content"][:500]
            }
        
        # Add metadata
        result = {
            "method": "zero_shot",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "extraction": extracted_data,
            "usage": {
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["prompt_tokens"] + response["completion_tokens"],
                "cost_usd": response["cost"],
                "latency_seconds": response["latency_seconds"]
            }
        }
        
        return result
    
    def extract_commentary(
        self,
        company: str,
        quarter: str,
        year: int,
        transcript_text: str
    ) -> Dict[str, Any]:
        """
        Extract management commentary using zero-shot prompting.
        
        Args:
            company: Company name
            quarter: Quarter
            year: Year
            transcript_text: Earnings call transcript
        
        Returns:
            Extracted commentary with metadata
        """
        logger.info(f"Extracting commentary for {company} {quarter} {year} (Zero-shot)")
        
        transcript_text = self._truncate_text(transcript_text, max_tokens=5000)
        
        # Generate prompt
        prompt = self.prompt_generator.create_commentary_extraction_prompt(
            company, quarter, year, transcript_text
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        try:
            extracted_data = json.loads(response["content"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": response["content"][:500]
            }
        
        # Add metadata
        result = {
            "method": "zero_shot",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "extraction": extracted_data,
            "usage": {
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["prompt_tokens"] + response["completion_tokens"],
                "cost_usd": response["cost"],
                "latency_seconds": response["latency_seconds"]
            }
        }
        
        return result
    
    def batch_extract(
        self,
        documents: List[Dict[str, Any]],
        output_dir: str = "results/outputs/zero_shot"
    ) -> List[Dict[str, Any]]:
        """
        Extract from multiple documents in batch.
        
        Args:
            documents: List of documents with metadata
            output_dir: Directory to save results
        
        Returns:
            List of extraction results
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch extraction for {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            # Extract metadata - handle both nested and flat structures
            metadata = doc.get('metadata', {})
            company = metadata.get('company', doc.get('company', 'Unknown'))
            quarter = metadata.get('quarter', doc.get('quarter', 'Unknown'))
            year = metadata.get('year', doc.get('year', 0))
            doc_type = metadata.get('document_type', doc.get('document_type', 'unknown'))
            cleaned_text = doc.get('cleaned_text', doc.get('raw_text', ''))
            
            logger.info(f"Processing document {i+1}/{len(documents)}: {company}")
            
            try:
                if doc_type == 'financial':
                    result = self.extract_metrics(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                else:  # transcript
                    result = self.extract_commentary(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                
                results.append(result)
                
                # Save individual result
                filename = f"{company}_{quarter}_{year}.json"
                safe_json_save(result, output_path / filename)
                
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                results.append({
                    "error": str(e),
                    "document": company
                })
        
        # Save summary
        summary = {
            "method": "zero_shot",
            "model": self.model_name,
            "total_documents": len(documents),
            "successful_extractions": len([r for r in results if "error" not in r]),
            "failed_extractions": len([r for r in results if "error" in r]),
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "avg_cost_per_doc": self.total_cost / len(documents) if documents else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        safe_json_save(summary, output_path / "batch_summary.json")
        
        logger.info(f"Batch extraction complete. Total cost: ${self.total_cost:.4f}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "method": "zero_shot",
            "model": self.model_name,
            "total_requests": self.num_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_request": self.total_cost / self.num_requests if self.num_requests > 0 else 0,
            "avg_tokens_per_request": self.total_tokens / self.num_requests if self.num_requests > 0 else 0
        }
