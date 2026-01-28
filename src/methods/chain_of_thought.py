"""Chain-of-thought extraction pipeline with step-by-step reasoning."""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.prompts.cot_prompt import ChainOfThoughtPrompt
from src.utils.helpers import timer, retry_on_failure, calculate_cost, safe_json_save

logger = logging.getLogger(__name__)

class ChainOfThoughtExtractor:
    """
    Chain-of-thought extraction that guides the model through 
    reasoning steps for improved accuracy.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2000,  # CoT needs more tokens for reasoning
        api_client=None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_client = api_client
        
        self.prompt_generator = ChainOfThoughtPrompt()
        
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
        """Make API call to LLM with CoT prompt."""
        start_time = datetime.now()
        
        try:
            if self.api_client and hasattr(self.api_client, 'chat'):
                # Assumes self.api_client is the groq.Groq instance
                
                # Use the system prompt originally intended for GPT, which works for Groq models
                response = self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a financial analyst. Think step-by-step and show your reasoning before providing the final answer."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
            else:
                raise ValueError(f"Unsupported client or client not initialized for model: {self.model_name}")
            
            # if "gpt" in self.model_name.lower():
            #     response = self.api_client.chat.completions.create(
            #         model=self.model_name,
            #         messages=[
            #             {
            #                 "role": "system", 
            #                 "content": "You are a financial analyst. Think step-by-step and show your reasoning before providing the final answer."
            #             },
            #             {"role": "user", "content": prompt}
            #         ],
            #         temperature=self.temperature,
            #         max_tokens=self.max_tokens
            #     )
                
            #     content = response.choices[0].message.content
            #     prompt_tokens = response.usage.prompt_tokens
            #     completion_tokens = response.usage.completion_tokens
                
            # elif "claude" in self.model_name.lower():
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
            
            cost = calculate_cost(prompt_tokens, completion_tokens, self.model_name)
            latency = (datetime.now() - start_time).total_seconds()
            
            self.total_cost += cost
            self.total_tokens += prompt_tokens + completion_tokens
            self.num_requests += 1
            
            logger.info(f"CoT API call completed in {latency:.2f}s, cost: ${cost:.4f}")
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "latency_seconds": latency
            }
            
        except Exception as e:
            logger.error(f"CoT LLM API call failed: {e}")
            raise
    
    def extract_metrics(
        self,
        company: str,
        quarter: str,
        year: int,
        document_text: str
    ) -> Dict[str, Any]:
        """Extract metrics using chain-of-thought reasoning."""
        logger.info(f"Extracting metrics for {company} {quarter} {year} (CoT)")
        
        document_text = self._truncate_text(document_text, max_tokens=5000)

        prompt = self.prompt_generator.create_extraction_prompt(
            company, quarter, year, document_text
        )
        
        response = self._call_llm(prompt)
        
        # Parse the reasoning and final JSON from response
        content = response["content"]
        
        # Try to extract JSON from the response (it may contain reasoning text before JSON)
        try:
            # Find JSON block in response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                extracted_data = json.loads(json_str)
                reasoning_text = content[:json_start].strip()
            else:
                extracted_data = {"error": "No JSON found in response"}
                reasoning_text = content
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CoT JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": content[:500]
            }
            reasoning_text = content
        
        result = {
            "method": "chain_of_thought",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning_text,
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
        """Extract commentary using chain-of-thought reasoning."""
        logger.info(f"Extracting commentary for {company} {quarter} {year} (CoT)")
        
        transcript_text = self._truncate_text(transcript_text, max_tokens=5000)
        
        prompt = self.prompt_generator.create_commentary_extraction_prompt(
            company, quarter, year, transcript_text
        )
        
        response = self._call_llm(prompt)
        
        content = response["content"]
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                extracted_data = json.loads(json_str)
                reasoning_text = content[:json_start].strip()
            else:
                extracted_data = {"error": "No JSON found in response"}
                reasoning_text = content
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CoT JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": content[:500]
            }
            reasoning_text = content
        
        result = {
            "method": "chain_of_thought",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning_text,
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
        output_dir: str = "results/outputs/chain_of_thought"
    ) -> List[Dict[str, Any]]:
        """Batch extraction with CoT."""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting CoT batch extraction for {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            # Extract metadata - handle both nested and flat structures
            metadata = doc.get('metadata', {})
            company = metadata.get('company', doc.get('company', 'Unknown'))
            quarter = metadata.get('quarter', doc.get('quarter', 'Unknown'))
            year = metadata.get('year', doc.get('year', 0))
            doc_type = metadata.get('document_type', doc.get('document_type', 'unknown'))
            cleaned_text = doc.get('cleaned_text', doc.get('raw_text', ''))
            
            logger.info(f"Processing document {i+1}/{len(documents)} (CoT): {company}")
            
            try:
                if doc_type == 'financial':
                    result = self.extract_metrics(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                else:
                    result = self.extract_commentary(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                
                results.append(result)
                
                filename = f"{company}_{quarter}_{year}.json"
                safe_json_save(result, output_path / filename)
                
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                results.append({
                    "error": str(e),
                    "document": company
                })
        
        summary = {
            "method": "chain_of_thought",
            "model": self.model_name,
            "total_documents": len(documents),
            "successful_extractions": len([r for r in results if "error" not in r.get("extraction", {})]),
            "failed_extractions": len([r for r in results if "error" in r.get("extraction", {})]),
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "avg_cost_per_doc": self.total_cost / len(documents) if documents else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        safe_json_save(summary, output_path / "batch_summary.json")
        
        logger.info(f"CoT batch extraction complete. Total cost: ${self.total_cost:.4f}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "method": "chain_of_thought",
            "model": self.model_name,
            "total_requests": self.num_requests,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_request": self.total_cost / self.num_requests if self.num_requests > 0 else 0,
            "avg_tokens_per_request": self.total_tokens / self.num_requests if self.num_requests > 0 else 0
        }

    def batch_extract(
        self,
        documents: List[Dict[str, Any]],
        output_dir: str = "results/outputs/chain_of_thought"
    ) -> List[Dict[str, Any]]:
        """Batch extraction with CoT."""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for financial and transcript
        financial_dir = output_path / "financial"
        transcript_dir = output_path / "transcript"
        financial_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting CoT batch extraction for {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            # Extract metadata - handle both nested and flat structures
            metadata = doc.get('metadata', {})
            company = metadata.get('company', doc.get('company', 'Unknown'))
            quarter = metadata.get('quarter', doc.get('quarter', 'Unknown'))
            year = metadata.get('year', doc.get('year', 0))
            doc_type = metadata.get('document_type', doc.get('document_type', 'unknown'))
            cleaned_text = doc.get('cleaned_text', doc.get('raw_text', ''))
            
            logger.info(f"Processing document {i+1}/{len(documents)} (CoT): {company} ({doc_type})")
            
            try:
                if doc_type == 'financial':
                    result = self.extract_metrics(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                    filename = f"{company}_{quarter}_{year}.json"
                    safe_json_save(result, financial_dir / filename)
                else:
                    result = self.extract_commentary(
                        company,
                        quarter,
                        year,
                        cleaned_text
                    )
                    filename = f"{company}_{quarter}_{year}.json"
                    safe_json_save(result, transcript_dir / filename)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                results.append({
                    "error": str(e),
                    "document": company
                })
        
        summary = {
            "method": "chain_of_thought",
            "model": self.model_name,
            "total_documents": len(documents),
            "financial_documents": len([r for r in results if r.get('extraction', {}).get('metrics')]),
            "transcript_documents": len([r for r in results if r.get('extraction', {}).get('commentary')]),
            "successful_extractions": len([r for r in results if "error" not in r.get("extraction", {})]),
            "failed_extractions": len([r for r in results if "error" in r.get("extraction", {})]),
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "avg_cost_per_doc": self.total_cost / len(documents) if documents else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        safe_json_save(summary, output_path / "batch_summary.json")
        
        logger.info(f"CoT batch extraction complete. Total cost: ${self.total_cost:.4f}")
        
        return results