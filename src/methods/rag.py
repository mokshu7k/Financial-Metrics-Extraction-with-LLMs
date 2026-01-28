"""RAG-based extraction pipeline with retrieval."""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from src.prompts.rag_prompt import RAGPrompt
from src.data_preprocessing.embed_store import FinancialVectorStoreManager
from src.utils.helpers import timer, retry_on_failure, calculate_cost, safe_json_save

logger = logging.getLogger(__name__)

class RAGExtractor:
    """
    RAG-based extraction that retrieves relevant chunks 
    before generating responses.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        retrieval_k: int = 5,
        vector_store_manager: Optional[FinancialVectorStoreManager] = None,
        api_client=None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retrieval_k = retrieval_k
        self.vector_store_manager = vector_store_manager
        self.api_client = api_client
        
        self.prompt_generator = RAGPrompt()
        
        # Track metrics
        self.total_cost = 0.0
        self.total_tokens = 0
        self.num_requests = 0
        self.num_retrievals = 0
        self.total_retrieval_time = 0.0
    
    @retry_on_failure(max_attempts=3, delay_seconds=2.0)
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Make API call to LLM."""
        start_time = datetime.now()
        
        try:
            if self.api_client and hasattr(self.api_client, 'chat'):
                # Assumes self.api_client is the groq.Groq instance
                
                response = self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a financial analyst using provided context to extract accurate information."
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
            #                 "content": "You are a financial analyst using provided context to extract accurate information."
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
            
            logger.info(f"RAG API call completed in {latency:.2f}s, cost: ${cost:.4f}")
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
                "latency_seconds": latency
            }
            
        except Exception as e:
            logger.error(f"RAG LLM API call failed: {e}")
            raise
    
    def _retrieve_contexts(
        self,
        query: str,
        company: str,
        quarter: str,
        year: int,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks.
        
        Args:
            query: Search query
            company: Filter by company
            quarter: Filter by quarter
            year: Filter by year
            document_type: Optional filter by document type
        
        Returns:
            List of retrieved contexts
        """
        if not self.vector_store_manager:
            logger.error("No vector store manager initialized")
            return []
        
        start_time = datetime.now()
        
        # Build filter criteria
        filter_criteria = {
            'company': company,
            'quarter': quarter,
            'year': year
        }
        
        if document_type:
            filter_criteria['document_type'] = document_type
        
        # Retrieve chunks
        retrieved = self.vector_store_manager.search_similar_chunks(
            query=query,
            k=self.retrieval_k,
            filter_criteria=filter_criteria
        )
        
        retrieval_time = (datetime.now() - start_time).total_seconds()
        self.total_retrieval_time += retrieval_time
        self.num_retrievals += 1
        
        logger.info(f"Retrieved {len(retrieved)} contexts in {retrieval_time:.2f}s for: {query[:50]}...")
        
        return retrieved
    
    def extract_metrics(
        self,
        company: str,
        quarter: str,
        year: int,
        query: str = "Extract all financial metrics"
    ) -> Dict[str, Any]:
        """Extract metrics using RAG."""
        logger.info(f"Extracting metrics for {company} {quarter} {year} (RAG)")
        
        # Retrieve relevant contexts
        contexts = self._retrieve_contexts(
            query=query,
            company=company,
            quarter=quarter,
            year=year,
            document_type='financial'
        )
        
        if not contexts:
            logger.warning("No contexts retrieved, falling back to empty context")
            contexts = [{"content": "No relevant context found", "score": 0.0}]
        
        # Generate prompt with contexts
        prompt = self.prompt_generator.create_extraction_prompt(
            company=company,
            quarter=quarter,
            year=year,
            query=query,
            retrieved_contexts=contexts
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        content = response["content"]
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = {
                    "error": "No JSON found in response",
                    "raw_response": content[:500]
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse RAG JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": content[:500]
            }
        
        result = {
            "method": "rag",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "retrieval": {
                "num_contexts": len(contexts),
                "top_scores": [c.get('score', 0) for c in contexts[:3]],
                "retrieval_time_seconds": self.total_retrieval_time / self.num_retrievals if self.num_retrievals > 0 else 0,
                "contexts_used": [
                    {
                        "chunk_id": c.get('chunk_id', 'unknown'),
                        "score": c.get('score', 0),
                        "section_type": c.get('section_type', 'unknown')
                    }
                    for c in contexts
                ]
            },
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
        query: str = "Extract management commentary and key discussion points"
    ) -> Dict[str, Any]:
        """Extract commentary using RAG."""
        logger.info(f"Extracting commentary for {company} {quarter} {year} (RAG)")
        
        # Retrieve relevant contexts from transcripts
        contexts = self._retrieve_contexts(
            query=query,
            company=company,
            quarter=quarter,
            year=year,
            document_type='transcript'
        )
        
        if not contexts:
            logger.warning("No transcript contexts retrieved")
            contexts = [{"content": "No relevant context found", "score": 0.0}]
        
        # Generate prompt
        prompt = self.prompt_generator.create_commentary_extraction_prompt(
            company=company,
            quarter=quarter,
            year=year,
            query=query,
            retrieved_contexts=contexts
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        content = response["content"]
        
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = {
                    "error": "No JSON found in response",
                    "raw_response": content[:500]
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse RAG JSON response: {e}")
            extracted_data = {
                "error": "Invalid JSON response",
                "raw_response": content[:500]
            }
        
        result = {
            "method": "rag",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "retrieval": {
                "num_contexts": len(contexts),
                "top_scores": [c.get('score', 0) for c in contexts[:3]],
                "retrieval_time_seconds": self.total_retrieval_time / self.num_retrievals if self.num_retrievals > 0 else 0,
                "contexts_used": [
                    {
                        "chunk_id": c.get('chunk_id', 'unknown'),
                        "score": c.get('score', 0),
                        "section_type": c.get('section_type', 'unknown')
                    }
                    for c in contexts
                ]
            },
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
        output_dir: str = "results/outputs/rag"
    ) -> List[Dict[str, Any]]:
        """Batch extraction with RAG."""
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting RAG batch extraction for {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            # Extract metadata - handle both nested and flat structures
            metadata = doc.get('metadata', {})
            company = metadata.get('company', doc.get('company', 'Unknown'))
            quarter = metadata.get('quarter', doc.get('quarter', 'Unknown'))
            year = metadata.get('year', doc.get('year', 0))
            doc_type = metadata.get('document_type', doc.get('document_type', 'unknown'))
            
            logger.info(f"Processing document {i+1}/{len(documents)} (RAG): {company}")
            
            try:
                if doc_type == 'financial':
                    result = self.extract_metrics(
                        company=company,
                        quarter=quarter,
                        year=year
                    )
                else:
                    result = self.extract_commentary(
                        company=company,
                        quarter=quarter,
                        year=year
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
            "method": "rag",
            "model": self.model_name,
            "retrieval_k": self.retrieval_k,
            "total_documents": len(documents),
            "successful_extractions": len([r for r in results if "error" not in r.get("extraction", {})]),
            "failed_extractions": len([r for r in results if "error" in r.get("extraction", {})]),
            "total_retrievals": self.num_retrievals,
            "avg_retrieval_time_seconds": self.total_retrieval_time / self.num_retrievals if self.num_retrievals > 0 else 0,
            "total_cost_usd": self.total_cost,
            "total_tokens": self.total_tokens,
            "avg_cost_per_doc": self.total_cost / len(documents) if documents else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        safe_json_save(summary, output_path / "batch_summary.json")
        
        logger.info(f"RAG batch extraction complete. Total cost: ${self.total_cost:.4f}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "method": "rag",
            "model": self.model_name,
            "retrieval_k": self.retrieval_k,
            "total_requests": self.num_requests,
            "total_retrievals": self.num_retrievals,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "avg_cost_per_request": self.total_cost / self.num_requests if self.num_requests > 0 else 0,
            "avg_tokens_per_request": self.total_tokens / self.num_requests if self.num_requests > 0 else 0,
            "avg_retrieval_time_seconds": self.total_retrieval_time / self.num_retrievals if self.num_retrievals > 0 else 0
        }