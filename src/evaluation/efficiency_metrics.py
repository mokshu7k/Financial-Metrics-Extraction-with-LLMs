
"""Efficiency metrics for evaluating computational performance."""

import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Import from utils
from ..utils.logger import get_logger
from ..utils.helpers import safe_json_load, format_number

logger = get_logger(__name__)

class EfficiencyEvaluator:
    """
    Evaluates efficiency metrics: token usage, cost, and latency
    for different extraction methods.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Cost per 1K tokens (as of 2024)
        self.pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'claude-3-opus': {'input': 0.015, 'output': 0.075}
        }
    
    def extract_usage_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage metrics from a result dictionary."""
        usage = result.get('usage', {})
        
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)),
            'cost_usd': usage.get('cost_usd', 0),
            'latency_seconds': usage.get('latency_seconds', 0)
        }
    
    def evaluate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate efficiency for a single extraction result.
        
        Args:
            result: Extraction result with usage data
        
        Returns:
            Efficiency metrics
        """
        metrics = self.extract_usage_metrics(result)
        
        # Add method-specific metrics
        method = result.get('method', 'unknown')
        
        efficiency_result = {
            'method': method,
            'model': result.get('model', 'unknown'),
            'timestamp': result.get('timestamp', ''),
            **metrics
        }
        
        # For RAG, add retrieval metrics
        if method == 'rag':
            retrieval = result.get('retrieval', {})
            efficiency_result['retrieval_metrics'] = {
                'num_contexts_retrieved': retrieval.get('num_contexts', 0),
                'retrieval_time_seconds': retrieval.get('retrieval_time_seconds', 0),
                'avg_retrieval_score': np.mean(retrieval.get('top_scores', [0])) if retrieval.get('top_scores') else 0
            }
        
        # Calculate derived metrics
        if metrics['total_tokens'] > 0:
            efficiency_result['cost_per_1k_tokens'] = (metrics['cost_usd'] / metrics['total_tokens']) * 1000
        else:
            efficiency_result['cost_per_1k_tokens'] = 0
        
        if metrics['latency_seconds'] > 0:
            efficiency_result['tokens_per_second'] = metrics['total_tokens'] / metrics['latency_seconds']
        else:
            efficiency_result['tokens_per_second'] = 0
        
        return efficiency_result
    
    def batch_evaluate(
        self,
        predictions_dir: Path,
        method_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate efficiency for all predictions of a given method.
        
        Args:
            predictions_dir: Directory containing prediction files
            method_name: Name of the method
        
        Returns:
            Aggregate efficiency metrics
        """
        results = {
            'method': method_name,
            'total_documents': 0,
            'successful_evaluations': 0,
            'individual_results': [],
            'aggregate_metrics': {}
        }
        
        # Collect metrics from all files
        for pred_file in predictions_dir.glob("*.json"):
            if 'summary' in pred_file.name:
                continue
            
            try:
                prediction = safe_json_load(pred_file)
                if not prediction:
                    continue

                efficiency_result = self.evaluate_single_result(prediction)
                results['individual_results'].append(efficiency_result)
                results['successful_evaluations'] += 1
                results['total_documents'] += 1
                
            except Exception as e:
                self.logger.error(f"Error evaluating efficiency for {pred_file.name}: {e}")
        
        # Calculate aggregate metrics
        if results['individual_results']:
            results['aggregate_metrics'] = self._calculate_aggregate_metrics(
                results['individual_results']
            )
        
        return results
    
    def _calculate_aggregate_metrics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate efficiency metrics."""
        # Extract all metrics
        prompt_tokens = [r['prompt_tokens'] for r in individual_results]
        completion_tokens = [r['completion_tokens'] for r in individual_results]
        total_tokens = [r['total_tokens'] for r in individual_results]
        costs = [r['cost_usd'] for r in individual_results]
        latencies = [r['latency_seconds'] for r in individual_results]
        
        aggregate = {
            'total_prompt_tokens': sum(prompt_tokens),
            'total_completion_tokens': sum(completion_tokens),
            'total_tokens': sum(total_tokens),
            'total_cost_usd': sum(costs),
            'avg_prompt_tokens': np.mean(prompt_tokens),
            'avg_completion_tokens': np.mean(completion_tokens),
            'avg_total_tokens': np.mean(total_tokens),
            'avg_cost_per_doc': np.mean(costs),
            'avg_latency_seconds': np.mean(latencies),
            'median_latency_seconds': np.median(latencies),
            'p95_latency_seconds': np.percentile(latencies, 95),
            'total_processing_time': sum(latencies)
        }
        
        # Add RAG-specific metrics if applicable
        retrieval_metrics = [r.get('retrieval_metrics') for r in individual_results if 'retrieval_metrics' in r]
        if retrieval_metrics:
            avg_contexts = np.mean([rm['num_contexts_retrieved'] for rm in retrieval_metrics])
            avg_retrieval_time = np.mean([rm['retrieval_time_seconds'] for rm in retrieval_metrics])
            
            aggregate['avg_contexts_retrieved'] = avg_contexts
            aggregate['avg_retrieval_time_seconds'] = avg_retrieval_time
            aggregate['retrieval_overhead_pct'] = (avg_retrieval_time / aggregate['avg_latency_seconds']) * 100 if aggregate['avg_latency_seconds'] > 0 else 0
        
        # Calculate throughput
        if aggregate['total_processing_time'] > 0:
            aggregate['throughput_docs_per_minute'] = (len(individual_results) / aggregate['total_processing_time']) * 60
        else:
            aggregate['throughput_docs_per_minute'] = 0
        
        return aggregate
    
    def compare_methods(
        self,
        method_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare efficiency across multiple methods.
        
        Args:
            method_results: Dictionary mapping method names to their efficiency results
        
        Returns:
            Comparison results
        """
        comparison = {
            'methods_compared': list(method_results.keys()),
            'comparisons': {},
            'rankings': {}
        }
        
        # Extract aggregate metrics for each method
        metrics_by_method = {}
        for method_name, results in method_results.items():
            metrics_by_method[method_name] = results.get('aggregate_metrics', {})
        
        # Compare key metrics
        comparison_metrics = [
            'avg_total_tokens',
            'avg_cost_per_doc',
            'avg_latency_seconds',
            'total_cost_usd'
        ]
        
        for metric in comparison_metrics:
            values = {
                method: metrics.get(metric, 0)
                for method, metrics in metrics_by_method.items()
            }
            
            if values:
                comparison['comparisons'][metric] = {
                    'values': values,
                    'best_method': min(values, key=values.get),
                    'worst_method': max(values, key=values.get),
                    'range': max(values.values()) - min(values.values())
                }
        
        # Calculate rankings
        ranking_metrics = {
            'cost_efficiency': 'avg_cost_per_doc',  # Lower is better
            'speed': 'avg_latency_seconds',  # Lower is better
            'token_efficiency': 'avg_total_tokens'  # Lower is better
        }
        
        for rank_name, metric_name in ranking_metrics.items():
            values = {
                method: metrics.get(metric_name, float('inf'))
                for method, metrics in metrics_by_method.items()
            }
            sorted_methods = sorted(values.items(), key=lambda x: x[1])
            comparison['rankings'][rank_name] = [method for method, _ in sorted_methods]
        
        return comparison

