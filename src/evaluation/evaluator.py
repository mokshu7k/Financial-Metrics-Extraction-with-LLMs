"""Unified evaluator combining accuracy and efficiency metrics."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Import from utils and other modules
from ..utils.logger import get_logger, ExperimentLogger
from ..utils.config_loader import ConfigLoader
from ..utils.helpers import (
    timer, 
    safe_json_save, 
    safe_json_load,
    ensure_directory,
    get_timestamp,
    format_number
)

from .accuracy_metrics import AccuracyEvaluator
from .efficiency_metrics import EfficiencyEvaluator

logger = get_logger(__name__)

class UnifiedEvaluator:
    """
    Unified evaluator that combines accuracy and efficiency evaluation
    to provide comprehensive comparison of extraction methods.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = ConfigLoader(config_path)
        # Setup paths
        self.results_dir = Path(self.config.get('paths.results', 'results'))
        self.outputs_dir = self.results_dir / "outputs"
        self.ground_truth_dir = Path(self.config.get('paths.ground_truth', 'data/ground_truth'))
        self.ground_truth_financial_dir = self.ground_truth_dir / "financial"
        self.ground_truth_transcript_dir = self.ground_truth_dir / "transcript"
        self.evaluation_reports_dir = ensure_directory(self.results_dir / "evaluation_reports")
        
        # Initialize evaluators
        self.accuracy_evaluator = AccuracyEvaluator()
        self.efficiency_evaluator = EfficiencyEvaluator()
        
        # Setup logging
        log_file = self.config.get('logging.file', 'results/logs/pipeline.log')
        log_level = self.config.get('logging.level', 'INFO')
        self.logger = get_logger(__name__)
        
        # Experiment logger
        self.exp_logger = ExperimentLogger("unified_evaluation", log_dir=str(self.results_dir / "logs"))
    
    @timer("Single method evaluation")
    def evaluate_method(
        self,
        method_name: str,
        evaluate_accuracy: bool = True,
        evaluate_efficiency: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single method on both accuracy and efficiency.
        
        Args:
            method_name: Name of the method (zero_shot, chain_of_thought, rag)
            evaluate_accuracy: Whether to evaluate accuracy
            evaluate_efficiency: Whether to evaluate efficiency
        
        Returns:
            Complete evaluation results
        """
        self.logger.info(f"Evaluating method: {method_name}")
        
        predictions_dir = self.outputs_dir / method_name
        
        if not predictions_dir.exists():
            self.logger.error(f"Predictions directory not found: {predictions_dir}")
            return {'error': f"Predictions not found for {method_name}"}
        
        results = {
            'method': method_name,
            'evaluation_timestamp': get_timestamp("%Y-%m-%d %H:%M:%S"),
            'predictions_dir': str(predictions_dir)
        }
        
        # Accuracy evaluation
        if evaluate_accuracy and self.ground_truth_dir.exists():
            self.logger.info(f"Evaluating accuracy for {method_name}")
            # Evaluate financial documents
            financial_results = None
            if self.ground_truth_financial_dir.exists():
                financial_results = self.accuracy_evaluator.batch_evaluate(
                    predictions_dir=predictions_dir,
                    ground_truth_dir=self.ground_truth_financial_dir,
                    method_name=method_name,
                    doc_type_filter='financial'
                )
        
            # Evaluate transcript documents
            transcript_results = None
            if self.ground_truth_transcript_dir.exists():
                transcript_results = self.accuracy_evaluator.batch_evaluate(
                    predictions_dir=predictions_dir,
                    ground_truth_dir=self.ground_truth_transcript_dir,
                    method_name=method_name,
                    doc_type_filter='transcript'
                )
        
            # Combine results
            accuracy_results = self._combine_accuracy_results(
                financial_results, 
                transcript_results
            )
            results['accuracy'] = accuracy_results
        
        # Efficiency evaluation
        if evaluate_efficiency:
            self.logger.info(f"Evaluating efficiency for {method_name}")
            efficiency_results = self.efficiency_evaluator.batch_evaluate(
                predictions_dir=predictions_dir,
                method_name=method_name
            )
            results['efficiency'] = efficiency_results
        
        return results
    
    def _combine_accuracy_results(
        self,
        financial_results: Optional[Dict[str, Any]],
        transcript_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine financial and transcript evaluation results."""
        combined = {
            'financial_results': financial_results.get('financial_results', []) if financial_results else [],
            'transcript_results': transcript_results.get('transcript_results', []) if transcript_results else [],
            'aggregate_metrics': {}
        }
    
        # Extract aggregate metrics
        if financial_results:
            combined['aggregate_metrics']['financial'] = financial_results.get('aggregate_metrics', {}).get('financial', {})
        
        if transcript_results:
            combined['aggregate_metrics']['transcript'] = transcript_results.get('aggregate_metrics', {}).get('transcript', {})
        
        # Add totals
        combined['total_documents'] = (
            financial_results.get('total_documents', 0) if financial_results else 0
        ) + (
            transcript_results.get('total_documents', 0) if transcript_results else 0
        )
        
        combined['successful_evaluations'] = (
            financial_results.get('successful_evaluations', 0) if financial_results else 0
        ) + (
            transcript_results.get('successful_evaluations', 0) if transcript_results else 0
        )
        
        return combined
    
    @timer("Comparative evaluation")
    def evaluate_all_methods(
        self,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all methods and generate comparative analysis.
        
        Args:
            methods: List of method names to evaluate. If None, evaluates all found methods.
        
        Returns:
            Comparative evaluation results
        """
        if methods is None:
            # Auto-detect methods from outputs directory
            methods = []
            for method_dir in self.outputs_dir.iterdir():
                if method_dir.is_dir() and not method_dir.name.startswith('.'):
                    methods.append(method_dir.name)
        
        if not methods:
            self.logger.error("No methods found to evaluate")
            return {'error': 'No methods found'}
        
        self.logger.info(f"Evaluating methods: {', '.join(methods)}")
        self.exp_logger.log_parameters({'methods': methods})
        
        # Evaluate each method
        method_results = {}
        for method in methods:
            result = self.evaluate_method(method)
            if 'error' not in result:
                method_results[method] = result
        
        # Generate comparative analysis
        comparison = {
            'evaluation_timestamp': get_timestamp("%Y-%m-%d %H:%M:%S"),
            'methods_evaluated': list(method_results.keys()),
            'individual_results': method_results,
            'comparative_analysis': {}
        }
        
        # Compare accuracy
        accuracy_comparison = self._compare_accuracy(method_results)
        comparison['comparative_analysis']['accuracy'] = accuracy_comparison
        
        # Compare efficiency
        efficiency_comparison = self._compare_efficiency(method_results)
        comparison['comparative_analysis']['efficiency'] = efficiency_comparison
        
        # Overall rankings
        overall_rankings = self._calculate_overall_rankings(method_results)
        comparison['comparative_analysis']['overall_rankings'] = overall_rankings
        
        # Log summary metrics
        self.exp_logger.log_metrics({
            'methods_evaluated': len(method_results),
            'best_accuracy_method': accuracy_comparison.get('best_method', 'N/A'),
            'best_efficiency_method': efficiency_comparison.get('best_method', 'N/A')
        })
        
        return comparison
    
    def _compare_accuracy(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare accuracy metrics across methods."""
        comparison = {
            'financial_metrics': {},
            'transcript_metrics': {},
            'best_method': None,
            'rankings': {}
        }
        
        # Extract financial metrics
        financial_accuracies = {}
        financial_f1_scores = {}
        financial_mape = {}
        
        for method, results in method_results.items():
            accuracy_data = results.get('accuracy', {})
            aggregate = accuracy_data.get('aggregate_metrics', {}).get('financial', {})
            
            if aggregate:
                financial_accuracies[method] = aggregate.get('avg_accuracy', 0)
                financial_f1_scores[method] = aggregate.get('avg_f1_score', 0)
                financial_mape[method] = aggregate.get('avg_mape', 0)
        
        if financial_accuracies:
            comparison['financial_metrics'] = {
                'accuracy': {
                    'values': financial_accuracies,
                    'best': max(financial_accuracies, key=financial_accuracies.get),
                    'worst': min(financial_accuracies, key=financial_accuracies.get)
                },
                'f1_score': {
                    'values': financial_f1_scores,
                    'best': max(financial_f1_scores, key=financial_f1_scores.get),
                    'worst': min(financial_f1_scores, key=financial_f1_scores.get)
                },
                'mape': {
                    'values': financial_mape,
                    'best': min(financial_mape, key=financial_mape.get),  # Lower is better
                    'worst': max(financial_mape, key=financial_mape.get)
                }
            }
        
        # Extract transcript metrics
        transcript_similarity = {}
        transcript_rouge = {}
        transcript_bleu = {}
        
        for method, results in method_results.items():
            accuracy_data = results.get('accuracy', {})
            aggregate = accuracy_data.get('aggregate_metrics', {}).get('transcript', {})
            
            if aggregate:
                transcript_similarity[method] = aggregate.get('avg_similarity', 0)
                transcript_rouge[method] = aggregate.get('avg_rouge_l', 0)
                transcript_bleu[method] = aggregate.get('avg_bleu', 0)
        
        if transcript_similarity:
            comparison['transcript_metrics'] = {
                'similarity': {
                    'values': transcript_similarity,
                    'best': max(transcript_similarity, key=transcript_similarity.get),
                    'worst': min(transcript_similarity, key=transcript_similarity.get)
                },
                'rouge_l': {
                    'values': transcript_rouge,
                    'best': max(transcript_rouge, key=transcript_rouge.get),
                    'worst': min(transcript_rouge, key=transcript_rouge.get)
                },
                'bleu': {
                    'values': transcript_bleu,
                    'best': max(transcript_bleu, key=transcript_bleu.get),
                    'worst': min(transcript_bleu, key=transcript_bleu.get)
                }
            }
        
        # Determine overall best method for accuracy
        if financial_accuracies:
            comparison['best_method'] = max(financial_accuracies, key=financial_accuracies.get)
        
        return comparison
    
    def _compare_efficiency(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare efficiency metrics across methods."""
        efficiency_data = {}
        
        for method, results in method_results.items():
            eff_results = results.get('efficiency', {})
            aggregate = eff_results.get('aggregate_metrics', {})
            
            if aggregate:
                efficiency_data[method] = aggregate
        
        if not efficiency_data:
            return {'error': 'No efficiency data available'}
        
        comparison = self.efficiency_evaluator.compare_methods(
            {method: {'aggregate_metrics': data} for method, data in efficiency_data.items()}
        )
        
        # Add best method determination
        cost_values = {
            method: data.get('avg_cost_per_doc', float('inf'))
            for method, data in efficiency_data.items()
        }
        
        if cost_values:
            comparison['best_method'] = min(cost_values, key=cost_values.get)
        
        return comparison
    
    def _calculate_overall_rankings(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall rankings combining accuracy and efficiency."""
        rankings = {
            'accuracy_weighted': {},
            'efficiency_weighted': {},
            'balanced': {}
        }
        
        # Extract key metrics for each method
        method_scores = {}
        
        for method, results in method_results.items():
            # Accuracy score (normalized to 0-1)
            accuracy_data = results.get('accuracy', {}).get('aggregate_metrics', {})
            financial = accuracy_data.get('financial', {})
            accuracy_score = financial.get('avg_accuracy', 0)
            
            # Efficiency score (inverse of cost, normalized)
            efficiency_data = results.get('efficiency', {}).get('aggregate_metrics', {})
            cost_per_doc = efficiency_data.get('avg_cost_per_doc', 1)
            
            # Normalize efficiency (inverse relationship with cost)
            method_scores[method] = {
                'accuracy': accuracy_score,
                'cost_per_doc': cost_per_doc,
                'latency': efficiency_data.get('avg_latency_seconds', 0)
            }
        
        # Calculate weighted scores
        for method, scores in method_scores.items():
            # Accuracy-weighted (70% accuracy, 30% efficiency)
            accuracy_weighted = (scores['accuracy'] * 0.7) + ((1 / (scores['cost_per_doc'] + 0.001)) * 0.3)
            rankings['accuracy_weighted'][method] = accuracy_weighted
            
            # Efficiency-weighted (30% accuracy, 70% efficiency)
            efficiency_weighted = (scores['accuracy'] * 0.3) + ((1 / (scores['cost_per_doc'] + 0.001)) * 0.7)
            rankings['efficiency_weighted'][method] = efficiency_weighted
            
            # Balanced (50% accuracy, 50% efficiency)
            balanced = (scores['accuracy'] * 0.5) + ((1 / (scores['cost_per_doc'] + 0.001)) * 0.5)
            rankings['balanced'][method] = balanced
        
        # Sort rankings
        for ranking_type in rankings:
            sorted_methods = sorted(rankings[ranking_type].items(), key=lambda x: x[1], reverse=True)
            rankings[ranking_type] = {
                'scores': dict(sorted_methods),
                'ranking': [method for method, _ in sorted_methods]
            }
        
        return rankings
    
    def generate_evaluation_report(
        self,
        comparison_results: Dict[str, Any],
        output_format: str = 'json'
    ) -> Path:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            comparison_results: Results from evaluate_all_methods
            output_format: 'json', 'csv', or 'html'
        
        Returns:
            Path to generated report
        """
        timestamp = get_timestamp()
        
        if output_format == 'json':
            report_path = self.evaluation_reports_dir / f"evaluation_report_{timestamp}.json"
            safe_json_save(comparison_results, report_path)
            
        elif output_format == 'csv':
            report_path = self.evaluation_reports_dir / f"evaluation_report_{timestamp}.csv"
            df = self._create_comparison_dataframe(comparison_results)
            df.to_csv(report_path, index=False)
            
        elif output_format == 'html':
            report_path = self.evaluation_reports_dir / f"evaluation_report_{timestamp}.html"
            html_content = self._generate_html_report(comparison_results)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        self.logger.info(f"Generated evaluation report: {report_path}")
        return report_path
    
    def _create_comparison_dataframe(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a pandas DataFrame for comparison results."""
        rows = []
        
        for method, results in comparison_results.get('individual_results', {}).items():
            row = {'method': method}
            
            # Accuracy metrics
            accuracy = results.get('accuracy', {}).get('aggregate_metrics', {})
            financial = accuracy.get('financial', {})
            transcript = accuracy.get('transcript', {})
            
            row['financial_accuracy'] = financial.get('avg_accuracy', 0)
            row['financial_f1'] = financial.get('avg_f1_score', 0)
            row['financial_mape'] = financial.get('avg_mape', 0)
            row['transcript_similarity'] = transcript.get('avg_similarity', 0)
            row['transcript_rouge_l'] = transcript.get('avg_rouge_l', 0)
            row['transcript_bleu'] = transcript.get('avg_bleu', 0)
            
            # Efficiency metrics
            efficiency = results.get('efficiency', {}).get('aggregate_metrics', {})
            row['avg_tokens'] = efficiency.get('avg_total_tokens', 0)
            row['avg_cost_usd'] = efficiency.get('avg_cost_per_doc', 0)
            row['avg_latency_sec'] = efficiency.get('avg_latency_seconds', 0)
            row['total_cost_usd'] = efficiency.get('total_cost_usd', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_html_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate HTML report with visualizations."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial RAG Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .best {{ background-color: #90EE90; }}
                .worst {{ background-color: #FFB6C1; }}
                .metric-section {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <h1>Financial RAG Evaluation Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Methods Evaluated: {methods}</p>
        """.format(
            timestamp=comparison_results.get('evaluation_timestamp', ''),
            methods=', '.join(comparison_results.get('methods_evaluated', []))
        )
        
        # Add accuracy comparison
        html += "<div class='metric-section'><h2>Accuracy Comparison</h2>"
        accuracy = comparison_results.get('comparative_analysis', {}).get('accuracy', {})
        
        if 'financial_metrics' in accuracy:
            html += "<h3>Financial Metrics Extraction</h3><table>"
            html += "<tr><th>Method</th><th>Accuracy</th><th>F1 Score</th><th>MAPE</th></tr>"
            
            methods = accuracy['financial_metrics']['accuracy']['values'].keys()
            for method in methods:
                acc = accuracy['financial_metrics']['accuracy']['values'][method]
                f1 = accuracy['financial_metrics']['f1_score']['values'][method]
                mape = accuracy['financial_metrics']['mape']['values'][method]
                
                html += f"<tr><td>{method}</td><td>{acc:.3f}</td><td>{f1:.3f}</td><td>{mape:.2f}%</td></tr>"
            
            html += "</table>"
        
        html += "</div>"
        
        # Add efficiency comparison
        html += "<div class='metric-section'><h2>Efficiency Comparison</h2>"
        efficiency = comparison_results.get('comparative_analysis', {}).get('efficiency', {})
        
        if 'comparisons' in efficiency:
            html += "<table><tr><th>Method</th><th>Avg Cost (USD)</th><th>Avg Latency (s)</th><th>Avg Tokens</th></tr>"
            
            cost_data = efficiency['comparisons'].get('avg_cost_per_doc', {}).get('values', {})
            latency_data = efficiency['comparisons'].get('avg_latency_seconds', {}).get('values', {})
            tokens_data = efficiency['comparisons'].get('avg_total_tokens', {}).get('values', {})
            
            for method in cost_data.keys():
                html += f"<tr><td>{method}</td><td>${cost_data.get(method, 0):.4f}</td>"
                html += f"<td>{latency_data.get(method, 0):.2f}</td><td>{tokens_data.get(method, 0):.0f}</td></tr>"
            
            html += "</table>"
        
        html += "</div></body></html>"
        
        return html
    
    def print_summary(self, comparison_results: Dict[str, Any]):
        """Print a summary of evaluation results to console."""
        print("\n" + "="*80)
        print("FINANCIAL RAG EVALUATION SUMMARY")
        print("="*80)
        
        methods = comparison_results.get('methods_evaluated', [])
        print(f"\nMethods Evaluated: {', '.join(methods)}")
        print(f"Evaluation Timestamp: {comparison_results.get('evaluation_timestamp', 'N/A')}")
        
        # Accuracy summary
        print("\n" + "-"*80)
        print("ACCURACY METRICS")
        print("-"*80)
        
        accuracy = comparison_results.get('comparative_analysis', {}).get('accuracy', {})
        financial = accuracy.get('financial_metrics', {})
        
        if financial:
            print("\nFinancial Metrics Extraction:")
            for method in methods:
                acc = financial['accuracy']['values'].get(method, 0)
                f1 = financial['f1_score']['values'].get(method, 0)
                mape = financial['mape']['values'].get(method, 0)
                print(f"  {method:20s} - Accuracy: {acc:.3f}, F1: {f1:.3f}, MAPE: {mape:.2f}%")
        
        transcript = accuracy.get('transcript_metrics', {})
        if transcript:
            print("\nTranscript Commentary Extraction:")
            for method in methods:
                sim = transcript['similarity']['values'].get(method, 0)
                rouge = transcript['rouge_l']['values'].get(method, 0)
                bleu = transcript['bleu']['values'].get(method, 0)
                print(f"  {method:20s} - Similarity: {sim:.3f}, ROUGE-L: {rouge:.3f}, BLEU: {bleu:.3f}")
        
        # Efficiency summary
        print("\n" + "-"*80)
        print("EFFICIENCY METRICS")
        print("-"*80)
        
        efficiency = comparison_results.get('comparative_analysis', {}).get('efficiency', {})
        comparisons = efficiency.get('comparisons', {})
        
        if comparisons:
            print("\nCost and Performance:")
            for method in methods:
                cost = comparisons.get('avg_cost_per_doc', {}).get('values', {}).get(method, 0)
                latency = comparisons.get('avg_latency_seconds', {}).get('values', {}).get(method, 0)
                tokens = comparisons.get('avg_total_tokens', {}).get('values', {}).get(method, 0)
                print(f"  {method:20s} - Cost: ${cost:.4f}, Latency: {latency:.2f}s, Tokens: {tokens:.0f}")
        
        # Overall rankings
        print("\n" + "-"*80)
        print("OVERALL RANKINGS")
        print("-"*80)
        
        rankings = comparison_results.get('comparative_analysis', {}).get('overall_rankings', {})
        
        if rankings:
            print("\nBalanced Ranking (50% Accuracy, 50% Efficiency):")
            balanced = rankings.get('balanced', {}).get('ranking', [])
            for i, method in enumerate(balanced, 1):
                print(f"  {i}. {method}")
        
        print("\n" + "="*80)


# Main execution
if __name__ == "__main__":
    print("Running unified evaluation...")
    
    evaluator = UnifiedEvaluator(config_path="config.yaml")
    
    # Evaluate all methods
    comparison_results = evaluator.evaluate_all_methods()
    
    # Print summary
    evaluator.print_summary(comparison_results)
    
    # Generate reports
    json_report = evaluator.generate_evaluation_report(comparison_results, output_format='json')
    csv_report = evaluator.generate_evaluation_report(comparison_results, output_format='csv')
    html_report = evaluator.generate_evaluation_report(comparison_results, output_format='html')
    
    print(f"\nReports generated:")
    print(f"  - JSON: {json_report}")
    print(f"  - CSV: {csv_report}")
    print(f"  - HTML: {html_report}")