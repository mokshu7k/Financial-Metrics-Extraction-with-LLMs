"""accuracy_metrics.py"""

"""Accuracy metrics for evaluating extraction quality."""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from difflib import SequenceMatcher

# Import from utils
from ..utils.logger import get_logger
from ..utils.helpers import safe_json_load, format_number

logger = get_logger(__name__)

class AccuracyEvaluator:
    """
    Evaluates accuracy of extracted financial metrics and commentary
    against ground truth annotations.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Define metric types
        self.metric_types = {
            'revenue_from_operations': 'numerical',
            'total_income': 'numerical',
            'net_profit': 'numerical',
            'basic_eps': 'numerical',
            'total_assets': 'numerical',
            'total_liabilities': 'numerical',
            'interim_equity_dividend': 'numerical',
            'gross_margin': 'percentage',
            'operating_margin': 'percentage',
            'foods_business_revenue': 'numerical',
            'premium_personal_care_contribution': 'numerical'
        }
        
        # Define tolerance levels for numerical comparison
        self.tolerance = {
            'numerical': 0.05,  # 5% tolerance
            'percentage': 0.02  # 2 percentage points
        }
    
    def extract_numerical_value(self, value: Any) -> Optional[float]:
        """Extract numerical value from various formats."""
        if value is None or value == "Not Available":
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, dict):
            value = value.get('value', None)
        
        if isinstance(value, str):
            # Remove commas and units
            value = re.sub(r'[,\s]', '', value)
            # Extract first number
            match = re.search(r'[-+]?\d*\.?\d+', value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        
        return None
    
    def exact_match(self, predicted: Any, ground_truth: Any, metric_name: str) -> bool:
        """
        Check if prediction exactly matches ground truth.
        
        Args:
            predicted: Predicted value
            ground_truth: Ground truth value
            metric_name: Name of the metric
        
        Returns:
            True if exact match, False otherwise
        """
        pred_val = self.extract_numerical_value(predicted)
        gt_val = self.extract_numerical_value(ground_truth)
        
        # Both missing
        if pred_val is None and gt_val is None:
            return True
        
        # One missing
        if pred_val is None or gt_val is None:
            return False
        
        # Numerical comparison with tolerance
        metric_type = self.metric_types.get(metric_name, 'numerical')
        tolerance = self.tolerance.get(metric_type, 0.05)
        
        if gt_val == 0:
            return abs(pred_val - gt_val) < 0.01
        
        relative_error = abs(pred_val - gt_val) / abs(gt_val)
        return relative_error <= tolerance
    
    def calculate_numerical_error(
        self, 
        predicted: Any, 
        ground_truth: Any
    ) -> Optional[float]:
        """Calculate absolute percentage error for numerical values."""
        pred_val = self.extract_numerical_value(predicted)
        gt_val = self.extract_numerical_value(ground_truth)
        
        if pred_val is None or gt_val is None or gt_val == 0:
            return None
        
        return abs(pred_val - gt_val) / abs(gt_val) * 100
    
    def evaluate_metrics_extraction(
        self,
        predicted_metrics: Dict[str, Any],
        ground_truth_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate accuracy of extracted financial metrics.
        
        Args:
            predicted_metrics: Predicted metric values
            ground_truth_metrics: Ground truth metric values
        
        Returns:
            Dictionary with accuracy metrics
        """
        results = {
            'total_metrics': 0,
            'exact_matches': 0,
            'within_tolerance': 0,
            'missing_predictions': 0,
            'incorrect_predictions': 0,
            'metric_wise_results': {},
            'errors': []
        }
        
        for metric_name in self.metric_types.keys():
            gt_value = ground_truth_metrics.get(metric_name)
            pred_value = predicted_metrics.get(metric_name)
            
            results['total_metrics'] += 1
            
            # Check if prediction exists
            if pred_value is None or pred_value == "Not Available":
                if gt_value is not None and gt_value != "Not Available":
                    results['missing_predictions'] += 1
                    results['metric_wise_results'][metric_name] = {
                        'status': 'missing',
                        'ground_truth': gt_value,
                        'predicted': None
                    }
                else:
                    results['exact_matches'] += 1
                    results['metric_wise_results'][metric_name] = {
                        'status': 'correct_na',
                        'ground_truth': None,
                        'predicted': None
                    }
                continue
            
            # Check exact match
            is_match = self.exact_match(pred_value, gt_value, metric_name)
            
            if is_match:
                results['exact_matches'] += 1
                results['within_tolerance'] += 1
                results['metric_wise_results'][metric_name] = {
                    'status': 'exact_match',
                    'ground_truth': gt_value,
                    'predicted': pred_value,
                    'error_pct': 0.0
                }
            else:
                error = self.calculate_numerical_error(pred_value, gt_value)
                results['incorrect_predictions'] += 1
                results['metric_wise_results'][metric_name] = {
                    'status': 'incorrect',
                    'ground_truth': gt_value,
                    'predicted': pred_value,
                    'error_pct': error
                }
                
                if error is not None:
                    results['errors'].append(error)
        
        # Calculate aggregate metrics
        if results['total_metrics'] > 0:
            results['accuracy'] = results['exact_matches'] / results['total_metrics']
            results['precision'] = results['exact_matches'] / (results['exact_matches'] + results['incorrect_predictions']) if (results['exact_matches'] + results['incorrect_predictions']) > 0 else 0
            results['recall'] = results['exact_matches'] / (results['exact_matches'] + results['missing_predictions']) if (results['exact_matches'] + results['missing_predictions']) > 0 else 0
            
            if results['precision'] + results['recall'] > 0:
                results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])
            else:
                results['f1_score'] = 0
        
        # Calculate average error
        if results['errors']:
            results['mean_absolute_percentage_error'] = np.mean(results['errors'])
            results['median_absolute_percentage_error'] = np.median(results['errors'])
        else:
            results['mean_absolute_percentage_error'] = 0.0
            results['median_absolute_percentage_error'] = 0.0
        
        return results
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def calculate_rouge_l(self, predicted: str, reference: str) -> float:
        """
        Calculate ROUGE-L score (Longest Common Subsequence).
        Simplified implementation without external dependencies.
        """
        if not predicted or not reference:
            return 0.0
        
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()
        
        # Dynamic programming for LCS
        m, n = len(pred_words), len(ref_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # Calculate precision and recall
        precision = lcs_length / len(pred_words) if pred_words else 0
        recall = lcs_length / len(ref_words) if ref_words else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """
        Calculate simplified BLEU score (unigram and bigram).
        Simplified implementation without external dependencies.
        """
        if not predicted or not reference:
            return 0.0
        
        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()
        
        # Unigram precision
        pred_unigrams = set(pred_words)
        ref_unigrams = set(ref_words)
        unigram_matches = len(pred_unigrams & ref_unigrams)
        unigram_precision = unigram_matches / len(pred_unigrams) if pred_unigrams else 0
        
        # Bigram precision
        pred_bigrams = set(zip(pred_words[:-1], pred_words[1:]))
        ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
        bigram_matches = len(pred_bigrams & ref_bigrams)
        bigram_precision = bigram_matches / len(pred_bigrams) if pred_bigrams else 0
        
        # Geometric mean
        if unigram_precision > 0 and bigram_precision > 0:
            bleu = (unigram_precision * bigram_precision) ** 0.5
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, len(pred_words) / len(ref_words)) if ref_words else 0
        
        return bleu * bp
    
    def evaluate_commentary_extraction(
        self,
        predicted_commentary: Dict[str, str],
        ground_truth_commentary: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate accuracy of extracted commentary using text similarity metrics.
        
        Args:
            predicted_commentary: Predicted commentary
            ground_truth_commentary: Ground truth commentary
        
        Returns:
            Dictionary with commentary evaluation metrics
        """
        results = {
            'total_fields': 0,
            'fields_evaluated': 0,
            'similarity_scores': {},
            'rouge_l_scores': {},
            'bleu_scores': {},
            'missing_fields': []
        }
        
        commentary_fields = [
            'revenue_performance',
            'profitability',
            'business_outlook',
            'key_challenges',
            'strategic_initiatives',
            'market_conditions'
        ]
        
        for field in commentary_fields:
            results['total_fields'] += 1
            
            gt_text = ground_truth_commentary.get(field, '')
            pred_text = predicted_commentary.get(field, '')
            
            if isinstance(pred_text, dict):
                pred_text = pred_text.get('summary', '')
            
            if not pred_text or pred_text == "Not Available":
                results['missing_fields'].append(field)
                continue
            
            if not gt_text:
                continue
            
            results['fields_evaluated'] += 1
            
            # Calculate similarity metrics
            similarity = self.calculate_text_similarity(pred_text, gt_text)
            rouge_l = self.calculate_rouge_l(pred_text, gt_text)
            bleu = self.calculate_bleu_score(pred_text, gt_text)
            
            results['similarity_scores'][field] = similarity
            results['rouge_l_scores'][field] = rouge_l
            results['bleu_scores'][field] = bleu
        
        # Calculate aggregate scores
        if results['fields_evaluated'] > 0:
            results['avg_similarity'] = np.mean(list(results['similarity_scores'].values()))
            results['avg_rouge_l'] = np.mean(list(results['rouge_l_scores'].values()))
            results['avg_bleu'] = np.mean(list(results['bleu_scores'].values()))
        else:
            results['avg_similarity'] = 0.0
            results['avg_rouge_l'] = 0.0
            results['avg_bleu'] = 0.0
        
        results['coverage'] = results['fields_evaluated'] / results['total_fields'] if results['total_fields'] > 0 else 0
        
        return results
    
    def evaluate_extraction_result(
        self,
        predicted: Dict[str, Any],
        ground_truth: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single extraction result.
        
        Args:
            predicted: Predicted extraction
            ground_truth: Ground truth extraction
            document_type: 'financial' or 'transcript'
        
        Returns:
            Evaluation results
        """
        result = {
            'document_type': document_type,
            'timestamp': predicted.get('timestamp', ''),
            'method': predicted.get('method', 'unknown')
        }
        
        if document_type == 'financial':
            # Evaluate metrics
            pred_metrics = predicted.get('extraction', {}).get('metrics', {})
            gt_metrics = ground_truth.get('metrics', {})
            
            metrics_eval = self.evaluate_metrics_extraction(pred_metrics, gt_metrics)
            result['metrics_evaluation'] = metrics_eval
            
        elif document_type == 'transcript':
            # Evaluate commentary
            pred_commentary = predicted.get('extraction', {}).get('commentary', {})
            gt_commentary = ground_truth.get('commentary', {})
            
            commentary_eval = self.evaluate_commentary_extraction(pred_commentary, gt_commentary)
            result['commentary_evaluation'] = commentary_eval
        
        return result
    
    def batch_evaluate(
        self,
        predictions_dir: Path,
        ground_truth_dir: Path,
        method_name: str,
        doc_type_filter: Optional[str] = None  # Add this parameter: 'financial' or 'transcript'
    ) -> Dict[str, Any]:
        """
        Evaluate all predictions for a given method against ground truth.
        
        Args:
            predictions_dir: Directory containing prediction files
            ground_truth_dir: Directory containing ground truth files
            method_name: Name of the method being evaluated
            doc_type_filter: Only evaluate this document type ('financial' or 'transcript')
        """
        results = {
            'method': method_name,
            'total_documents': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'financial_results': [],
            'transcript_results': [],
            'aggregate_metrics': {}
        }
        
        for pred_file in predictions_dir.glob("*.json"):
            if 'summary' in pred_file.name:
                continue
            
            try:
                prediction = safe_json_load(pred_file)
                if not prediction:
                    continue
                
                # Determine document type FIRST
                extraction = prediction.get('extraction', {})
                if 'metrics' in extraction:
                    doc_type = 'financial'
                elif 'commentary' in extraction:
                    doc_type = 'transcript'
                else:
                    self.logger.warning(f"Cannot determine document type for {pred_file.name}")
                    continue
                
                # Skip if doesn't match filter
                if doc_type_filter and doc_type != doc_type_filter:
                    continue
                
                # Now search for ground truth in the CORRECT directory
                gt_file = ground_truth_dir / pred_file.name
                
                if not gt_file.exists():
                    company = extraction.get('company', '')
                    quarter = extraction.get('quarter', '')
                    
                    for candidate in ground_truth_dir.glob("*.json"):
                        candidate_data = safe_json_load(candidate)
                        if (candidate_data.get('company') == company and 
                            candidate_data.get('quarter') == quarter):
                            gt_file = candidate
                            break
                
                if not gt_file.exists():
                    self.logger.warning(f"No ground truth found for {pred_file.name}")
                    results['failed_evaluations'] += 1
                    continue
                
                ground_truth = safe_json_load(gt_file)
                if not ground_truth:
                    continue
                
                eval_result = self.evaluate_extraction_result(
                    prediction, ground_truth, doc_type
                )
                
                if doc_type == 'financial':
                    results['financial_results'].append(eval_result)
                else:
                    results['transcript_results'].append(eval_result)
                
                results['successful_evaluations'] += 1
                results['total_documents'] += 1
                
            except Exception as e:
                self.logger.error(f"Error evaluating {pred_file.name}: {e}")
                results['failed_evaluations'] += 1
        
        results['aggregate_metrics'] = self._calculate_aggregate_metrics(results)
        return results
    
    def _calculate_aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all evaluations."""
        aggregate = {
            'financial': {},
            'transcript': {}
        }
        
        # Aggregate financial metrics
        if results['financial_results']:
            accuracies = []
            f1_scores = []
            mape_values = []
            
            for eval_result in results['financial_results']:
                metrics_eval = eval_result.get('metrics_evaluation', {})
                if 'accuracy' in metrics_eval:
                    accuracies.append(metrics_eval['accuracy'])
                if 'f1_score' in metrics_eval:
                    f1_scores.append(metrics_eval['f1_score'])
                if 'mean_absolute_percentage_error' in metrics_eval:
                    mape_values.append(metrics_eval['mean_absolute_percentage_error'])
            
            aggregate['financial'] = {
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'avg_f1_score': np.mean(f1_scores) if f1_scores else 0,
                'avg_mape': np.mean(mape_values) if mape_values else 0,
                'num_documents': len(results['financial_results'])
            }
        
        # Aggregate transcript metrics
        if results['transcript_results']:
            similarities = []
            rouge_scores = []
            bleu_scores = []
            coverages = []
            
            for eval_result in results['transcript_results']:
                commentary_eval = eval_result.get('commentary_evaluation', {})
                if 'avg_similarity' in commentary_eval:
                    similarities.append(commentary_eval['avg_similarity'])
                if 'avg_rouge_l' in commentary_eval:
                    rouge_scores.append(commentary_eval['avg_rouge_l'])
                if 'avg_bleu' in commentary_eval:
                    bleu_scores.append(commentary_eval['avg_bleu'])
                if 'coverage' in commentary_eval:
                    coverages.append(commentary_eval['coverage'])
            
            aggregate['transcript'] = {
                'avg_similarity': np.mean(similarities) if similarities else 0,
                'avg_rouge_l': np.mean(rouge_scores) if rouge_scores else 0,
                'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0,
                'avg_coverage': np.mean(coverages) if coverages else 0,
                'num_documents': len(results['transcript_results'])
            }
        
        return aggregate
