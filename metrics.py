"""
Evaluation metrics for MoE model assessment.
"""

import math
import re
import ast
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity score
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')


def calculate_bleu_score(
    references: List[str],
    hypothesis: str,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """
    Calculate BLEU score for text generation evaluation.
    
    Args:
        references: List of reference texts
        hypothesis: Generated hypothesis text
        max_n: Maximum n-gram order
        smooth: Whether to apply smoothing
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    def get_ngrams(text: str, n: int) -> List[tuple]:
        """Extract n-grams from text."""
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def count_ngrams(text: str, n: int) -> Counter:
        """Count n-grams in text."""
        return Counter(get_ngrams(text, n))
    
    # Calculate precision for each n-gram order
    precisions = []
    
    for n in range(1, max_n + 1):
        hyp_ngrams = count_ngrams(hypothesis, n)
        ref_ngrams_list = [count_ngrams(ref, n) for ref in references]
        
        # Count matches
        matches = 0
        total_hyp = sum(hyp_ngrams.values())
        
        for ngram, count in hyp_ngrams.items():
            # Maximum count of this n-gram in any reference
            max_ref_count = max(ref_ngrams.get(ngram, 0) for ref_ngrams in ref_ngrams_list)
            matches += min(count, max_ref_count)
        
        if total_hyp == 0:
            precision = 0.0
        else:
            precision = matches / total_hyp
        
        # Apply smoothing for zero counts
        if smooth and precision == 0.0:
            precision = 1.0 / (2 * total_hyp) if total_hyp > 0 else 0.0
        
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if any(p == 0.0 for p in precisions):
        bleu = 0.0
    else:
        log_precisions = [math.log(p) for p in precisions]
        bleu = math.exp(sum(log_precisions) / len(log_precisions))
    
    # Brevity penalty
    hyp_len = len(hypothesis.split())
    ref_lens = [len(ref.split()) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - hyp_len))
    
    if hyp_len > closest_ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - closest_ref_len / max(hyp_len, 1))
    
    return bleu * brevity_penalty


def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Calculate ROUGE-L score based on longest common subsequence.
    
    Args:
        reference: Reference text
        hypothesis: Generated hypothesis text
        
    Returns:
        ROUGE-L F1 score
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    lcs_len = lcs_length(ref_tokens, hyp_tokens)
    
    # Calculate precision and recall
    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)


def calculate_code_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate code-specific evaluation metrics.
    
    Args:
        predictions: List of predicted code snippets
        references: List of reference code snippets
        
    Returns:
        Dictionary of code metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    metrics = {
        'exact_match': 0.0,
        'bleu': 0.0,
        'rouge_l': 0.0,
        'syntax_validity': 0.0,
        'compilation_success': 0.0,
        'identifier_overlap': 0.0,
    }
    
    total_samples = len(predictions)
    if total_samples == 0:
        return metrics
    
    exact_matches = 0
    bleu_scores = []
    rouge_scores = []
    syntax_valid = 0
    compile_success = 0
    identifier_overlaps = []
    
    for pred, ref in zip(predictions, references):
        # Exact match
        if pred.strip() == ref.strip():
            exact_matches += 1
        
        # BLEU score
        bleu = calculate_bleu_score([ref], pred)
        bleu_scores.append(bleu)
        
        # ROUGE-L score
        rouge = calculate_rouge_l(ref, pred)
        rouge_scores.append(rouge)
        
        # Syntax validity
        if is_syntactically_valid_python(pred):
            syntax_valid += 1
        
        # Compilation success (simplified check)
        if can_compile_python(pred):
            compile_success += 1
        
        # Identifier overlap
        overlap = calculate_identifier_overlap(pred, ref)
        identifier_overlaps.append(overlap)
    
    metrics['exact_match'] = exact_matches / total_samples
    metrics['bleu'] = np.mean(bleu_scores)
    metrics['rouge_l'] = np.mean(rouge_scores)
    metrics['syntax_validity'] = syntax_valid / total_samples
    metrics['compilation_success'] = compile_success / total_samples
    metrics['identifier_overlap'] = np.mean(identifier_overlaps)
    
    return metrics


def is_syntactically_valid_python(code: str) -> bool:
    """
    Check if Python code is syntactically valid.
    
    Args:
        code: Python code string
        
    Returns:
        True if syntactically valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


def can_compile_python(code: str) -> bool:
    """
    Check if Python code can be compiled.
    
    Args:
        code: Python code string
        
    Returns:
        True if can compile, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except Exception:
        return False


def calculate_identifier_overlap(pred_code: str, ref_code: str) -> float:
    """
    Calculate overlap of identifiers between predicted and reference code.
    
    Args:
        pred_code: Predicted code
        ref_code: Reference code
        
    Returns:
        Identifier overlap ratio
    """
    def extract_identifiers(code: str) -> set:
        """Extract identifiers from Python code."""
        try:
            tree = ast.parse(code)
            identifiers = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    identifiers.add(node.id)
                elif isinstance(node, ast.FunctionDef):
                    identifiers.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    identifiers.add(node.name)
            
            return identifiers
        except Exception:
            # Fallback to regex-based extraction
            pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            return set(re.findall(pattern, code))
    
    pred_identifiers = extract_identifiers(pred_code)
    ref_identifiers = extract_identifiers(ref_code)
    
    if not ref_identifiers:
        return 1.0 if not pred_identifiers else 0.0
    
    overlap = len(pred_identifiers & ref_identifiers)
    return overlap / len(ref_identifiers)


def calculate_expert_diversity(expert_utilizations: np.ndarray) -> float:
    """
    Calculate diversity of expert utilization patterns.
    
    Args:
        expert_utilizations: Array of shape [samples, experts] or [experts]
        
    Returns:
        Diversity score (higher = more diverse)
    """
    if expert_utilizations.ndim == 1:
        utilizations = expert_utilizations
    else:
        utilizations = np.mean(expert_utilizations, axis=0)
    
    # Normalize utilizations
    total_utilization = np.sum(utilizations)
    if total_utilization == 0:
        return 0.0
    
    normalized_utils = utilizations / total_utilization
    
    # Calculate entropy as diversity measure
    epsilon = 1e-10
    entropy = -np.sum(normalized_utils * np.log(normalized_utils + epsilon))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log(len(utilizations))
    
    if max_entropy == 0:
        return 1.0
    
    return entropy / max_entropy


def calculate_token_level_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Calculate token-level accuracy metrics.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: True labels [batch_size, seq_len]
        ignore_index: Index to ignore in accuracy calculation
        
    Returns:
        Token accuracy metrics
    """
    # Flatten tensors
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    
    # Create mask for valid tokens
    mask = flat_labels != ignore_index
    
    if not mask.any():
        return {'accuracy': 0.0, 'top5_accuracy': 0.0, 'total_tokens': 0}
    
    valid_logits = flat_logits[mask]
    valid_labels = flat_labels[mask]
    
    # Top-1 accuracy
    predictions = torch.argmax(valid_logits, dim=-1)
    correct = (predictions == valid_labels).float()
    accuracy = correct.mean().item()
    
    # Top-5 accuracy
    top5_predictions = torch.topk(valid_logits, k=min(5, valid_logits.size(-1)), dim=-1)[1]
    top5_correct = (top5_predictions == valid_labels.unsqueeze(-1)).any(dim=-1).float()
    top5_accuracy = top5_correct.mean().item()
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'total_tokens': mask.sum().item(),
    }


def calculate_sequence_level_metrics(
    predictions: List[str],
    references: List[str],
    tokenizer=None,
) -> Dict[str, float]:
    """
    Calculate sequence-level metrics for text generation.
    
    Args:
        predictions: List of predicted sequences
        references: List of reference sequences
        tokenizer: Optional tokenizer for token-level metrics
        
    Returns:
        Sequence-level metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    metrics = {}
    
    # BLEU scores
    bleu_scores = [calculate_bleu_score([ref], pred) for pred, ref in zip(predictions, references)]
    metrics['bleu'] = np.mean(bleu_scores)
    metrics['bleu_std'] = np.std(bleu_scores)
    
    # ROUGE-L scores
    rouge_scores = [calculate_rouge_l(ref, pred) for pred, ref in zip(predictions, references)]
    metrics['rouge_l'] = np.mean(rouge_scores)
    metrics['rouge_l_std'] = np.std(rouge_scores)
    
    # Exact match
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    metrics['exact_match'] = exact_matches / len(predictions)
    
    # Length statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = [len(ref.split()) for ref in references]
    
    metrics['avg_pred_length'] = np.mean(pred_lengths)
    metrics['avg_ref_length'] = np.mean(ref_lengths)
    metrics['length_ratio'] = np.mean([p/max(r, 1) for p, r in zip(pred_lengths, ref_lengths)])
    
    # Diversity metrics
    unique_predictions = len(set(predictions))
    metrics['unique_ratio'] = unique_predictions / len(predictions)
    
    return metrics


def calculate_calibration_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate calibration metrics for model confidence.
    
    Args:
        probabilities: Model probabilities [N, num_classes]
        labels: True labels [N]
        num_bins: Number of bins for calibration
        
    Returns:
        Calibration metrics
    """
    # Get max probabilities and predictions
    max_probs, predictions = torch.max(probabilities, dim=-1)
    correct = (predictions == labels).float()
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate calibration error
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (max_probs > bin_lower.item()) & (max_probs <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error.item())
    
    return {
        'expected_calibration_error': ece.item(),
        'maximum_calibration_error': mce,
        'average_confidence': max_probs.mean().item(),
        'accuracy': correct.mean().item(),
    }


def calculate_efficiency_metrics(
    inference_times: List[float],
    sequence_lengths: List[int],
    memory_usage: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Calculate efficiency metrics for model inference.
    
    Args:
        inference_times: List of inference times
        sequence_lengths: List of sequence lengths
        memory_usage: Optional list of memory usage values
        
    Returns:
        Efficiency metrics
    """
    metrics = {}
    
    # Time-based metrics
    metrics['avg_inference_time'] = np.mean(inference_times)
    metrics['std_inference_time'] = np.std(inference_times)
    metrics['min_inference_time'] = np.min(inference_times)
    metrics['max_inference_time'] = np.max(inference_times)
    
    # Throughput metrics
    total_tokens = sum(sequence_lengths)
    total_time = sum(inference_times)
    metrics['tokens_per_second'] = total_tokens / total_time if total_time > 0 else 0.0
    
    # Per-token timing
    per_token_times = [t / max(l, 1) for t, l in zip(inference_times, sequence_lengths)]
    metrics['avg_time_per_token'] = np.mean(per_token_times)
    
    # Memory metrics
    if memory_usage:
        metrics['avg_memory_usage'] = np.mean(memory_usage)
        metrics['peak_memory_usage'] = np.max(memory_usage)
        metrics['memory_efficiency'] = total_tokens / max(np.mean(memory_usage), 1)
    
    return metrics


class MetricsTracker:
    """
    Utility class for tracking and aggregating metrics during training/evaluation.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(dict)
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
            if step is not None:
                self.step_metrics[step][key] = value
    
    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics[metric_name]
        return np.mean(values) if values else 0.0
    
    def get_latest(self, metric_name: str) -> float:
        """Get latest value for a metric."""
        values = self.metrics[metric_name]
        return values[-1] if values else 0.0
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1],
                    'count': len(values),
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0,
                    'max': 0.0, 'latest': 0.0, 'count': 0,
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()
    
    def save_to_file(self, filepath: str):
        """Save metrics to file."""
        import json
        
        data = {
            'summary': self.get_summary(),
            'step_metrics': dict(self.step_metrics),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """Load metrics from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'step_metrics' in data:
            self.step_metrics = defaultdict(dict, {
                int(k): v for k, v in data['step_metrics'].items()
            })
            
            # Rebuild metrics from step_metrics
            self.metrics = defaultdict(list)
            for step_data in self.step_metrics.values():
                for metric, value in step_data.items():
                    self.metrics[metric].append(value)


def run_code_execution_test(code: str, test_cases: List[Dict]) -> Dict[str, Any]:
    """
    Run code execution tests for generated code.
    
    Args:
        code: Generated code to test
        test_cases: List of test cases with 'input' and 'expected' keys
        
    Returns:
        Execution test results
    """
    results = {
        'passed': 0,
        'failed': 0,
        'error': 0,
        'total': len(test_cases),
        'execution_time': 0.0,
        'details': [],
    }
    
    if not test_cases:
        return results
    
    try:
        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        import time
        start_time = time.time()
        
        for i, test_case in enumerate(test_cases):
            test_input = test_case.get('input', '')
            expected_output = test_case.get('expected', '')
            
            try:
                # Execute the code with test input
                result = subprocess.run(
                    ['python', temp_file],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=5,  # 5 second timeout
                )
                
                if result.returncode == 0:
                    actual_output = result.stdout.strip()
                    if actual_output == expected_output.strip():
                        results['passed'] += 1
                        status = 'passed'
                    else:
                        results['failed'] += 1
                        status = 'failed'
                else:
                    results['error'] += 1
                    status = 'error'
                    actual_output = result.stderr
                
                results['details'].append({
                    'test_case': i,
                    'status': status,
                    'input': test_input,
                    'expected': expected_output,
                    'actual': actual_output,
                })
                
            except subprocess.TimeoutExpired:
                results['error'] += 1
                results['details'].append({
                    'test_case': i,
                    'status': 'timeout',
                    'input': test_input,
                    'expected': expected_output,
                    'actual': 'Execution timed out',
                })
            except Exception as e:
                results['error'] += 1
                results['details'].append({
                    'test_case': i,
                    'status': 'exception',
                    'input': test_input,
                    'expected': expected_output,
                    'actual': str(e),
                })
        
        results['execution_time'] = time.time() - start_time
        
        # Clean up temporary file
        import os
        os.unlink(temp_file)
        
    except Exception as e:
        logger.error(f"Error running code execution tests: {e}")
        results['error'] = len(test_cases)
    
    return results
