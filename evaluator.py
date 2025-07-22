"""
Model evaluation framework for MoE models.
"""

import math
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from models.moe_model import MoETransformerModel
from data.dataset import EvaluationDataset
from evaluation.metrics import calculate_perplexity, calculate_bleu_score, calculate_code_metrics, calculate_expert_diversity

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for MoE models.
    """
    
    def __init__(
        self,
        model: MoETransformerModel,
        tokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: MoE model to evaluate
            tokenizer: Tokenizer for text processing
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device if not already
        if next(model.parameters()).device != self.device:
            self.model.to(self.device)
    
    def evaluate_language_modeling(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate language modeling performance.
        
        Args:
            dataloader: Evaluation data loader
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating language modeling performance...")
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                labels = batch['labels']
                
                # Accumulate metrics
                total_loss += loss.item()
                
                # Calculate accuracy
                if labels is not None:
                    # Flatten for accuracy calculation
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_labels = labels.view(-1)
                    
                    # Ignore padding tokens
                    mask = flat_labels != -100
                    if mask.any():
                        predictions = torch.argmax(flat_logits[mask], dim=-1)
                        correct = (predictions == flat_labels[mask]).sum().item()
                        total_correct += correct
                        total_tokens += mask.sum().item()
                
                batch_count += 1
        
        # Calculate final metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'total_batches': batch_count,
        }
        
        logger.info(f"Language modeling metrics: {metrics}")
        return metrics
    
    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        num_samples: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate text generation quality.
        
        Args:
            prompts: List of prompts for generation
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            num_samples: Number of samples per prompt
            
        Returns:
            Generation quality metrics
        """
        logger.info(f"Evaluating generation quality on {len(prompts)} prompts...")
        
        self.model.eval()
        generated_texts = []
        generation_times = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt_times = []
                prompt_generations = []
                
                # Tokenize prompt
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt, add_special_tokens=True),
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                
                for _ in range(num_samples):
                    start_time = time.time()
                    
                    # Generate text
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    generation_time = time.time() - start_time
                    prompt_times.append(generation_time)
                    
                    # Decode generated text
                    generated_text = self.tokenizer.decode(
                        generated_ids[0],
                        skip_special_tokens=True
                    )
                    
                    prompt_generations.append(generated_text)
                
                generated_texts.append(prompt_generations)
                generation_times.extend(prompt_times)
        
        # Calculate metrics
        avg_generation_time = np.mean(generation_times)
        std_generation_time = np.std(generation_times)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_generation_diversity(generated_texts)
        
        metrics = {
            'avg_generation_time': avg_generation_time,
            'std_generation_time': std_generation_time,
            'total_generations': len(prompts) * num_samples,
            **diversity_metrics,
        }
        
        logger.info(f"Generation quality metrics: {metrics}")
        return metrics, generated_texts
    
    def _calculate_generation_diversity(self, generated_texts: List[List[str]]) -> Dict[str, float]:
        """Calculate diversity metrics for generated texts."""
        all_texts = [text for prompt_texts in generated_texts for text in prompt_texts]
        
        if not all_texts:
            return {'diversity': 0.0, 'unique_ratio': 0.0}
        
        # Calculate unique ratio
        unique_texts = set(all_texts)
        unique_ratio = len(unique_texts) / len(all_texts)
        
        # Calculate average self-BLEU (lower is more diverse)
        total_self_bleu = 0.0
        comparisons = 0
        
        for prompt_texts in generated_texts:
            if len(prompt_texts) > 1:
                for i, text1 in enumerate(prompt_texts):
                    for j, text2 in enumerate(prompt_texts):
                        if i != j:
                            bleu_score = calculate_bleu_score([text2], text1)
                            total_self_bleu += bleu_score
                            comparisons += 1
        
        avg_self_bleu = total_self_bleu / max(comparisons, 1)
        diversity_score = 1.0 - avg_self_bleu  # Higher diversity = lower self-BLEU
        
        return {
            'diversity': diversity_score,
            'unique_ratio': unique_ratio,
            'avg_self_bleu': avg_self_bleu,
        }
    
    def evaluate_code_generation(
        self,
        code_dataset: EvaluationDataset,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate code generation performance.
        
        Args:
            code_dataset: Code generation dataset
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Code generation metrics
        """
        logger.info("Evaluating code generation performance...")
        
        self.model.eval()
        predictions = []
        references = []
        
        dataloader = DataLoader(code_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                prompt = batch['prompt'][0]
                reference = batch['completion'][0]
                
                # Generate code
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt, add_special_tokens=True),
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=min(len(input_ids[0]) + 200, self.model.max_seq_len),
                    temperature=0.2,  # Lower temperature for code
                    top_p=0.95,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Extract generated completion (remove prompt)
                generated_text = self.tokenizer.decode(
                    generated_ids[0][len(input_ids[0]):],
                    skip_special_tokens=True
                )
                
                predictions.append(generated_text)
                references.append(reference)
        
        # Calculate code-specific metrics
        code_metrics = calculate_code_metrics(predictions, references)
        
        logger.info(f"Code generation metrics: {code_metrics}")
        return code_metrics
    
    def analyze_expert_utilization(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Analyze expert utilization patterns.
        
        Args:
            dataloader: Data loader for analysis
            max_batches: Maximum number of batches to analyze
            
        Returns:
            Expert utilization metrics
        """
        logger.info("Analyzing expert utilization...")
        
        self.model.eval()
        all_expert_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with expert metrics
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    return_expert_metrics=True,
                )
                
                if 'expert_metrics' in outputs:
                    all_expert_metrics.append(outputs['expert_metrics'])
        
        if not all_expert_metrics:
            return {'expert_utilization_mean': 0.0, 'expert_utilization_std': 0.0}
        
        # Aggregate expert utilization across all batches
        utilizations = []
        routing_probs = []
        
        for metrics in all_expert_metrics:
            if 'expert_utilization' in metrics:
                utilizations.append(metrics['expert_utilization'].cpu().numpy())
            if 'routing_probabilities' in metrics:
                routing_probs.append(metrics['routing_probabilities'].cpu().numpy())
        
        if utilizations:
            utilizations = np.concatenate(utilizations, axis=0) if len(utilizations[0].shape) > 1 else np.stack(utilizations)
            mean_utilization = np.mean(utilizations, axis=0)
            
            # Calculate diversity metrics
            diversity = calculate_expert_diversity(utilizations)
            
            expert_metrics = {
                'expert_utilization_mean': float(np.mean(mean_utilization)),
                'expert_utilization_std': float(np.std(mean_utilization)),
                'expert_utilization_min': float(np.min(mean_utilization)),
                'expert_utilization_max': float(np.max(mean_utilization)),
                'expert_diversity': diversity,
                'load_balance_ratio': float(np.max(mean_utilization) / (np.mean(mean_utilization) + 1e-8)),
            }
        else:
            expert_metrics = {
                'expert_utilization_mean': 0.0,
                'expert_utilization_std': 0.0,
                'expert_utilization_min': 0.0,
                'expert_utilization_max': 0.0,
                'expert_diversity': 0.0,
                'load_balance_ratio': 1.0,
            }
        
        logger.info(f"Expert utilization metrics: {expert_metrics}")
        return expert_metrics
    
    def evaluate_qa_performance(
        self,
        qa_dataset: EvaluationDataset,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate question-answering performance.
        
        Args:
            qa_dataset: QA evaluation dataset
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            QA performance metrics
        """
        logger.info("Evaluating QA performance...")
        
        self.model.eval()
        predictions = []
        references = []
        
        dataloader = DataLoader(qa_dataset, batch_size=1, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                question = batch['question'][0]
                answer = batch['answer'][0]
                context = batch.get('context', [''])[0]
                
                # Format prompt
                if context:
                    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                else:
                    prompt = f"Question: {question}\nAnswer:"
                
                # Generate answer
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt, add_special_tokens=True),
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=min(len(input_ids[0]) + 100, self.model.max_seq_len),
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Extract generated answer
                generated_text = self.tokenizer.decode(
                    generated_ids[0][len(input_ids[0]):],
                    skip_special_tokens=True
                ).strip()
                
                predictions.append(generated_text)
                references.append(answer)
        
        # Calculate QA metrics (using BLEU as a proxy)
        bleu_scores = [calculate_bleu_score([ref], pred) for pred, ref in zip(predictions, references)]
        
        qa_metrics = {
            'bleu': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
            'total_samples': len(predictions),
        }
        
        logger.info(f"QA performance metrics: {qa_metrics}")
        return qa_metrics
    
    def run_comprehensive_evaluation(
        self,
        eval_config: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all tasks.
        
        Args:
            eval_config: Evaluation configuration
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        logger.info("Running comprehensive evaluation...")
        
        results = {
            'timestamp': time.time(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_layers': self.model.n_layers,
                'n_experts': self.model.n_experts,
                'top_k': self.model.top_k,
            }
        }
        
        # Language modeling evaluation
        if 'language_modeling' in eval_config:
            lm_config = eval_config['language_modeling']
            lm_dataset = EvaluationDataset(**lm_config)
            lm_dataloader = DataLoader(lm_dataset, batch_size=lm_config.get('batch_size', 4))
            
            results['language_modeling'] = self.evaluate_language_modeling(
                lm_dataloader,
                max_batches=lm_config.get('max_batches'),
            )
        
        # Code generation evaluation
        if 'code_generation' in eval_config:
            code_config = eval_config['code_generation']
            code_dataset = EvaluationDataset(task_type="code_generation", **code_config)
            
            results['code_generation'] = self.evaluate_code_generation(
                code_dataset,
                max_batches=code_config.get('max_batches'),
            )
        
        # QA evaluation
        if 'qa' in eval_config:
            qa_config = eval_config['qa']
            qa_dataset = EvaluationDataset(task_type="qa", **qa_config)
            
            results['qa'] = self.evaluate_qa_performance(
                qa_dataset,
                max_batches=qa_config.get('max_batches'),
            )
        
        # Generation quality evaluation
        if 'generation_quality' in eval_config:
            gen_config = eval_config['generation_quality']
            prompts = gen_config.get('prompts', ["The future of AI is", "In a world where"])
            
            gen_metrics, generated_texts = self.evaluate_generation_quality(
                prompts=prompts,
                **{k: v for k, v in gen_config.items() if k != 'prompts'}
            )
            results['generation_quality'] = gen_metrics
            results['generated_samples'] = generated_texts[:5]  # Save first 5 samples
        
        # Expert utilization analysis
        if 'expert_analysis' in eval_config and hasattr(self.model, 'n_experts'):
            expert_config = eval_config['expert_analysis']
            # Use language modeling data for expert analysis
            if 'language_modeling' in eval_config:
                lm_dataloader = DataLoader(
                    EvaluationDataset(**eval_config['language_modeling']),
                    batch_size=expert_config.get('batch_size', 4)
                )
                results['expert_utilization'] = self.analyze_expert_utilization(
                    lm_dataloader,
                    max_batches=expert_config.get('max_batches'),
                )
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path / "evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {output_path}")
        
        logger.info("Comprehensive evaluation completed!")
        return results
    
    def benchmark_inference_speed(
        self,
        sequence_lengths: List[int] = [128, 256, 512, 1024],
        batch_sizes: List[int] = [1, 4, 8],
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark inference speed across different configurations.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs for each configuration
            
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking inference speed...")
        
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    config_key = f"seq_{seq_len}_batch_{batch_size}"
                    times = []
                    
                    # Create dummy input
                    input_ids = torch.randint(
                        0, self.model.vocab_size,
                        (batch_size, seq_len),
                        device=self.device,
                        dtype=torch.long
                    )
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Benchmark
                    torch.cuda.synchronize()
                    for _ in range(num_runs):
                        start_time = time.time()
                        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    # Calculate statistics
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    throughput = (batch_size * seq_len) / avg_time
                    
                    results[config_key] = {
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'throughput_tokens_per_sec': throughput,
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                    }
        
        logger.info(f"Inference speed benchmark completed: {results}")
        return results
