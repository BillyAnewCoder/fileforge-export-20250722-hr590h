"""
Main trainer for MoE model with distributed training support.
"""

import os
import time
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from transformers import get_cosine_schedule_with_warmup

from models.moe_model import MoETransformerModel
from data.dataset import MoEDataLoader
from evaluation.evaluator import ModelEvaluator
from utils.logging_utils import get_logger, log_metrics
from utils.memory_utils import MemoryTracker, log_memory_stats
from utils.distributed_utils import is_main_process, get_world_size, get_rank

logger = get_logger(__name__)


class MoETrainer:
    """
    Trainer for Mixture-of-Experts model with advanced features.
    """
    
    def __init__(
        self,
        model: MoETransformerModel,
        train_dataloader: MoEDataLoader,
        eval_dataloader: Optional[MoEDataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        config: Optional[Dict] = None,
        deepspeed_config: Optional[Dict] = None,
        output_dir: str = "./checkpoints",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 1000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        logging_first_step: bool = True,
        save_total_limit: int = 5,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        ignore_data_skip: bool = False,
        gate_temperature_schedule: Optional[Dict] = None,
        expert_load_balancing: bool = True,
    ):
        """
        Initialize MoE trainer.
        
        Args:
            model: MoE model to train
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            optimizer: Optimizer (if None, will be created by DeepSpeed)
            lr_scheduler: Learning rate scheduler
            config: Training configuration
            deepspeed_config: DeepSpeed configuration
            output_dir: Directory to save checkpoints
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_steps: Maximum training steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Warmup steps for learning rate
            logging_first_step: Whether to log first step
            save_total_limit: Maximum number of checkpoints to keep
            load_best_model_at_end: Load best model at end of training
            metric_for_best_model: Metric to use for best model selection
            greater_is_better: Whether higher metric values are better
            ignore_data_skip: Ignore data skip in resuming
            gate_temperature_schedule: Schedule for gate temperature
            expert_load_balancing: Enable expert load balancing
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        
        # Training parameters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.logging_first_step = logging_first_step
        self.save_total_limit = save_total_limit
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.ignore_data_skip = ignore_data_skip
        self.gate_temperature_schedule = gate_temperature_schedule or {}
        self.expert_load_balancing = expert_load_balancing
        
        # Initialize DeepSpeed
        self.deepspeed_config = deepspeed_config
        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.best_model_checkpoint = None
        
        # Memory tracking
        self.memory_tracker = MemoryTracker()
        
        # Evaluation
        if eval_dataloader:
            self.evaluator = ModelEvaluator(model=model, tokenizer=train_dataloader.dataset.tokenizer)
        else:
            self.evaluator = None
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        
        # Expert utilization tracking
        self.expert_stats = defaultdict(list)
        
        logger.info(f"Trainer initialized with max_steps={max_steps}, output_dir={output_dir}")
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed engine."""
        logger.info("Initializing DeepSpeed...")
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=self.deepspeed_config,
            model_parameters=self.model.parameters(),
        )
        
        logger.info(f"DeepSpeed initialized with config: {self.deepspeed_config}")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        if is_main_process():
            expert_stats = self.model.get_expert_stats()
            logger.info(f"Expert configuration: {expert_stats}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup DeepSpeed
        self._setup_deepspeed()
        
        # Training loop
        self.model_engine.train()
        train_iterator = iter(self.train_dataloader)
        
        start_time = time.time()
        
        while self.global_step < self.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reset iterator for new epoch
                self.epoch += 1
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
                logger.info(f"Starting epoch {self.epoch}")
            
            # Training step
            loss_dict = self._training_step(batch)
            
            # Update global step
            self.global_step += 1
            
            # Logging
            if self.global_step % self.logging_steps == 0 or (self.logging_first_step and self.global_step == 1):
                self._log_training_metrics(loss_dict, start_time)
            
            # Evaluation
            if self.eval_dataloader and self.global_step % self.eval_steps == 0:
                eval_metrics = self._evaluate()
                self._log_eval_metrics(eval_metrics)
                
                # Save best model
                if self._is_better_metric(eval_metrics.get(self.metric_for_best_model, 0)):
                    self.best_metric = eval_metrics[self.metric_for_best_model]
                    self.best_model_checkpoint = self.global_step
                    self._save_checkpoint(is_best=True)
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self._save_checkpoint()
            
            # Update gate temperature
            if self.gate_temperature_schedule:
                self._update_gate_temperature()
            
            # Memory cleanup
            if self.global_step % 100 == 0:
                torch.cuda.empty_cache()
        
        logger.info("Training completed!")
        
        # Load best model if requested
        if self.load_best_model_at_end and self.best_model_checkpoint:
            self._load_best_model()
        
        # Final evaluation
        if self.eval_dataloader:
            final_metrics = self._evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(self.model_engine.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model_engine(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            return_expert_metrics=self.expert_load_balancing,
        )
        
        loss = outputs['loss']
        
        # Backward pass
        self.model_engine.backward(loss)
        
        # Gradient clipping and optimization step
        if self.global_step % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.model_engine.step()
        
        # Collect metrics
        loss_dict = {
            'loss': loss.item(),
            'lm_loss': outputs.get('lm_loss', loss).item(),
            'load_balance_loss': outputs.get('load_balance_loss', 0.0).item() if isinstance(outputs.get('load_balance_loss', 0.0), torch.Tensor) else outputs.get('load_balance_loss', 0.0),
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else self.model_engine.optimizer.param_groups[0]['lr'],
        }
        
        # Expert metrics
        if 'expert_metrics' in outputs:
            expert_metrics = outputs['expert_metrics']
            if 'expert_utilization' in expert_metrics:
                utilization = expert_metrics['expert_utilization'].cpu().numpy()
                loss_dict['expert_utilization_mean'] = utilization.mean()
                loss_dict['expert_utilization_std'] = utilization.std()
                loss_dict['expert_utilization_min'] = utilization.min()
                loss_dict['expert_utilization_max'] = utilization.max()
        
        return loss_dict
    
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        logger.info("Running evaluation...")
        
        if not self.evaluator:
            return {}
        
        self.model_engine.eval()
        eval_metrics = {}
        
        with torch.no_grad():
            # Language modeling evaluation
            lm_metrics = self.evaluator.evaluate_language_modeling(self.eval_dataloader)
            eval_metrics.update({f"eval_{k}": v for k, v in lm_metrics.items()})
            
            # Perplexity calculation
            if 'loss' in lm_metrics:
                eval_metrics['eval_perplexity'] = math.exp(lm_metrics['loss'])
            
            # Expert utilization metrics
            expert_metrics = self.evaluator.analyze_expert_utilization(self.eval_dataloader)
            eval_metrics.update({f"eval_expert_{k}": v for k, v in expert_metrics.items()})
        
        self.model_engine.train()
        return eval_metrics
    
    def _log_training_metrics(self, loss_dict: Dict[str, float], start_time: float):
        """Log training metrics."""
        elapsed_time = time.time() - start_time
        steps_per_second = self.global_step / elapsed_time
        
        metrics = {
            'step': self.global_step,
            'epoch': self.epoch,
            'steps_per_second': steps_per_second,
            **loss_dict
        }
        
        # Memory stats
        if torch.cuda.is_available():
            memory_stats = self.memory_tracker.get_memory_stats()
            metrics.update(memory_stats)
        
        # Store metrics
        for key, value in metrics.items():
            self.train_metrics[key].append(value)
        
        # Log to console
        if is_main_process():
            log_message = f"Step {self.global_step}: " + ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
            logger.info(log_message)
            
            # Log to tensorboard/mlflow
            log_metrics(metrics, step=self.global_step, prefix="train")
    
    def _log_eval_metrics(self, eval_metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if not eval_metrics:
            return
        
        # Store metrics
        for key, value in eval_metrics.items():
            self.eval_metrics[key].append(value)
        
        if is_main_process():
            log_message = f"Eval Step {self.global_step}: " + ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
            logger.info(log_message)
            
            # Log to tensorboard/mlflow
            log_metrics(eval_metrics, step=self.global_step, prefix="eval")
    
    def _is_better_metric(self, metric_value: float) -> bool:
        """Check if metric value is better than current best."""
        if self.greater_is_better:
            return metric_value > self.best_metric
        else:
            return metric_value < self.best_metric
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if not is_main_process():
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using DeepSpeed
        self.model_engine.save_checkpoint(str(checkpoint_dir))
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'best_model_checkpoint': self.best_model_checkpoint,
            'train_metrics': dict(self.train_metrics),
            'eval_metrics': dict(self.eval_metrics),
            'config': self.config,
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        if is_best:
            # Create symlink to best model
            best_link = self.output_dir / "best_model"
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(checkpoint_dir.name)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to stay within save_total_limit."""
        checkpoints = [d for d in self.output_dir.iterdir() if d.name.startswith("checkpoint-")]
        
        if len(checkpoints) <= self.save_total_limit:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.save_total_limit]:
            if checkpoint.name != f"checkpoint-{self.best_model_checkpoint}":
                try:
                    import shutil
                    shutil.rmtree(checkpoint)
                    logger.info(f"Removed old checkpoint: {checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
    
    def _load_best_model(self):
        """Load the best model checkpoint."""
        if not self.best_model_checkpoint:
            logger.warning("No best model checkpoint found")
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{self.best_model_checkpoint}"
        if not checkpoint_dir.exists():
            logger.warning(f"Best model checkpoint not found: {checkpoint_dir}")
            return
        
        logger.info(f"Loading best model from {checkpoint_dir}")
        
        # Load checkpoint using DeepSpeed
        _, client_states = self.model_engine.load_checkpoint(str(checkpoint_dir))
        
        logger.info(f"Best model loaded (step {self.best_model_checkpoint})")
    
    def _update_gate_temperature(self):
        """Update gate temperature according to schedule."""
        if not self.gate_temperature_schedule:
            return
        
        schedule_type = self.gate_temperature_schedule.get('type', 'linear')
        start_temp = self.gate_temperature_schedule.get('start', 1.0)
        end_temp = self.gate_temperature_schedule.get('end', 0.5)
        
        if schedule_type == 'linear':
            # Linear decay
            progress = self.global_step / self.max_steps
            current_temp = start_temp + (end_temp - start_temp) * progress
        elif schedule_type == 'exponential':
            # Exponential decay
            decay_rate = self.gate_temperature_schedule.get('decay_rate', 0.95)
            current_temp = start_temp * (decay_rate ** (self.global_step / 1000))
            current_temp = max(current_temp, end_temp)
        else:
            current_temp = start_temp
        
        # Update temperature in all transformer layers
        for layer in self.model.layers:
            layer.set_gate_temperature(current_temp)
        
        if self.global_step % self.logging_steps == 0:
            logger.debug(f"Gate temperature updated to {current_temp:.4f}")
    
    def save_model(self, output_dir: str):
        """Save the final trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if is_main_process():
            # Save model state dict
            model_state = self.model.state_dict()
            torch.save(model_state, output_path / "pytorch_model.bin")
            
            # Save model config
            model_config = {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
                'n_experts': self.model.n_experts,
                'expert_capacity': self.model.expert_capacity,
                'top_k': self.model.top_k,
                'mla_interval': self.model.mla_interval,
                'mla_latent_dim': self.model.mla_latent_dim,
                'max_seq_len': self.model.max_seq_len,
            }
            
            torch.save(model_config, output_path / "config.json")
            
            logger.info(f"Model saved to {output_path}")
    
    def resume_from_checkpoint(self, checkpoint_dir: str):
        """Resume training from a checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
        
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Load DeepSpeed checkpoint
        _, client_states = self.model_engine.load_checkpoint(str(checkpoint_path))
        
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path)
            
            self.global_step = training_state['global_step']
            self.epoch = training_state['epoch']
            self.best_metric = training_state.get('best_metric', self.best_metric)
            self.best_model_checkpoint = training_state.get('best_model_checkpoint')
            
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.warning("Training state not found, starting fresh")
