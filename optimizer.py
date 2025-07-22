"""
Optimizers and learning rate schedulers for MoE training.
"""

import math
import warnings
from typing import Dict, Any, Optional, Union, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class AdamWWithGateRegularization(torch.optim.AdamW):
    """
    AdamW optimizer with additional regularization for MoE gating networks.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        gate_regularization: float = 0.0,
        gate_noise_decay: float = 0.999,
    ):
        """
        Initialize AdamW with gate regularization.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Betas for Adam
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use AMSGrad variant
            gate_regularization: Gate weight regularization strength
            gate_noise_decay: Decay rate for gate noise
        """
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.gate_regularization = gate_regularization
        self.gate_noise_decay = gate_noise_decay
    
    def step(self, closure=None):
        """Perform optimization step with gate regularization."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply gate regularization if parameter name contains 'gate'
                if hasattr(p, 'param_name') and 'gate' in p.param_name:
                    if self.gate_regularization > 0:
                        # Add L2 regularization on gate weights
                        p.grad = p.grad + self.gate_regularization * p.data
                
                # Apply standard AdamW update
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class LayerWiseAdamW(Optimizer):
    """
    Layer-wise adaptive learning rates for large models.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        layer_decay: float = 0.65,
    ):
        """
        Initialize LayerWise AdamW.
        
        Args:
            params: Model parameters with layer information
            lr: Base learning rate
            betas: Betas for Adam
            eps: Epsilon for numerical stability  
            weight_decay: Weight decay coefficient
            layer_decay: Layer-wise decay factor
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= layer_decay <= 1.0:
            raise ValueError(f"Invalid layer_decay value: {layer_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, layer_decay=layer_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform optimization step with layer-wise learning rates."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get layer-specific learning rate
                layer_id = group.get('layer_id', 0)
                layer_lr = group['lr'] * (group['layer_decay'] ** layer_id)
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = layer_lr / bias_correction1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * layer_lr)
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class WarmupCosineLRWithRestart(_LRScheduler):
    """
    Cosine learning rate with warmup and restarts.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        restart_periods: Optional[list] = None,
        restart_multiplier: float = 2.0,
        last_epoch: int = -1,
    ):
        """
        Initialize warmup cosine scheduler with restarts.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum LR as ratio of max LR
            restart_periods: List of restart periods
            restart_multiplier: Multiplier for restart periods
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_periods = restart_periods or []
        self.restart_multiplier = restart_multiplier
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get current learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        
        # Check for restarts
        current_step = self.last_epoch
        for restart_step in self.restart_periods:
            if current_step >= restart_step:
                current_step = current_step % restart_step
                break
        
        # Cosine decay
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
            for base_lr in self.base_lrs
        ]


class LinearWarmupPolynomialDecay(_LRScheduler):
    """
    Linear warmup followed by polynomial decay.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.power = power
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        
        # Polynomial decay
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        decay_factor = (1 - progress) ** self.power
        
        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor)
            for base_lr in self.base_lrs
        ]


class ExpertSpecificOptimizer:
    """
    Wrapper for applying different optimization strategies to different expert groups.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer_class=AdamWWithGateRegularization,
        base_optimizer_kwargs=None,
        expert_optimizer_kwargs=None,
        gate_optimizer_kwargs=None,
    ):
        """
        Initialize expert-specific optimizer.
        
        Args:
            model: Model to optimize
            base_optimizer_class: Base optimizer class
            base_optimizer_kwargs: Base optimizer arguments
            expert_optimizer_kwargs: Expert-specific optimizer arguments
            gate_optimizer_kwargs: Gate-specific optimizer arguments
        """
        base_optimizer_kwargs = base_optimizer_kwargs or {}
        expert_optimizer_kwargs = expert_optimizer_kwargs or {}
        gate_optimizer_kwargs = gate_optimizer_kwargs or {}
        
        # Separate parameters by type
        base_params = []
        expert_params = []
        gate_params = []
        
        for name, param in model.named_parameters():
            if 'expert' in name.lower():
                expert_params.append(param)
            elif 'gate' in name.lower() or 'gating' in name.lower():
                param.param_name = name  # Store name for gate regularization
                gate_params.append(param)
            else:
                base_params.append(param)
        
        # Create optimizers
        self.optimizers = {}
        
        if base_params:
            self.optimizers['base'] = base_optimizer_class(base_params, **base_optimizer_kwargs)
        
        if expert_params:
            expert_kwargs = {**base_optimizer_kwargs, **expert_optimizer_kwargs}
            self.optimizers['expert'] = base_optimizer_class(expert_params, **expert_kwargs)
        
        if gate_params:
            gate_kwargs = {**base_optimizer_kwargs, **gate_optimizer_kwargs}
            self.optimizers['gate'] = base_optimizer_class(gate_params, **gate_kwargs)
    
    def step(self, closure=None):
        """Perform optimization step for all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.step(closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none)
    
    def state_dict(self):
        """Get state dict for all optimizers."""
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}
    
    def load_state_dict(self, state_dict):
        """Load state dict for all optimizers."""
        for name, opt in self.optimizers.items():
            if name in state_dict:
                opt.load_state_dict(state_dict[name])


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
    use_expert_specific: bool = True,
    gate_regularization: float = 0.01,
    layer_wise_decay: bool = False,
    layer_decay_rate: float = 0.65,
    **kwargs
) -> Union[Optimizer, ExpertSpecificOptimizer]:
    """
    Create optimizer for MoE model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adamw', 'layerwise_adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Beta parameters for Adam
        eps: Epsilon for numerical stability
        use_expert_specific: Use expert-specific optimization
        gate_regularization: Gate regularization strength
        layer_wise_decay: Use layer-wise learning rate decay
        layer_decay_rate: Layer decay rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    base_kwargs = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
        'betas': betas,
        'eps': eps,
        **kwargs
    }
    
    if optimizer_type == "adamw":
        optimizer_class = AdamWWithGateRegularization
        base_kwargs['gate_regularization'] = gate_regularization
    elif optimizer_type == "layerwise_adamw":
        optimizer_class = LayerWiseAdamW
        base_kwargs['layer_decay'] = layer_decay_rate
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    if use_expert_specific:
        # Expert-specific optimization
        expert_kwargs = {'lr': learning_rate * 0.5}  # Lower LR for experts
        gate_kwargs = {'gate_regularization': gate_regularization * 2}  # Higher regularization for gates
        
        return ExpertSpecificOptimizer(
            model=model,
            base_optimizer_class=optimizer_class,
            base_optimizer_kwargs=base_kwargs,
            expert_optimizer_kwargs=expert_kwargs,
            gate_optimizer_kwargs=gate_kwargs,
        )
    else:
        # Standard optimization
        return optimizer_class(model.parameters(), **base_kwargs)


def create_scheduler(
    optimizer: Union[Optimizer, ExpertSpecificOptimizer],
    scheduler_type: str = "cosine_warmup",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    **kwargs
) -> Union[_LRScheduler, Dict[str, _LRScheduler]]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        warmup_steps: Warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR ratio
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler(s)
    """
    scheduler_kwargs = {
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'min_lr_ratio': min_lr_ratio,
        **kwargs
    }
    
    if scheduler_type == "cosine_warmup":
        scheduler_class = WarmupCosineLRWithRestart
    elif scheduler_type == "polynomial":
        scheduler_class = LinearWarmupPolynomialDecay
        scheduler_kwargs['power'] = kwargs.get('power', 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    if isinstance(optimizer, ExpertSpecificOptimizer):
        # Create schedulers for each optimizer
        schedulers = {}
        for name, opt in optimizer.optimizers.items():
            schedulers[name] = scheduler_class(opt, **scheduler_kwargs)
        return schedulers
    else:
        return scheduler_class(optimizer, **scheduler_kwargs)


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    layer_decay: float = 0.65,
    skip_list: Optional[list] = None,
) -> list:
    """
    Get parameter groups with layer-wise decay and weight decay rules.
    
    Args:
        model: Model to get parameter groups from
        weight_decay: Weight decay value
        layer_decay: Layer-wise decay rate
        skip_list: Parameters to skip weight decay
        
    Returns:
        List of parameter groups
    """
    skip_list = skip_list or []
    parameter_groups = []
    
    # Get layer information
    layer_scales = {}
    for name, _ in model.named_parameters():
        layer_id = 0
        if 'layers.' in name:
            try:
                layer_id = int(name.split('layers.')[1].split('.')[0])
            except (ValueError, IndexError):
                layer_id = 0
        
        layer_scales[name] = layer_decay ** layer_id
    
    # Group parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if should skip weight decay
        skip_decay = any(skip_name in name for skip_name in skip_list)
        skip_decay = skip_decay or len(param.shape) == 1  # Skip bias and layer norm
        
        group_config = {
            'params': [param],
            'layer_id': layer_scales.get(name, 1.0),
            'param_name': name,
        }
        
        if skip_decay:
            group_config['weight_decay'] = 0.0
            no_decay_params.append(group_config)
        else:
            group_config['weight_decay'] = weight_decay
            decay_params.append(group_config)
    
    parameter_groups.extend(decay_params)
    parameter_groups.extend(no_decay_params)
    
    return parameter_groups
