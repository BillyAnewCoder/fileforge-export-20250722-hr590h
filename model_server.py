"""
Model server for efficient MoE model inference.
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
import numpy as np

from models.moe_model import MoETransformerModel
from data.tokenizer_trainer import TokenizerWrapper
from utils.memory_utils import MemoryTracker

logger = logging.getLogger(__name__)


class ModelServer:
    """
    High-performance model server for MoE inference.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: torch.device,
        max_batch_size: int = 8,
        max_sequence_length: int = 2048,
        dtype: torch.dtype = torch.float16,
        compile_model: bool = False,
    ):
        """
        Initialize model server.
        
        Args:
            model_path: Path to model files
            tokenizer_path: Path to tokenizer files  
            device: Device to load model on
            max_batch_size: Maximum batch size for inference
            max_sequence_length: Maximum sequence length
            dtype: Model data type
            compile_model: Whether to compile model with torch.compile
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = dtype
        self.compile_model = compile_model
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        # Performance tracking
        self.memory_tracker = MemoryTracker()
        self.inference_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'batch_sizes': [],
            'sequence_lengths': [],
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model_lock = threading.Lock()
        
        # Load model and tokenizer
        self._load_model()
        self._load_tokenizer()
    
    def _load_model(self):
        """Load the MoE model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load model configuration
            config_path = self.model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
            else:
                raise FileNotFoundError(f"Model config not found: {config_path}")
            
            # Initialize model
            self.model = MoETransformerModel(**self.model_config)
            
            # Load state dict
            model_state_path = self.model_path / "pytorch_model.bin"
            if model_state_path.exists():
                state_dict = torch.load(model_state_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Model weights not found: {model_state_path}")
            
            # Move to device and set precision
            self.model.to(device=self.device, dtype=self.dtype)
            self.model.eval()
            
            # Compile model if requested
            if self.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Log model stats
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model size: {total_params:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer from {self.tokenizer_path}")
        
        try:
            # Look for tokenizer files
            tokenizer_model_path = None
            tokenizer_config_path = None
            
            # Check common tokenizer file patterns
            for pattern in ["*.model", "tokenizer.model", "sp.model"]:
                matches = list(self.tokenizer_path.glob(pattern))
                if matches:
                    tokenizer_model_path = matches[0]
                    break
            
            for pattern in ["tokenizer_config.json", "tokenizer.json"]:
                config_file = self.tokenizer_path / pattern
                if config_file.exists():
                    tokenizer_config_path = config_file
                    break
            
            if tokenizer_model_path is None:
                raise FileNotFoundError(f"Tokenizer model not found in {self.tokenizer_path}")
            
            # Load tokenizer
            self.tokenizer = TokenizerWrapper(
                model_file=str(tokenizer_model_path),
                config_file=str(tokenizer_config_path) if tokenizer_config_path else None,
            )
            
            logger.info(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model and tokenizer are loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model_config.copy() if self.model_config else {}
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get expert statistics."""
        if self.model is None:
            return {}
        
        return self.model.get_expert_stats()
    
    def preprocess_inputs(
        self,
        prompts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess input prompts for batch inference.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum sequence length
            
        Returns:
            Batch of tokenized inputs
        """
        if max_length is None:
            max_length = self.max_sequence_length
        
        # Tokenize prompts
        batch_data = self.tokenizer.encode_batch(
            prompts,
            max_length=max_length,
            padding=True,
        )
        
        # Convert to tensors and move to device
        input_ids = torch.tensor(batch_data["input_ids"], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(batch_data["attention_mask"], device=self.device, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def postprocess_outputs(
        self,
        generated_ids: torch.Tensor,
        input_lengths: List[int],
        include_prompt: bool = False,
        stop_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Postprocess generated outputs.
        
        Args:
            generated_ids: Generated token IDs
            input_lengths: Length of input prompts
            include_prompt: Whether to include prompt in output
            stop_tokens: Stop tokens to truncate at
            
        Returns:
            List of generated text strings
        """
        results = []
        
        for i, (seq, input_len) in enumerate(zip(generated_ids, input_lengths)):
            if include_prompt:
                tokens_to_decode = seq
            else:
                tokens_to_decode = seq[input_len:]
            
            # Decode tokens
            generated_text = self.tokenizer.decode(
                tokens_to_decode.cpu().tolist(),
                skip_special_tokens=True,
            )
            
            # Apply stop tokens
            if stop_tokens:
                for stop_token in stop_tokens:
                    if stop_token in generated_text:
                        generated_text = generated_text.split(stop_token)[0]
                        break
            
            results.append(generated_text.strip())
        
        return results
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_tokens: Optional[List[str]] = None,
        include_prompt: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate text from prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences per prompt
            stop_tokens: Stop tokens for early termination
            include_prompt: Whether to include prompt in output
            
        Returns:
            List of generation results
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        if len(prompts) > self.max_batch_size:
            raise ValueError(f"Batch size {len(prompts)} exceeds maximum {self.max_batch_size}")
        
        start_time = time.time()
        
        with self.model_lock:
            # Preprocess inputs
            batch_inputs = self.preprocess_inputs(prompts, max_length)
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            
            batch_size, seq_len = input_ids.shape
            input_lengths = attention_mask.sum(dim=1).cpu().tolist()
            
            # Generate for each sequence in batch
            all_results = []
            
            for i in range(batch_size):
                prompt_input_ids = input_ids[i:i+1, :input_lengths[i]]
                prompt_results = []
                
                # Generate multiple sequences for this prompt
                for _ in range(num_return_sequences):
                    generated_ids = self._generate_sequence(
                        input_ids=prompt_input_ids,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=do_sample,
                    )
                    
                    # Postprocess
                    generated_text = self.postprocess_outputs(
                        generated_ids,
                        [input_lengths[i]],
                        include_prompt=include_prompt,
                        stop_tokens=stop_tokens,
                    )[0]
                    
                    prompt_results.append(generated_text)
                
                generation_time = time.time() - start_time
                
                all_results.append({
                    "generated_text": prompt_results,
                    "generation_time": generation_time,
                    "input_length": input_lengths[i],
                })
        
        # Update stats
        total_time = time.time() - start_time
        self._update_stats(len(prompts), input_lengths, total_time)
        
        return all_results
    
    def _generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        do_sample: bool,
    ) -> torch.Tensor:
        """Generate a single sequence."""
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.model(input_ids=generated)
            logits = outputs["logits"]
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                    min_top_k = top_k_logits[:, -1:].expand_as(next_token_logits)
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove.unsqueeze(1), float('-inf'))
                
                # Sample from filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Check for EOS token
            if next_tokens.item() == self.tokenizer.eos_token_id:
                break
            
            # Check sequence length limit
            if generated.size(1) >= self.max_sequence_length:
                break
        
        return generated
    
    async def generate_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Async wrapper for generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompts,
            **kwargs
        )
    
    async def load_model_async(self):
        """Async model loading."""
        if not self.is_loaded():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)
            await loop.run_in_executor(self.executor, self._load_tokenizer)
    
    async def reload_model(self):
        """Reload the model."""
        logger.info("Reloading model...")
        
        with self.model_lock:
            # Clear existing model
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            # Reload
            await self.load_model_async()
        
        logger.info("Model reloaded successfully")
    
    def _update_stats(self, batch_size: int, input_lengths: List[int], total_time: float):
        """Update inference statistics."""
        self.inference_stats['total_requests'] += batch_size
        self.inference_stats['total_tokens'] += sum(input_lengths)
        self.inference_stats['total_time'] += total_time
        self.inference_stats['batch_sizes'].append(batch_size)
        self.inference_stats['sequence_lengths'].extend(input_lengths)
        
        # Keep only recent stats (last 1000 requests)
        if len(self.inference_stats['batch_sizes']) > 1000:
            self.inference_stats['batch_sizes'] = self.inference_stats['batch_sizes'][-1000:]
        if len(self.inference_stats['sequence_lengths']) > 1000:
            self.inference_stats['sequence_lengths'] = self.inference_stats['sequence_lengths'][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        stats = self.inference_stats.copy()
        
        if stats['total_time'] > 0:
            stats['avg_requests_per_second'] = stats['total_requests'] / stats['total_time']
            stats['avg_tokens_per_second'] = stats['total_tokens'] / stats['total_time']
        else:
            stats['avg_requests_per_second'] = 0.0
            stats['avg_tokens_per_second'] = 0.0
        
        if stats['batch_sizes']:
            stats['avg_batch_size'] = np.mean(stats['batch_sizes'])
            stats['avg_sequence_length'] = np.mean(stats['sequence_lengths'])
        else:
            stats['avg_batch_size'] = 0.0
            stats['avg_sequence_length'] = 0.0
        
        # Memory stats
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up model server...")
        
        with self.model_lock:
            if self.model is not None:
                del self.model
                self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.executor.shutdown(wait=True)
        logger.info("Cleanup completed")


class BatchedModelServer(ModelServer):
    """
    Enhanced model server with dynamic batching capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        self.batch_timeout = kwargs.pop('batch_timeout', 0.1)  # 100ms timeout
        self.pending_requests = []
        self.batch_lock = asyncio.Lock()
        super().__init__(*args, **kwargs)
    
    async def generate_with_batching(
        self,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with dynamic batching.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generation result
        """
        # Create request
        request = {
            'prompt': prompt,
            'kwargs': kwargs,
            'future': asyncio.Future(),
            'timestamp': time.time(),
        }
        
        async with self.batch_lock:
            self.pending_requests.append(request)
            
            # Check if we should process batch now
            if (len(self.pending_requests) >= self.max_batch_size or
                time.time() - self.pending_requests[0]['timestamp'] > self.batch_timeout):
                await self._process_batch()
        
        # Wait for result
        return await request['future']
    
    async def _process_batch(self):
        """Process accumulated batch of requests."""
        if not self.pending_requests:
            return
        
        # Extract requests
        batch_requests = self.pending_requests.copy()
        self.pending_requests.clear()
        
        try:
            # Prepare batch
            prompts = [req['prompt'] for req in batch_requests]
            
            # Use kwargs from first request (assume same for batch)
            kwargs = batch_requests[0]['kwargs']
            
            # Generate
            results = await self.generate_async(prompts, **kwargs)
            
            # Return results to futures
            for request, result in zip(batch_requests, results):
                request['future'].set_result(result)
                
        except Exception as e:
            # Set exception for all futures
            for request in batch_requests:
                request['future'].set_exception(e)
    
    async def start_batch_processor(self):
        """Start background batch processor."""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            async with self.batch_lock:
                if (self.pending_requests and
                    time.time() - self.pending_requests[0]['timestamp'] > self.batch_timeout):
                    await self._process_batch()
