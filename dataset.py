"""
Dataset implementation for MoE model training with efficient data loading.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

logger = logging.getLogger(__name__)


class MoEDataset(IterableDataset):
    """
    Iterable dataset for MoE training that supports streaming from sharded files.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        min_length: int = 10,
        multi_token_prediction: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 10000,
    ):
        """
        Initialize MoE dataset.
        
        Args:
            data_dir: Directory containing processed shards
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            multi_token_prediction: Number of future tokens to predict
            shuffle: Whether to shuffle data
            seed: Random seed
            buffer_size: Buffer size for shuffling
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.multi_token_prediction = multi_token_prediction
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        
        # Find all shard files
        self.shard_files = self._find_shard_files()
        if not self.shard_files:
            raise ValueError(f"No shard files found in {data_dir}")
        
        logger.info(f"Found {len(self.shard_files)} shard files in {data_dir}")
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        
        # Initialize random state
        self.rng = random.Random(seed)
    
    def _find_shard_files(self) -> List[Path]:
        """Find all shard files in the data directory."""
        shard_files = []
        
        # Look for JSONL files (with or without gzip compression)
        patterns = ["shard_*.jsonl", "shard_*.jsonl.gz"]
        
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            shard_files.extend(files)
        
        return sorted(shard_files)
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata if available."""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _read_shard(self, shard_file: Path) -> Iterator[Dict]:
        """Read documents from a shard file."""
        if shard_file.suffix == '.gz':
            import gzip
            file_opener = gzip.open
        else:
            file_opener = open
        
        try:
            with file_opener(shard_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        doc = json.loads(line)
                        yield doc
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error in {shard_file}:{line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading shard {shard_file}: {e}")
    
    def _tokenize_document(self, doc: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenize a document and prepare for training.
        
        Args:
            doc: Document dictionary with 'text' field
            
        Returns:
            Tokenized data or None if filtered out
        """
        text = doc.get('text', '').strip()
        if len(text) < self.min_length:
            return None
        
        # Tokenize text
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Filter by length
        if len(token_ids) < self.min_length or len(token_ids) > self.max_length:
            return None
        
        # Prepare input and target sequences for multi-token prediction
        if self.multi_token_prediction > 1:
            # For multi-token prediction, we predict N future tokens at each position
            input_ids = token_ids[:-self.multi_token_prediction]
            labels = []
            
            for i in range(len(input_ids)):
                # Collect next N tokens as labels
                future_tokens = token_ids[i+1:i+1+self.multi_token_prediction]
                
                # Pad if necessary
                while len(future_tokens) < self.multi_token_prediction:
                    future_tokens.append(self.tokenizer.pad_token_id)
                
                labels.append(future_tokens)
            
            if not labels:
                return None
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            }
        else:
            # Standard next-token prediction
            input_ids = token_ids[:-1]
            labels = token_ids[1:]
            
            if len(input_ids) == 0:
                return None
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized documents."""
        shard_files = self.shard_files.copy()
        
        if self.shuffle:
            self.rng.shuffle(shard_files)
        
        buffer = []
        
        for shard_file in shard_files:
            logger.debug(f"Reading shard: {shard_file}")
            
            for doc in self._read_shard(shard_file):
                tokenized = self._tokenize_document(doc)
                
                if tokenized is not None:
                    if self.shuffle:
                        # Add to shuffle buffer
                        buffer.append(tokenized)
                        
                        if len(buffer) >= self.buffer_size:
                            # Shuffle and yield from buffer
                            self.rng.shuffle(buffer)
                            while buffer:
                                yield buffer.pop()
                    else:
                        yield tokenized
        
        # Yield remaining items in buffer
        if self.shuffle and buffer:
            self.rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()


class MoEDataLoader:
    """
    Data loader wrapper with distributed training support.
    """
    
    def __init__(
        self,
        dataset: MoEDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        
        # Create data loader
        self.dataloader = self._create_dataloader()
    
    def _create_dataloader(self) -> DataLoader:
        """Create PyTorch DataLoader with appropriate settings."""
        
        def collate_fn(batch):
            """Custom collate function for variable-length sequences."""
            if not batch:
                return {}
            
            # Pad sequences to the same length within batch
            max_len = max(item['input_ids'].size(0) for item in batch)
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for item in batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                labels = item['labels']
                
                # Pad input_ids and attention_mask
                pad_len = max_len - input_ids.size(0)
                if pad_len > 0:
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((pad_len,), self.dataset.tokenizer.pad_token_id, dtype=torch.long)
                    ])
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
                
                # Pad labels (shape depends on multi-token prediction)
                if labels.dim() == 1:
                    # Standard next-token prediction
                    if pad_len > 0:
                        labels = torch.cat([
                            labels,
                            torch.full((pad_len,), -100, dtype=torch.long)  # -100 is ignored in loss
                        ])
                else:
                    # Multi-token prediction
                    if pad_len > 0:
                        pad_labels = torch.full(
                            (pad_len, labels.size(1)), -100, dtype=torch.long
                        )
                        labels = torch.cat([labels, pad_labels], dim=0)
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            
            return {
                'input_ids': torch.stack(batch_input_ids),
                'attention_mask': torch.stack(batch_attention_mask),
                'labels': torch.stack(batch_labels),
            }
        
        # For distributed training, we manually handle data distribution
        # since IterableDataset doesn't work well with DistributedSampler
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,  # Important for consistent batch sizes across workers
        )
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get approximate number of batches."""
        if hasattr(self.dataset.metadata, 'total_documents'):
            total_docs = self.dataset.metadata['total_documents']
            return total_docs // (self.batch_size * self.world_size)
        return 1000  # Fallback estimate


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation tasks (language modeling, code generation, QA).
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        task_type: str = "language_modeling",
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize evaluation dataset.
        
        Args:
            data_path: Path to evaluation data
            tokenizer: Tokenizer for text encoding
            task_type: Type of evaluation task
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to load
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.max_length = max_length
        self.max_samples = max_samples
        
        # Load evaluation data
        self.samples = self._load_evaluation_data()
        
        logger.info(f"Loaded {len(self.samples)} samples for {task_type} evaluation")
    
    def _load_evaluation_data(self) -> List[Dict]:
        """Load evaluation data from file."""
        samples = []
        
        if not self.data_path.exists():
            logger.warning(f"Evaluation data file not found: {self.data_path}")
            return samples
        
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if self.max_samples and len(samples) >= self.max_samples:
                        break
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
        
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:self.max_samples] if self.max_samples else data
                else:
                    samples = [data]
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single evaluation sample."""
        sample = self.samples[idx]
        
        if self.task_type == "language_modeling":
            return self._prepare_lm_sample(sample)
        elif self.task_type == "code_generation":
            return self._prepare_code_sample(sample)
        elif self.task_type == "qa":
            return self._prepare_qa_sample(sample)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _prepare_lm_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Prepare language modeling sample."""
        text = sample.get('text', '')
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # Prepare input and target
        input_ids = token_ids[:-1] if len(token_ids) > 1 else [self.tokenizer.bos_token_id]
        labels = token_ids[1:] if len(token_ids) > 1 else [self.tokenizer.eos_token_id]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'text': text,
        }
    
    def _prepare_code_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Prepare code generation sample."""
        prompt = sample.get('prompt', '')
        completion = sample.get('completion', '')
        
        # Tokenize prompt and completion separately
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine
        full_ids = prompt_ids + completion_ids
        
        # Truncate if necessary
        if len(full_ids) > self.max_length:
            # Prioritize keeping the prompt
            if len(prompt_ids) < self.max_length:
                completion_ids = completion_ids[:self.max_length - len(prompt_ids)]
                full_ids = prompt_ids + completion_ids
            else:
                full_ids = prompt_ids[:self.max_length]
        
        # Labels: -100 for prompt tokens, actual tokens for completion
        input_ids = full_ids[:-1] if len(full_ids) > 1 else [self.tokenizer.bos_token_id]
        labels = [-100] * (len(prompt_ids) - 1) + full_ids[len(prompt_ids):]
        
        # Ensure labels and input_ids have same length
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'prompt': prompt,
            'completion': completion,
        }
    
    def _prepare_qa_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Prepare QA sample."""
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        context = sample.get('context', '')
        
        # Format as: Context: ... Question: ... Answer: ...
        if context:
            text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Find where the answer starts for labeling
        answer_start_text = f"\nAnswer: {answer}"
        answer_start_ids = self.tokenizer.encode(answer_start_text, add_special_tokens=False)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        input_ids = token_ids[:-1] if len(token_ids) > 1 else [self.tokenizer.bos_token_id]
        labels = token_ids[1:] if len(token_ids) > 1 else [self.tokenizer.eos_token_id]
        
        # Mask labels before answer (simple heuristic)
        answer_pattern = self.tokenizer.encode("Answer:", add_special_tokens=False)
        if len(answer_pattern) > 0:
            answer_token = answer_pattern[0]
            try:
                answer_start_idx = labels.index(answer_token)
                labels = [-100] * answer_start_idx + labels[answer_start_idx:]
            except ValueError:
                pass  # Answer token not found, use all labels
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'question': question,
            'answer': answer,
            'context': context,
        }


def create_dataloaders(
    train_data_dir: str,
    eval_data_path: Optional[str],
    tokenizer,
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    max_length: int = 2048,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs
) -> Dict[str, Union[MoEDataLoader, DataLoader]]:
    """
    Create training and evaluation data loaders.
    
    Returns:
        Dictionary with 'train' and optionally 'eval' data loaders
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    dataloaders = {}
    
    # Training data loader
    train_dataset = MoEDataset(
        data_dir=train_data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        **dataset_kwargs
    )
    
    train_dataloader = MoEDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    
    dataloaders['train'] = train_dataloader
    
    # Evaluation data loader (optional)
    if eval_data_path:
        eval_dataset = EvaluationDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            }
        )
        
        dataloaders['eval'] = eval_dataloader
    
    return dataloaders
