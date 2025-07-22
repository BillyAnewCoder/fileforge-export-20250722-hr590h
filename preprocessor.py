"""
Data preprocessing pipeline for text, code, and dialogue corpora.
Handles cleaning, normalization, deduplication, and sharding.
"""

import os
import json
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Set
import logging
from collections import defaultdict
import re

import torch
from datasets import Dataset, load_dataset
import sentencepiece as spm


logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing for diverse corpora.
    """
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 8192,
        min_words: int = 5,
        dedupe_threshold: float = 0.8,
        language_filters: Optional[List[str]] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.dedupe_threshold = dedupe_threshold
        self.language_filters = language_filters or ['en']
        
        # Deduplication storage
        self.seen_hashes: Set[str] = set()
        self.content_hashes: Dict[str, str] = {}
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Compiled regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.whitespace_pattern = re.compile(r'\s+')
        self.unicode_control_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')
    
    def clean_text(self, text: str, text_type: str = "general") -> Optional[str]:
        """
        Clean and normalize text based on type.
        
        Args:
            text: Raw text to clean
            text_type: Type of text ("general", "code", "dialogue")
            
        Returns:
            Cleaned text or None if should be filtered out
        """
        if not isinstance(text, str):
            self.stats['invalid_type'] += 1
            return None
        
        original_length = len(text)
        
        # Basic length filtering
        if original_length < self.min_length or original_length > self.max_length:
            self.stats['length_filtered'] += 1
            return None
        
        if text_type == "code":
            return self._clean_code(text)
        elif text_type == "dialogue":
            return self._clean_dialogue(text)
        else:
            return self._clean_general_text(text)
    
    def _clean_general_text(self, text: str) -> Optional[str]:
        """Clean general text content."""
        # Remove control characters
        text = self.unicode_control_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove excessive URLs and emails (keep some for context)
        url_count = len(self.url_pattern.findall(text))
        email_count = len(self.email_pattern.findall(text))
        
        if url_count > 5 or email_count > 3:
            self.stats['spam_filtered'] += 1
            return None
        
        # Basic quality checks
        words = text.split()
        if len(words) < self.min_words:
            self.stats['too_few_words'] += 1
            return None
        
        # Check for repetitive content
        if self._is_repetitive(text):
            self.stats['repetitive_filtered'] += 1
            return None
        
        # Language detection (simplified)
        if not self._is_target_language(text):
            self.stats['language_filtered'] += 1
            return None
        
        self.stats['general_text_cleaned'] += 1
        return text.strip()
    
    def _clean_code(self, text: str) -> Optional[str]:
        """Clean code content."""
        # Remove excessive comments
        lines = text.split('\n')
        comment_ratio = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*'))) / max(len(lines), 1)
        
        if comment_ratio > 0.7:
            self.stats['too_many_comments'] += 1
            return None
        
        # Check for minimum code structure
        if not any(keyword in text.lower() for keyword in ['def ', 'class ', 'function', 'import', 'from ', 'return']):
            if len(text.split('\n')) > 5:  # Only check structure for multi-line code
                self.stats['no_code_structure'] += 1
                return None
        
        # Remove excessive empty lines
        cleaned_lines = []
        empty_count = 0
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                empty_count = 0
            else:
                empty_count += 1
                if empty_count <= 2:  # Allow max 2 consecutive empty lines
                    cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        self.stats['code_cleaned'] += 1
        return cleaned_text
    
    def _clean_dialogue(self, text: str) -> Optional[str]:
        """Clean dialogue/conversation content."""
        # Split into turns/messages
        turns = []
        
        # Simple dialogue parsing (assumes speaker: message format)
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line and len(line.split(':', 1)) == 2:
                speaker, message = line.split(':', 1)
                speaker = speaker.strip()
                message = message.strip()
                
                if len(message) > 5 and len(speaker) < 50:
                    turns.append(f"{speaker}: {message}")
        
        if len(turns) < 2:
            self.stats['insufficient_dialogue'] += 1
            return None
        
        cleaned_text = '\n'.join(turns)
        self.stats['dialogue_cleaned'] += 1
        return cleaned_text
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text is overly repetitive."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        # If any single word makes up more than 30% of the text
        max_word_ratio = max(word_counts.values()) / len(words)
        return max_word_ratio > 0.3
    
    def _is_target_language(self, text: str) -> bool:
        """Simple language detection (can be enhanced with proper language detection libraries)."""
        if 'en' not in self.language_filters:
            return True  # Skip language filtering if not filtering for English
        
        # Simple heuristic: check for common English words
        common_english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        
        words = set(text.lower().split())
        english_word_count = len(words & common_english_words)
        total_words = len(words)
        
        if total_words == 0:
            return False
        
        english_ratio = english_word_count / total_words
        return english_ratio > 0.1  # At least 10% common English words
    
    def compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = self.compute_hash(text)
        
        if text_hash in self.seen_hashes:
            self.stats['duplicates_filtered'] += 1
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process a batch of documents."""
        processed = []
        
        for doc in batch:
            text = doc.get('text', '')
            text_type = doc.get('type', 'general')
            
            # Clean text
            cleaned_text = self.clean_text(text, text_type)
            if cleaned_text is None:
                continue
            
            # Check for duplicates
            if self.is_duplicate(cleaned_text):
                continue
            
            # Add processed document
            processed_doc = {
                'text': cleaned_text,
                'type': text_type,
                'length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
            }
            
            # Preserve metadata
            for key, value in doc.items():
                if key not in processed_doc:
                    processed_doc[key] = value
            
            processed.append(processed_doc)
            self.stats['processed'] += 1
        
        return processed
    
    def get_stats(self) -> Dict[str, int]:
        """Get preprocessing statistics."""
        return dict(self.stats)


class DataShardWriter:
    """
    Writes processed data to sharded files in JSONL or TFRecord format.
    """
    
    def __init__(
        self,
        output_dir: str,
        shard_size: int = 100000,
        format: str = "jsonl",
        compression: Optional[str] = "gzip",
    ):
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.format = format
        self.compression = compression
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current shard state
        self.current_shard = 0
        self.current_shard_size = 0
        self.current_file = None
        
        # Statistics
        self.total_documents = 0
        self.total_shards = 0
    
    def _get_shard_filename(self, shard_id: int) -> str:
        """Get filename for shard."""
        base_name = f"shard_{shard_id:06d}.{self.format}"
        if self.compression == "gzip":
            base_name += ".gz"
        return base_name
    
    def _open_new_shard(self):
        """Open a new shard file for writing."""
        if self.current_file is not None:
            self.current_file.close()
        
        filename = self._get_shard_filename(self.current_shard)
        filepath = self.output_dir / filename
        
        if self.compression == "gzip":
            import gzip
            self.current_file = gzip.open(filepath, 'wt', encoding='utf-8')
        else:
            self.current_file = open(filepath, 'w', encoding='utf-8')
        
        self.current_shard_size = 0
        logger.info(f"Opened new shard: {filepath}")
    
    def write_documents(self, documents: List[Dict]):
        """Write documents to sharded files."""
        for doc in documents:
            # Check if we need a new shard
            if self.current_file is None or self.current_shard_size >= self.shard_size:
                if self.current_file is not None:
                    self.total_shards += 1
                    self.current_shard += 1
                self._open_new_shard()
            
            # Write document
            if self.format == "jsonl":
                json.dump(doc, self.current_file, ensure_ascii=False)
                self.current_file.write('\n')
            else:
                raise ValueError(f"Unsupported format: {self.format}")
            
            self.current_shard_size += 1
            self.total_documents += 1
    
    def close(self):
        """Close current shard and finalize writing."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.total_shards += 1
        
        # Write metadata
        metadata = {
            "total_documents": self.total_documents,
            "total_shards": self.total_shards,
            "shard_size": self.shard_size,
            "format": self.format,
            "compression": self.compression,
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Finished writing {self.total_documents} documents to {self.total_shards} shards")


def process_dataset(
    input_path: str,
    output_dir: str,
    dataset_type: str = "general",
    num_workers: int = 4,
    shard_size: int = 100000,
    **preprocessor_kwargs
) -> Dict[str, int]:
    """
    Process a complete dataset with multiprocessing.
    
    Args:
        input_path: Path to input dataset
        output_dir: Output directory for processed shards
        dataset_type: Type of dataset ("general", "code", "dialogue")
        num_workers: Number of worker processes
        shard_size: Documents per shard
        **preprocessor_kwargs: Additional arguments for TextPreprocessor
        
    Returns:
        Processing statistics
    """
    # Initialize preprocessor and writer
    preprocessor = TextPreprocessor(**preprocessor_kwargs)
    writer = DataShardWriter(output_dir, shard_size=shard_size)
    
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    
    try:
        # Try loading as HuggingFace dataset
        if os.path.isdir(input_path) or input_path.startswith("https://"):
            dataset = load_dataset(input_path, split="train")
        else:
            # Load from file
            if input_path.endswith('.jsonl'):
                dataset = load_dataset("json", data_files=input_path, split="train")
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    logger.info(f"Loaded {len(dataset)} documents")
    
    # Process in batches
    batch_size = 1000
    total_processed = 0
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # Add dataset type to each document
        batch_docs = []
        for doc in batch:
            doc_dict = dict(doc)
            doc_dict['type'] = dataset_type
            batch_docs.append(doc_dict)
        
        # Process batch
        processed_docs = preprocessor.process_batch(batch_docs)
        
        # Write to shards
        if processed_docs:
            writer.write_documents(processed_docs)
            total_processed += len(processed_docs)
        
        # Log progress
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {i}/{len(dataset)} batches, {total_processed} documents written")
    
    # Finalize
    writer.close()
    stats = preprocessor.get_stats()
    stats['total_input_documents'] = len(dataset)
    stats['total_output_documents'] = total_processed
    
    logger.info(f"Processing complete. Stats: {stats}")
    return stats


def merge_datasets(
    input_dirs: List[str],
    output_dir: str,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Merge multiple processed datasets into a single unified dataset.
    
    Args:
        input_dirs: List of directories containing processed shards
        output_dir: Output directory for merged dataset
        shuffle: Whether to shuffle the merged data
        seed: Random seed for shuffling
        
    Returns:
        Merge statistics
    """
    import random
    
    random.seed(seed)
    
    # Collect all shard files
    all_shards = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        shard_files = list(input_path.glob("shard_*.jsonl*"))
        all_shards.extend(shard_files)
    
    if shuffle:
        random.shuffle(all_shards)
    
    logger.info(f"Found {len(all_shards)} shards to merge")
    
    # Initialize output writer
    writer = DataShardWriter(output_dir, shard_size=100000)
    
    total_docs = 0
    
    # Process each shard
    for shard_file in all_shards:
        logger.info(f"Processing shard: {shard_file}")
        
        # Read shard
        if shard_file.suffix == '.gz':
            import gzip
            file_opener = gzip.open
        else:
            file_opener = open
        
        with file_opener(shard_file, 'rt', encoding='utf-8') as f:
            documents = []
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    documents.append(doc)
                    
                    # Write in batches
                    if len(documents) >= 1000:
                        writer.write_documents(documents)
                        total_docs += len(documents)
                        documents = []
            
            # Write remaining documents
            if documents:
                writer.write_documents(documents)
                total_docs += len(documents)
    
    writer.close()
    
    stats = {
        "input_shards": len(all_shards),
        "total_documents": total_docs,
        "output_shards": writer.total_shards,
    }
    
    logger.info(f"Merge complete. Stats: {stats}")
    return stats
