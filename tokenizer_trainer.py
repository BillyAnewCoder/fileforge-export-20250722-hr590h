"""
SentencePiece tokenizer training for the MoE model.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Iterator
import logging
import json

import sentencepiece as spm


logger = logging.getLogger(__name__)


class SentencePieceTrainer:
    """
    Trains SentencePiece tokenizer on diverse text corpora.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        normalization_rule_name: str = "nmt_nfkc_cf",
        max_sentence_length: int = 8192,
        shuffle_input_sentence: bool = True,
        input_sentence_size: int = 10000000,
        num_threads: int = 16,
    ):
        """
        Initialize SentencePiece trainer.
        
        Args:
            vocab_size: Target vocabulary size
            model_type: Model type ("bpe", "unigram", "char", "word")
            character_coverage: Character coverage for vocabulary
            normalization_rule_name: Text normalization rules
            max_sentence_length: Maximum sentence length to consider
            shuffle_input_sentence: Whether to shuffle input sentences
            input_sentence_size: Maximum number of sentences to use for training
            num_threads: Number of threads for training
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.normalization_rule_name = normalization_rule_name
        self.max_sentence_length = max_sentence_length
        self.shuffle_input_sentence = shuffle_input_sentence
        self.input_sentence_size = input_sentence_size
        self.num_threads = num_threads
        
        # Special tokens
        self.special_tokens = [
            "<pad>",      # Padding token
            "<unk>",      # Unknown token
            "<s>",        # Start of sequence
            "</s>",       # End of sequence
            "<mask>",     # Mask token for MLM
            "<sep>",      # Separator token
            "<cls>",      # Classification token
            "<code>",     # Code block marker
            "<dialogue>", # Dialogue marker
            "<math>",     # Math expression marker
        ]
    
    def prepare_training_data(
        self,
        input_dirs: List[str],
        output_file: str,
        max_lines: Optional[int] = None,
        min_length: int = 10,
        max_length: int = 8192,
    ) -> Dict[str, int]:
        """
        Prepare training data from processed shards.
        
        Args:
            input_dirs: List of directories containing processed shards
            output_file: Output file for training data
            max_lines: Maximum number of lines to extract
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Statistics about extracted data
        """
        logger.info(f"Preparing training data from {len(input_dirs)} directories")
        
        lines_written = 0
        total_chars = 0
        file_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for input_dir in input_dirs:
                input_path = Path(input_dir)
                
                # Find all shard files
                shard_files = list(input_path.glob("shard_*.jsonl*"))
                logger.info(f"Found {len(shard_files)} shard files in {input_dir}")
                
                for shard_file in shard_files:
                    file_count += 1
                    logger.info(f"Processing {shard_file}")
                    
                    # Read shard
                    if shard_file.suffix == '.gz':
                        import gzip
                        file_opener = gzip.open
                    else:
                        file_opener = open
                    
                    with file_opener(shard_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            if max_lines and lines_written >= max_lines:
                                break
                                
                            try:
                                doc = json.loads(line)
                                text = doc.get('text', '').strip()
                                
                                # Length filtering
                                if len(text) < min_length or len(text) > max_length:
                                    continue
                                
                                # Write text (one sentence per line for better tokenizer training)
                                sentences = self._split_into_sentences(text)
                                for sentence in sentences:
                                    if len(sentence.strip()) >= min_length:
                                        outf.write(sentence.strip() + '\n')
                                        lines_written += 1
                                        total_chars += len(sentence)
                                        
                                        if max_lines and lines_written >= max_lines:
                                            break
                                            
                            except json.JSONDecodeError:
                                continue
                    
                    if max_lines and lines_written >= max_lines:
                        break
                
                if max_lines and lines_written >= max_lines:
                    break
        
        stats = {
            "files_processed": file_count,
            "lines_written": lines_written,
            "total_characters": total_chars,
            "avg_line_length": total_chars / max(lines_written, 1),
        }
        
        logger.info(f"Training data preparation complete: {stats}")
        return stats
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better tokenizer training.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with proper sentence segmentation)
        import re
        
        # Split on sentence boundaries
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        
        # Also split on line breaks for code and structured text
        all_sentences = []
        for sentence in sentences:
            if '\n' in sentence:
                all_sentences.extend(sentence.split('\n'))
            else:
                all_sentences.append(sentence)
        
        # Filter and clean
        cleaned_sentences = []
        for sentence in all_sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def train(
        self,
        input_file: str,
        model_prefix: str,
        vocab_size: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Train SentencePiece model.
        
        Args:
            input_file: Training data file
            model_prefix: Output model prefix
            vocab_size: Override vocabulary size
            
        Returns:
            Dictionary with paths to trained model files
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        logger.info(f"Training SentencePiece model with vocab_size={vocab_size}")
        
        # Prepare training arguments
        training_args = [
            f"--input={input_file}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={self.model_type}",
            f"--character_coverage={self.character_coverage}",
            f"--normalization_rule_name={self.normalization_rule_name}",
            f"--max_sentence_length={self.max_sentence_length}",
            f"--shuffle_input_sentence={self.shuffle_input_sentence}",
            f"--input_sentence_size={self.input_sentence_size}",
            f"--num_threads={self.num_threads}",
            "--split_digits=true",
            "--allow_whitespace_only_pieces=true",
            "--remove_extra_whitespaces=false",
            "--hard_vocab_limit=false",
        ]
        
        # Add special tokens
        if self.special_tokens:
            user_defined_symbols = ",".join(self.special_tokens)
            training_args.append(f"--user_defined_symbols={user_defined_symbols}")
        
        # Add control symbols for different content types
        control_symbols = [
            "<|code|>", "<|/code|>",          # Code blocks
            "<|math|>", "<|/math|>",          # Math expressions
            "<|dialogue|>", "<|/dialogue|>",  # Dialogue
            "<|document|>", "<|/document|>",  # Document boundaries
        ]
        control_symbols_str = ",".join(control_symbols)
        training_args.append(f"--control_symbols={control_symbols_str}")
        
        # Train model
        command = " ".join(training_args)
        logger.info(f"Training command: spm_train {command}")
        
        try:
            spm.SentencePieceTrainer.train(command)
            logger.info("SentencePiece training completed successfully")
            
            # Verify model files exist
            model_file = f"{model_prefix}.model"
            vocab_file = f"{model_prefix}.vocab"
            
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not created: {model_file}")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"Vocab file not created: {vocab_file}")
            
            return {
                "model_file": model_file,
                "vocab_file": vocab_file,
            }
            
        except Exception as e:
            logger.error(f"SentencePiece training failed: {e}")
            raise
    
    def test_tokenizer(self, model_file: str, test_texts: Optional[List[str]] = None) -> Dict:
        """
        Test the trained tokenizer on sample texts.
        
        Args:
            model_file: Path to trained model file
            test_texts: Optional list of test texts
            
        Returns:
            Test results and statistics
        """
        logger.info(f"Testing tokenizer: {model_file}")
        
        # Load tokenizer
        sp = smp.SentencePieceProcessor()
        sp.load(model_file)
        
        # Default test texts if none provided
        if test_texts is None:
            test_texts = [
                "This is a simple English sentence.",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "User: Hello, how are you? Assistant: I'm doing well, thank you for asking!",
                "The quick brown fox jumps over the lazy dog. 1234567890",
                "import torch; import torch.nn as nn; model = nn.Linear(768, 1024)",
            ]
        
        results = {}
        total_tokens = 0
        total_chars = 0
        
        for i, text in enumerate(test_texts):
            # Tokenize
            tokens = sp.encode(text, out_type=str)
            token_ids = sp.encode(text, out_type=int)
            
            # Detokenize
            reconstructed = sp.decode(token_ids)
            
            result = {
                "original": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "reconstructed": reconstructed,
                "num_tokens": len(tokens),
                "compression_ratio": len(text) / len(tokens),
                "perfect_reconstruction": text == reconstructed,
            }
            
            results[f"test_{i}"] = result
            total_tokens += len(tokens)
            total_chars += len(text)
        
        # Overall statistics
        results["overall"] = {
            "vocab_size": sp.vocab_size(),
            "avg_compression_ratio": total_chars / total_tokens,
            "total_test_cases": len(test_texts),
            "pad_id": sp.pad_id(),
            "unk_id": sp.unk_id(),
            "bos_id": sp.bos_id(),
            "eos_id": sp.eos_id(),
        }
        
        logger.info(f"Tokenizer test complete. Average compression ratio: {results['overall']['avg_compression_ratio']:.2f}")
        return results
    
    def save_tokenizer_config(
        self,
        model_file: str,
        config_file: str,
        additional_config: Optional[Dict] = None,
    ):
        """
        Save tokenizer configuration for easy loading.
        
        Args:
            model_file: Path to SentencePiece model file
            config_file: Path to save configuration JSON
            additional_config: Additional configuration parameters
        """
        sp = smp.SentencePieceProcessor()
        sp.load(model_file)
        
        config = {
            "model_file": model_file,
            "vocab_size": sp.vocab_size(),
            "model_type": self.model_type,
            "special_tokens": {
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "mask_token": "<mask>",
                "sep_token": "<sep>",
                "cls_token": "<cls>",
            },
            "special_token_ids": {
                "pad_id": sp.pad_id(),
                "unk_id": sp.unk_id(),
                "bos_id": sp.bos_id(),
                "eos_id": sp.eos_id(),
            },
            "training_params": {
                "character_coverage": self.character_coverage,
                "normalization_rule_name": self.normalization_rule_name,
                "max_sentence_length": self.max_sentence_length,
            }
        }
        
        if additional_config:
            config.update(additional_config)
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Tokenizer configuration saved to {config_file}")


class TokenizerWrapper:
    """
    Wrapper class for easy tokenizer usage in training and inference.
    """
    
    def __init__(self, model_file: str, config_file: Optional[str] = None):
        self.sp = smp.SentencePieceProcessor()
        self.sp.load(model_file)
        
        # Load configuration if available
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        token_ids = self.sp.encode(text, out_type=int)
        
        if add_special_tokens:
            # Add BOS token
            if self.sp.bos_id() != -1:
                token_ids = [self.sp.bos_id()] + token_ids
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        return self.sp.decode(token_ids)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, padding: bool = True) -> Dict[str, List]:
        """Encode batch of texts with optional padding."""
        batch_token_ids = []
        
        for text in texts:
            token_ids = self.encode(text)
            
            # Truncate if necessary
            if max_length and len(token_ids) > max_length:
                token_ids = token_ids[:max_length-1] + [self.sp.eos_id()]
            
            batch_token_ids.append(token_ids)
        
        # Pad sequences if requested
        if padding:
            max_len = max(len(seq) for seq in batch_token_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            padded_ids = []
            attention_masks = []
            
            for token_ids in batch_token_ids:
                # Truncate if necessary
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                
                # Create attention mask
                attention_mask = [1] * len(token_ids)
                
                # Pad sequence
                while len(token_ids) < max_len:
                    token_ids.append(self.sp.pad_id())
                    attention_mask.append(0)
                
                padded_ids.append(token_ids)
                attention_masks.append(attention_mask)
            
            return {
                "input_ids": padded_ids,
                "attention_mask": attention_masks,
            }
        else:
            return {"input_ids": batch_token_ids}
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.sp.pad_id()
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.sp.unk_id()
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.sp.bos_id()
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.sp.eos_id()
