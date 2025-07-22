#!/usr/bin/env python3
"""
Main entry point for the Mixture-of-Experts (MoE) Transformer training system.
Provides unified command-line interface for training, evaluation, data preprocessing,
model export, and serving operations.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Mixture-of-Experts Transformer with Multi-Head Latent Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --data_dir ./data/processed --output_dir ./checkpoints --tokenizer_path ./tokenizer

  # Evaluate a trained model
  python main.py evaluate --model_path ./checkpoints/final --data_dir ./data/eval --output_dir ./eval_results

  # Preprocess training data
  python main.py preprocess --input_dir ./data/raw --output_dir ./data/processed --train_tokenizer

  # Export model for inference
  python main.py export --model_path ./checkpoints/final --output_path ./exported_model --format torchscript

  # Start inference server
  python main.py serve --model_path ./exported_model --port 5000

For detailed help on each command, use:
  python main.py <command> --help
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--debug", action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log_dir", type=str, 
        help="Directory for log files"
    )
    parser.add_argument(
        "--config_dir", type=str, default="configs",
        help="Directory containing configuration files"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Training command
    train_parser = subparsers.add_parser(
        "train", 
        help="Train MoE transformer model",
        description="Train a Mixture-of-Experts transformer model with distributed training support"
    )
    add_train_arguments(train_parser)
    
    # Evaluation command
    eval_parser = subparsers.add_parser(
        "evaluate", 
        help="Evaluate trained model",
        description="Comprehensive evaluation of trained MoE models across multiple tasks"
    )
    add_evaluate_arguments(eval_parser)
    
    # Data preprocessing command
    preprocess_parser = subparsers.add_parser(
        "preprocess", 
        help="Preprocess training data",
        description="Process and prepare data for MoE model training, including tokenizer training"
    )
    add_preprocess_arguments(preprocess_parser)
    
    # Model export command
    export_parser = subparsers.add_parser(
        "export", 
        help="Export model for inference",
        description="Export trained models to various formats for optimized inference"
    )
    add_export_arguments(export_parser)
    
    # Serving command
    serve_parser = subparsers.add_parser(
        "serve", 
        help="Start inference server",
        description="Start FastAPI server for model inference"
    )
    add_serve_arguments(serve_parser)
    
    return parser


def add_train_arguments(parser: argparse.ArgumentParser):
    """Add training-specific arguments."""
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--data_dir", type=str, required=True, help="Training data directory")
    required.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    required.add_argument("--tokenizer_path", type=str, required=True, help="Tokenizer path")
    
    # Model configuration
    model_group = parser.add_argument_group("model configuration")
    model_group.add_argument("--model_config", type=str, default="base", help="Model configuration name or file")
    model_group.add_argument("--num_experts", type=int, help="Number of experts")
    model_group.add_argument("--top_k", type=int, help="Top-k expert routing")
    model_group.add_argument("--d_model", type=int, help="Model dimension")
    model_group.add_argument("--n_layers", type=int, help="Number of transformer layers")
    model_group.add_argument("--n_heads", type=int, help="Number of attention heads")
    
    # Training configuration
    train_group = parser.add_argument_group("training configuration")
    train_group.add_argument("--training_config", type=str, default="base", help="Training configuration name or file")
    train_group.add_argument("--max_steps", type=int, help="Maximum training steps")
    train_group.add_argument("--batch_size", type=int, help="Training batch size")
    train_group.add_argument("--learning_rate", type=float, help="Learning rate")
    train_group.add_argument("--warmup_steps", type=int, help="Warmup steps")
    
    # Data configuration
    data_group = parser.add_argument_group("data configuration")
    data_group.add_argument("--eval_data_dir", type=str, help="Evaluation data directory")
    data_group.add_argument("--max_seq_len", type=int, help="Maximum sequence length")
    
    # DeepSpeed configuration
    ds_group = parser.add_argument_group("deepspeed configuration")
    ds_group.add_argument("--deepspeed_config", type=str, help="DeepSpeed configuration file")
    ds_group.add_argument("--zero_stage", type=int, choices=[1, 2, 3], help="ZeRO optimization stage")
    ds_group.add_argument("--cpu_offload", action="store_true", help="Enable CPU offload")
    ds_group.add_argument("--fp16", action="store_true", help="Enable FP16 training")
    ds_group.add_argument("--bf16", action="store_true", help="Enable BF16 training")
    
    # Distributed training
    dist_group = parser.add_argument_group("distributed training")
    dist_group.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Logging and evaluation
    log_group = parser.add_argument_group("logging and evaluation")
    log_group.add_argument("--logging_steps", type=int, help="Logging frequency")
    log_group.add_argument("--eval_steps", type=int, help="Evaluation frequency")
    log_group.add_argument("--save_steps", type=int, help="Save frequency")
    log_group.add_argument("--experiment_name", type=str, help="Experiment name")
    log_group.add_argument("--use_tensorboard", action="store_true", help="Use TensorBoard logging")
    log_group.add_argument("--use_mlflow", action="store_true", help="Use MLflow tracking")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")


def add_evaluate_arguments(parser: argparse.ArgumentParser):
    """Add evaluation-specific arguments."""
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    required.add_argument("--data_dir", type=str, required=True, help="Evaluation data directory")
    required.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    # Model and data
    model_group = parser.add_argument_group("model and data")
    model_group.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (defaults to model_path)")
    model_group.add_argument("--config_path", type=str, help="Path to model config")
    
    # Evaluation configuration
    eval_group = parser.add_argument_group("evaluation configuration")
    eval_group.add_argument("--eval_tasks", type=str, nargs="+", 
                           default=["language_modeling", "generation_quality", "expert_analysis"],
                           help="Evaluation tasks to run")
    eval_group.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    eval_group.add_argument("--max_samples", type=int, help="Maximum samples per task")
    eval_group.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Generation parameters
    gen_group = parser.add_argument_group("generation parameters")
    gen_group.add_argument("--generation_max_length", type=int, default=100, help="Max generation length")
    gen_group.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    gen_group.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    gen_group.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate")
    
    # Device and precision
    device_group = parser.add_argument_group("device and precision")
    device_group.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    device_group.add_argument("--fp16", action="store_true", help="Use FP16 for evaluation")
    device_group.add_argument("--bf16", action="store_true", help="Use BF16 for evaluation")
    
    # Analysis options
    analysis_group = parser.add_argument_group("analysis options")
    analysis_group.add_argument("--analyze_experts", action="store_true", help="Perform detailed expert analysis")
    analysis_group.add_argument("--benchmark_speed", action="store_true", help="Benchmark inference speed")
    analysis_group.add_argument("--benchmark_memory", action="store_true", help="Benchmark memory usage")
    analysis_group.add_argument("--save_predictions", action="store_true", help="Save model predictions")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")


def add_preprocess_arguments(parser: argparse.ArgumentParser):
    """Add preprocessing-specific arguments."""
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--input_dir", type=str, required=True, help="Input data directory")
    required.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    
    # Input configuration
    input_group = parser.add_argument_group("input configuration")
    input_group.add_argument("--input_files", type=str, nargs="+", help="Specific input files to process")
    input_group.add_argument("--data_type", type=str, default="general", 
                            choices=["general", "code", "dialogue"], help="Type of data being processed")
    
    # Processing parameters
    process_group = parser.add_argument_group("processing parameters")
    process_group.add_argument("--min_length", type=int, default=10, help="Minimum text length")
    process_group.add_argument("--max_length", type=int, default=8192, help="Maximum text length")
    process_group.add_argument("--min_words", type=int, default=5, help="Minimum number of words")
    process_group.add_argument("--dedupe_threshold", type=float, default=0.8, help="Deduplication threshold")
    process_group.add_argument("--language_filters", type=str, nargs="+", default=["en"], help="Language filters")
    
    # Sharding configuration
    shard_group = parser.add_argument_group("sharding configuration")
    shard_group.add_argument("--shard_size", type=int, default=100000, help="Documents per shard")
    shard_group.add_argument("--compression", type=str, choices=["none", "gzip"], default="gzip", help="Shard compression")
    shard_group.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    
    # Tokenizer training
    tokenizer_group = parser.add_argument_group("tokenizer training")
    tokenizer_group.add_argument("--train_tokenizer", action="store_true", help="Train new tokenizer")
    tokenizer_group.add_argument("--tokenizer_output", type=str, help="Output path for trained tokenizer")
    tokenizer_group.add_argument("--vocab_size", type=int, default=32000, help="Tokenizer vocabulary size")
    tokenizer_group.add_argument("--tokenizer_model", type=str, default="bpe", 
                                choices=["bpe", "unigram", "char", "word"], help="Tokenizer model type")
    tokenizer_group.add_argument("--character_coverage", type=float, default=0.9995, help="Character coverage")
    tokenizer_group.add_argument("--tokenizer_samples", type=int, default=10000000, help="Max samples for tokenizer training")
    
    # Multiple dataset processing
    multi_group = parser.add_argument_group("multiple dataset processing")
    multi_group.add_argument("--merge_datasets", action="store_true", help="Merge multiple datasets")
    multi_group.add_argument("--input_dirs", type=str, nargs="+", help="Multiple input directories to merge")
    multi_group.add_argument("--shuffle_merged", action="store_true", help="Shuffle merged data")
    multi_group.add_argument("--merge_seed", type=int, default=42, help="Random seed for merging")
    
    # Debug options
    parser.add_argument("--dry_run", action="store_true", help="Dry run - don't write files")


def add_export_arguments(parser: argparse.ArgumentParser):
    """Add export-specific arguments."""
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    required.add_argument("--output_path", type=str, required=True, help="Output path for exported model")
    
    # Export configuration
    export_group = parser.add_argument_group("export configuration")
    export_group.add_argument("--format", type=str, default="torchscript", 
                             choices=["torchscript", "deepspeed", "onnx", "tensorrt", "pytorch"],
                             help="Export format")
    export_group.add_argument("--checkpoint_path", type=str, help="Specific checkpoint path (for DeepSpeed models)")
    
    # Optimization options
    opt_group = parser.add_argument_group("optimization options")
    opt_group.add_argument("--optimize", action="store_true", help="Apply optimizations for inference")
    opt_group.add_argument("--quantize", action="store_true", help="Apply quantization")
    opt_group.add_argument("--quantization_bits", type=int, default=8, choices=[4, 8, 16], help="Quantization bits")
    opt_group.add_argument("--fp16", action="store_true", help="Export in FP16 precision")
    opt_group.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")
    
    # Model modifications
    mod_group = parser.add_argument_group("model modifications")
    mod_group.add_argument("--merge_experts", action="store_true", help="Merge experts for faster inference")
    mod_group.add_argument("--prune_experts", type=float, help="Prune experts below utilization threshold")
    
    # Batch processing
    batch_group = parser.add_argument_group("batch processing")
    batch_group.add_argument("--batch_size", type=int, default=1, help="Batch size for export tracing")
    batch_group.add_argument("--max_seq_len", type=int, help="Maximum sequence length for export")
    
    # Device and validation
    device_group = parser.add_argument_group("device and validation")
    device_group.add_argument("--device", type=str, default="cpu", help="Device for export")
    device_group.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for large models")
    device_group.add_argument("--validate", action="store_true", help="Validate exported model")
    device_group.add_argument("--benchmark", action="store_true", help="Benchmark exported model")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")


def add_serve_arguments(parser: argparse.ArgumentParser):
    """Add serving-specific arguments."""
    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--model_path", type=str, required=True, help="Path to model for serving")
    
    # Server configuration
    server_group = parser.add_argument_group("server configuration")
    server_group.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server")
    server_group.add_argument("--port", type=int, default=5000, help="Port to bind server")
    server_group.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    # Model configuration
    model_group = parser.add_argument_group("model configuration")
    model_group.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (defaults to model_path)")
    model_group.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    model_group.add_argument("--fp16", action="store_true", help="Use FP16 for serving")
    model_group.add_argument("--bf16", action="store_true", help="Use BF16 for serving")
    
    # Inference configuration
    inference_group = parser.add_argument_group("inference configuration")
    inference_group.add_argument("--max_batch_size", type=int, default=8, help="Maximum batch size")
    inference_group.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    inference_group.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    
    # Generation defaults
    gen_group = parser.add_argument_group("generation defaults")
    gen_group.add_argument("--default_max_length", type=int, default=100, help="Default generation max length")
    gen_group.add_argument("--default_temperature", type=float, default=1.0, help="Default temperature")
    gen_group.add_argument("--default_top_p", type=float, default=0.9, help="Default top-p")
    gen_group.add_argument("--default_top_k", type=int, default=50, help="Default top-k")
    
    # Performance options
    perf_group = parser.add_argument_group("performance options")
    perf_group.add_argument("--compile_model", action="store_true", help="Compile model for faster inference")
    perf_group.add_argument("--enable_batching", action="store_true", help="Enable request batching")
    perf_group.add_argument("--batch_timeout", type=float, default=0.1, help="Batch timeout in seconds")


def run_command(args):
    """Execute the specified command."""
    # Setup logging
    setup_logging(
        log_level="DEBUG" if args.debug else "INFO",
        log_dir=args.log_dir,
        include_timestamp=True,
    )
    
    try:
        if args.command == "train":
            from scripts.train import main as train_main
            train_main(args)
        
        elif args.command == "evaluate":
            from scripts.evaluate import main as evaluate_main
            evaluate_main(args)
        
        elif args.command == "preprocess":
            from scripts.preprocess_data import main as preprocess_main
            preprocess_main(args)
        
        elif args.command == "export":
            from scripts.export_model import main as export_main
            export_main(args)
        
        elif args.command == "serve":
            from scripts.start_server import main as serve_main
            serve_main(args)
        
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Command failed: {str(e)}", exc_info=args.debug)
        return 1
    
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not hasattr(args, 'command') or args.command is None:
        parser.print_help()
        return 0
    
    # Run the command
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
