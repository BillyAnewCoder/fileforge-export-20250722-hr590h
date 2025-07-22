"""
FastAPI-based inference server for MoE model serving.
"""

import asyncio
import time
import logging
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from models.moe_model import MoETransformerModel
from data.tokenizer_trainer import TokenizerWrapper
from serving.model_server import ModelServer
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_length: int = Field(100, ge=1, le=2048, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling threshold")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, ge=1, le=10, description="Number of sequences to return")
    stop_tokens: Optional[List[str]] = Field(None, description="Stop tokens for generation")
    include_prompt: bool = Field(False, description="Whether to include prompt in response")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: List[str] = Field(..., description="Generated text sequences")
    generation_time: float = Field(..., description="Generation time in seconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    tokens_per_second: float = Field(..., description="Generation speed")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    prompts: List[str] = Field(..., description="List of input prompts")
    max_length: int = Field(100, ge=1, le=2048, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling threshold")
    do_sample: bool = Field(True, description="Whether to use sampling")


class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation."""
    results: List[GenerationResponse] = Field(..., description="Generation results for each prompt")
    total_time: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Batch size")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    uptime: float = Field(..., description="Server uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_config: Dict[str, Any] = Field(..., description="Model configuration")
    expert_stats: Dict[str, Any] = Field(..., description="Expert statistics")
    vocab_size: int = Field(..., description="Vocabulary size")
    model_size: str = Field(..., description="Model size description")


class InferenceServer:
    """
    FastAPI-based inference server for MoE models.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "auto",
        max_batch_size: int = 8,
        model_parallel: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize inference server.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            device: Device to load model on
            max_batch_size: Maximum batch size for inference
            model_parallel: Whether to use model parallelism
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.max_batch_size = max_batch_size
        self.model_parallel = model_parallel
        self.load_in_8bit = load_in_8bit
        
        # Initialize device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model server
        self.model_server = ModelServer(
            model_path=str(self.model_path),
            tokenizer_path=str(self.tokenizer_path),
            device=self.device,
            max_batch_size=max_batch_size,
        )
        
        # Server state
        self.start_time = time.time()
        self.request_count = 0
        self.total_tokens_generated = 0
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="MoE Model Inference Server",
            description="High-performance inference server for Mixture-of-Experts models",
            version="1.0.0",
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes to FastAPI app."""
        
        @app.on_event("startup")
        async def startup_event():
            """Initialize model on startup."""
            logger.info("Starting up inference server...")
            try:
                await self.model_server.load_model_async()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            logger.info("Shutting down inference server...")
            await self.model_server.cleanup()
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            memory_usage = {}
            
            if torch.cuda.is_available():
                memory_usage["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_usage["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB
            
            import psutil
            process = psutil.Process()
            memory_usage["cpu_memory_mb"] = process.memory_info().rss / 1024**2  # MB
            
            return HealthResponse(
                status="healthy",
                model_loaded=self.model_server.is_loaded(),
                gpu_available=torch.cuda.is_available(),
                memory_usage=memory_usage,
                uptime=time.time() - self.start_time,
            )
        
        @app.get("/model/info", response_model=ModelInfoResponse)
        async def model_info():
            """Get model information."""
            if not self.model_server.is_loaded():
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            model_config = self.model_server.get_model_config()
            expert_stats = self.model_server.get_expert_stats()
            
            # Calculate model size
            total_params = sum(p.numel() for p in self.model_server.model.parameters())
            if total_params > 1e9:
                model_size = f"{total_params / 1e9:.1f}B parameters"
            elif total_params > 1e6:
                model_size = f"{total_params / 1e6:.1f}M parameters"
            else:
                model_size = f"{total_params / 1e3:.1f}K parameters"
            
            return ModelInfoResponse(
                model_config=model_config,
                expert_stats=expert_stats,
                vocab_size=self.model_server.tokenizer.vocab_size,
                model_size=model_size,
            )
        
        @app.post("/generate", response_model=GenerationResponse)
        async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
            """Generate text from prompt."""
            if not self.model_server.is_loaded():
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                # Generate text
                results = await self.model_server.generate_async(
                    prompts=[request.prompt],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    do_sample=request.do_sample,
                    num_return_sequences=request.num_return_sequences,
                    stop_tokens=request.stop_tokens,
                    include_prompt=request.include_prompt,
                )
                
                generation_time = time.time() - start_time
                
                # Calculate statistics
                generated_texts = results[0]["generated_text"]
                total_tokens = sum(len(text.split()) for text in generated_texts)
                tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
                
                # Update server statistics
                self.request_count += 1
                self.total_tokens_generated += total_tokens
                
                # Log request (background task)
                background_tasks.add_task(
                    self._log_request,
                    "generate",
                    request.dict(),
                    generation_time,
                    total_tokens
                )
                
                return GenerationResponse(
                    generated_text=generated_texts,
                    generation_time=generation_time,
                    tokens_generated=total_tokens,
                    tokens_per_second=tokens_per_second,
                    model_info=self.model_server.get_model_config(),
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        
        @app.post("/generate/batch", response_model=BatchGenerationResponse)
        async def generate_batch(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
            """Generate text for multiple prompts."""
            if not self.model_server.is_loaded():
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            if len(request.prompts) > self.max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size {len(request.prompts)} exceeds maximum {self.max_batch_size}"
                )
            
            start_time = time.time()
            
            try:
                # Generate text for batch
                batch_results = await self.model_server.generate_async(
                    prompts=request.prompts,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample,
                )
                
                total_time = time.time() - start_time
                
                # Process results
                results = []
                total_tokens = 0
                
                for result in batch_results:
                    generated_texts = result["generated_text"]
                    generation_time = result["generation_time"]
                    tokens_generated = sum(len(text.split()) for text in generated_texts)
                    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                    
                    total_tokens += tokens_generated
                    
                    results.append(GenerationResponse(
                        generated_text=generated_texts,
                        generation_time=generation_time,
                        tokens_generated=tokens_generated,
                        tokens_per_second=tokens_per_second,
                        model_info=self.model_server.get_model_config(),
                    ))
                
                # Update server statistics
                self.request_count += len(request.prompts)
                self.total_tokens_generated += total_tokens
                
                # Log batch request
                background_tasks.add_task(
                    self._log_request,
                    "generate_batch",
                    {"batch_size": len(request.prompts), **request.dict()},
                    total_time,
                    total_tokens
                )
                
                return BatchGenerationResponse(
                    results=results,
                    total_time=total_time,
                    batch_size=len(request.prompts),
                )
                
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
        
        @app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            uptime = time.time() - self.start_time
            avg_tokens_per_request = self.total_tokens_generated / max(self.request_count, 1)
            
            return {
                "uptime": uptime,
                "total_requests": self.request_count,
                "total_tokens_generated": self.total_tokens_generated,
                "avg_tokens_per_request": avg_tokens_per_request,
                "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
                "tokens_per_second": self.total_tokens_generated / uptime if uptime > 0 else 0,
            }
        
        @app.post("/model/reload")
        async def reload_model():
            """Reload the model."""
            try:
                await self.model_server.reload_model()
                return {"status": "success", "message": "Model reloaded successfully"}
            except Exception as e:
                logger.error(f"Model reload failed: {e}")
                raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")
    
    async def _log_request(self, endpoint: str, request_data: dict, duration: float, tokens: int):
        """Log request details (background task)."""
        log_data = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "duration": duration,
            "tokens": tokens,
            "request_size": len(str(request_data)),
        }
        
        logger.info(f"Request logged: {log_data}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the inference server."""
        logger.info(f"Starting inference server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


def start_server(args):
    """Start the inference server from command line arguments."""
    setup_logging()
    
    server = InferenceServer(
        model_path=args.model_path,
        tokenizer_path=args.model_path,  # Assume tokenizer is in same directory
        max_batch_size=getattr(args, 'batch_size', 8),
    )
    
    server.run(
        host=getattr(args, 'host', '0.0.0.0'),
        port=getattr(args, 'port', 8000),
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MoE Inference Server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--batch_size", type=int, default=8, help="Maximum batch size")
    
    args = parser.parse_args()
    start_server(args)
