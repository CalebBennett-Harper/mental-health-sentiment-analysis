#!/usr/bin/env python
"""
Training script for the Mental Health Sentiment Analysis model.
This script handles the training process for the BERT-based emotion classifier.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path

# Import the fine-tuning module
from models.fine_tuning import EmotionClassifier
from transformers import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BERT model for mental health sentiment analysis"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="bert-base-uncased",
        help="Base model to use for fine-tuning"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/fine-tuned",
        help="Directory to save the fine-tuned model"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=500,
        help="Number of warmup steps for learning rate scheduler"
    )
    
    # Data parameters
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/processed",
        help="Directory containing the processed datasets"
    )
    
    # Training features
    parser.add_argument(
        "--use_fp16", 
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Track experiment with Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="mental-health-sentiment",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default=None,
        help="Hugging Face token for downloading models"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=500,
        help="Steps between evaluations"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=1000,
        help="Steps between checkpoint saves"
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=3,
        help="Maximum number of checkpoints to keep"
    )
    
    return parser.parse_args()

def setup_wandb(args):
    """Set up Weights & Biases tracking."""
    try:
        import wandb
        
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"bert-sentiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "fp16": args.use_fp16,
                "gradient_checkpointing": args.gradient_checkpointing,
                "seed": args.seed
            }
        )
        logger.info(f"Weights & Biases initialized: {args.wandb_project}/{run_name}")
        return wandb.run
    except ImportError:
        logger.warning("Weights & Biases not installed. Skipping wandb initialization.")
        return None

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    logger.info(f"Random seeds set to {seed}")

def check_gpu_availability():
    """Check if a GPU is available and log device information."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name} ({device_count} devices available)")
        return True
    else:
        logger.warning("No GPU found. Training will be slow on CPU.")
        return False

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

def save_training_args(args, output_dir):
    """Save training arguments for reproducibility."""
    args_file = os.path.join(output_dir, "training_args.json")
    with open(args_file, "w") as f:
        # Convert args to dictionary, handling non-serializable types
        args_dict = {k: v if isinstance(v, (str, int, float, bool, list, dict)) else str(v) 
                    for k, v in vars(args).items()}
        json.dump(args_dict, f, indent=2)
    logger.info(f"Training arguments saved to {args_file}")

def load_huggingface_token(token):
    """Login to Hugging Face using token if provided."""
    if token:
        from huggingface_hub import login
        login(token)
        logger.info("Logged in to Hugging Face Hub")

def main():
    """Main training function."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    set_random_seeds(args.seed)
    has_gpu = check_gpu_availability()
    create_output_dir(args.output_dir)
    save_training_args(args, args.output_dir)
    load_huggingface_token(args.hf_token)
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.use_wandb:
        wandb_run = setup_wandb(args)
    
    # Create model trainer
    logger.info("Initializing emotion classifier")
    emotion_classifier = EmotionClassifier(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset, val_dataset = emotion_classifier.load_datasets()
    
    # Log dataset information
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Train model
    logger.info("Starting training")
    try:
        emotion_classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            use_wandb=args.use_wandb
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Clean up Weights & Biases
        if wandb_run:
            import wandb
            wandb.finish()

if __name__ == "__main__":
    main()