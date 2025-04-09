"""
Model fine-tuning script for sentiment analysis using BERT.
This script fine-tunes BERT for emotion classification.
"""

import os
import json
import logging
import torch
import numpy as np
from datasets import Dataset
import transformers
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from huggingface_hub import login
from argparse import ArgumentParser
from tqdm import tqdm
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "bert-base-uncased"  # Using BERT base model
DATA_DIR = "data/processed"
OUTPUT_DIR = "models/fine-tuned"
NUM_LABELS = 10  # Number of emotions

class EmotionClassifier:
    def __init__(self, model_name=MODEL_NAME, output_dir=OUTPUT_DIR):
        """Initialize model fine-tuning class."""
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load emotion mapping
        emotion_mapping_path = os.path.join(DATA_DIR, "emotion_mapping.json")
        if os.path.exists(emotion_mapping_path):
            with open(emotion_mapping_path, 'r') as f:
                self.id2label = json.load(f)
                self.label2id = {v: k for k, v in self.id2label.items()}
        else:
            logger.warning("Emotion mapping file not found. Using default mapping.")
            self.id2label = {str(i): emotion for i, emotion in enumerate([
                "stress", "anxiety", "sadness", "happiness", "neutrality", 
                "frustration", "fear", "excitement", "calm", "overwhelmed"
            ])}
            self.label2id = {v: str(i) for i, v in enumerate([
                "stress", "anxiety", "sadness", "happiness", "neutrality", 
                "frustration", "fear", "excitement", "calm", "overwhelmed"
            ])}
    
    def load_datasets(self):
        """Load processed datasets."""
        logger.info("Loading datasets")
        
        # Load train dataset
        with open(os.path.join(DATA_DIR, "train.json"), 'r') as f:
            train_data = json.load(f)
        
        # Load validation dataset
        with open(os.path.join(DATA_DIR, "val.json"), 'r') as f:
            val_data = json.load(f)
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_dict({
            'input_ids': [item['input_ids'] for item in train_data],
            'attention_mask': [item['attention_mask'] for item in train_data],
            'token_type_ids': [item['token_type_ids'] for item in train_data],  # BERT uses token_type_ids
            'labels': [item['label'] for item in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': [item['input_ids'] for item in val_data],
            'attention_mask': [item['attention_mask'] for item in val_data],
            'token_type_ids': [item['token_type_ids'] for item in val_data],  # BERT uses token_type_ids
            'labels': [item['label'] for item in val_data]
        })
        
        logger.info(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
    
    def load_model(self):
        """Load the pre-trained BERT model."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Load the model
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return model, tokenizer
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"]
        }
    
    def train(self, train_dataset, val_dataset, batch_size=16, learning_rate=2e-5, epochs=3, use_wandb=False):
        """Train the model."""
        logger.info("Starting model fine-tuning")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model()
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb" if use_wandb else "none",
            logging_steps=100,
            # Enable gradient checkpointing for memory efficiency
            gradient_checkpointing=True,
            # Enable mixed precision training for faster performance
            fp16=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        logger.info("Training model...")
        trainer.train()
        
        # Save the trained model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training complete!")
        
        # Evaluate model on test set
        self.evaluate_test_set(model, tokenizer)
    
    def evaluate_test_set(self, model, tokenizer):
        """Evaluate the model on the test set."""
        logger.info("Evaluating model on test set")
        
        # Load test data
        with open(os.path.join(DATA_DIR, "test.json"), 'r') as f:
            test_data = json.load(f)
        
        test_dataset = Dataset.from_dict({
            'input_ids': [item['input_ids'] for item in test_data],
            'attention_mask': [item['attention_mask'] for item in test_data],
            'token_type_ids': [item['token_type_ids'] for item in test_data],  # BERT uses token_type_ids
            'labels': [item['label'] for item in test_data]
        })
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            compute_metrics=self.compute_metrics,
        )
        
        # Run evaluation
        results = trainer.evaluate(test_dataset)
        
        # Save results
        with open(os.path.join(self.output_dir, "test_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results: {results}")

    def create_pipeline_model(self):
        """Create a pipeline model for inference."""
        logger.info("Creating model pipeline for inference")
        
        from transformers import pipeline
        
        # Load the trained model
        model_path = self.output_dir
        
        # Create pipeline
        emotion_pipeline = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return emotion_pipeline


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    parser.add_argument("--model", default=MODEL_NAME, help="Pre-trained model name")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for tracking")
    parser.add_argument("--hf-token", help="Hugging Face token for accessing models")
    
    args = parser.parse_args()
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        login(args.hf_token)
    
    # Create fine-tuner and run training
    classifier = EmotionClassifier(model_name=args.model, output_dir=args.output)
    train_dataset, val_dataset = classifier.load_datasets()
    classifier.train(
        train_dataset, 
        val_dataset, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        use_wandb=args.use_wandb
    )