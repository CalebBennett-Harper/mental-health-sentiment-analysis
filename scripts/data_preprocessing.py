"""
Data preprocessing script for mental health sentiment analysis.
This script handles loading, cleaning, and tokenizing dataset for Llama 4 fine-tuning.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import LlamaTokenizer
import json
import logging
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define constants
EMOTIONS = ["stress", "anxiety", "sadness", "happiness", "neutrality", 
            "frustration", "fear", "excitement", "calm", "overwhelmed"]
RANDOM_SEED = 42
OUTPUT_DIR = "data/processed"
TOKENIZER_MODEL = "meta-llama/Llama-2-7b-hf"  # Will be replaced with Llama 4 when available

class DataProcessor:
    def __init__(self, tokenizer_name=TOKENIZER_MODEL, max_length=512):
        """Initialize DataProcessor with tokenizer and parameters."""
        logger.info(f"Initializing tokenizer: {tokenizer_name}")
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def clean_text(self, text):
        """Clean text data by removing special characters, extra spaces, etc."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_clean_dataset(self, file_path):
        """Load dataset from file and clean text."""
        logger.info(f"Loading dataset from {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Check if required columns exist
        required_cols = ['text', 'emotion']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Dataset missing required columns: {required_cols}")
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        # Clean text column
        logger.info("Cleaning text data")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Filter for relevant emotions if needed
        if 'emotion' in df.columns and not df['emotion'].isin(EMOTIONS).all():
            logger.info(f"Remapping emotions to {EMOTIONS}")
            # Implement emotion mapping logic here
            # This will depend on your specific dataset
        
        return df
    
    def tokenize_dataset(self, df):
        """Tokenize text using Llama tokenizer."""
        logger.info("Tokenizing dataset")
        
        tokenized_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Create input text with instruction format
            input_text = f"Analyze the emotional content of this text: {row['cleaned_text']}"
            
            # Tokenize input
            tokenized_input = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Prepare output as emotion label
            label = EMOTIONS.index(row['emotion']) if row['emotion'] in EMOTIONS else -1
            
            if label == -1:
                continue  # Skip rows with unmapped emotions
                
            tokenized_data.append({
                "input_ids": tokenized_input["input_ids"].squeeze().tolist(),
                "attention_mask": tokenized_input["attention_mask"].squeeze().tolist(),
                "label": label
            })
        
        return tokenized_data
    
    def split_and_save_dataset(self, tokenized_data, train_ratio=0.8, val_ratio=0.1):
        """Split dataset into train, validation, and test sets and save."""
        logger.info("Splitting dataset into train/val/test sets")
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            tokenized_data, 
            test_size=(1 - train_ratio - val_ratio),
            random_state=RANDOM_SEED
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=RANDOM_SEED
        )
        
        # Save splits
        logger.info(f"Saving datasets to {OUTPUT_DIR}")
        with open(os.path.join(OUTPUT_DIR, "train.json"), "w") as f:
            json.dump(train, f)
        
        with open(os.path.join(OUTPUT_DIR, "val.json"), "w") as f:
            json.dump(val, f)
            
        with open(os.path.join(OUTPUT_DIR, "test.json"), "w") as f:
            json.dump(test, f)
        
        logger.info(f"Saved {len(train)} training, {len(val)} validation, and {len(test)} test examples")
        
        # Also save emotion mapping
        with open(os.path.join(OUTPUT_DIR, "emotion_mapping.json"), "w") as f:
            json.dump({i: emotion for i, emotion in enumerate(EMOTIONS)}, f)
    
    def process_dataset(self, file_path):
        """Full pipeline to process dataset."""
        df = self.load_and_clean_dataset(file_path)
        tokenized_data = self.tokenize_dataset(df)
        self.split_and_save_dataset(tokenized_data)
        logger.info("Dataset processing completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process dataset for Llama fine-tuning")
    parser.add_argument("--input", required=True, help="Path to input dataset file (CSV or JSON)")
    parser.add_argument("--tokenizer", default=TOKENIZER_MODEL, help="Tokenizer model name")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    processor = DataProcessor(tokenizer_name=args.tokenizer, max_length=args.max_length)
    processor.process_dataset(args.input)