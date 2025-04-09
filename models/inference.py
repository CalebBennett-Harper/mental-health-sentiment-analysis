"""
Model inference module for BERT-based sentiment analysis.
Provides efficient inference functions for both API and standalone usage.
"""

import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Class for efficient sentiment prediction using BERT model."""
    
    def __init__(
        self, 
        model_dir: str,
        device: Optional[str] = None
    ):
        """
        Initialize the sentiment predictor.
        
        Args:
            model_dir: Directory containing the fine-tuned model
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_dir = model_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load emotion mappings
        self.load_emotion_mappings()
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        logger.info(f"Sentiment predictor initialized on {self.device}")
    
    def load_emotion_mappings(self):
        """Load emotion ID to label mappings."""
        emotion_mapping_path = os.path.join(self.model_dir, "emotion_mapping.json")
        
        if os.path.exists(emotion_mapping_path):
            with open(emotion_mapping_path, 'r') as f:
                self.id2label = json.load(f)
                self.label2id = {v: k for k, v in self.id2label.items()}
            logger.info(f"Loaded emotion mappings: {list(self.id2label.values())}")
        else:
            # Default emotions if mapping file not found
            logger.warning("Emotion mapping file not found. Using default mapping.")
            self.id2label = {str(i): emotion for i, emotion in enumerate([
                "stress", "anxiety", "sadness", "happiness", "neutrality", 
                "frustration", "fear", "excitement", "calm", "overwhelmed"
            ])}
            self.label2id = {v: str(i) for i, v in enumerate(self.id2label.values())}
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            
            # Load model
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment from input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion, confidence, and all emotion scores
        """
        start_time = time.time()
        
        # Format input with instruction
        input_text = f"Analyze the emotional content of this text: {text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process logits
        logits = outputs.logits[0].cpu().numpy()
        probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
        
        # Get prediction and confidence
        prediction_id = np.argmax(probabilities)
        emotion = self.id2label[str(prediction_id)]
        confidence = float(probabilities[prediction_id])
        
        # Create scores dictionary
        scores = {self.id2label[str(i)]: float(prob) for i, prob in enumerate(probabilities)}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
            "processing_time": processing_time
        }
    
    def predict_pipeline(self, text: str) -> Dict:
        """
        Predict sentiment using the pipeline (simpler but slightly less flexible).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion, confidence, and scores
        """
        start_time = time.time()
        
        # Format input with instruction
        input_text = f"Analyze the emotional content of this text: {text}"
        
        # Get prediction
        result = self.pipeline(input_text)[0]
        label = result['label']
        score = result['score']
        
        # Map back to emotion
        emotion_id = int(label.split('_')[-1]) if '_' in label else int(label)
        emotion = self.id2label[str(emotion_id)]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create a simplified response with just the top emotion
        return {
            "emotion": emotion,
            "confidence": float(score),
            "processing_time": processing_time
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def analyze_text_with_explanation(self, text: str) -> Dict:
        """
        Analyze text and provide an explanation of the emotional content.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with emotion analysis and explanation
        """
        # Get basic prediction
        prediction = self.predict(text)
        
        # Create explanation based on the scores
        primary_emotion = prediction["emotion"]
        confidence = prediction["confidence"]
        scores = prediction["scores"]
        
        # Get top 3 emotions
        top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate explanation
        if confidence > 0.7:
            certainty = "strongly"
        elif confidence > 0.5:
            certainty = "moderately"
        else:
            certainty = "slightly"
            
        explanation = f"This text expresses {certainty} {primary_emotion}. "
        
        if top_emotions[0][1] - top_emotions[1][1] < 0.2:
            # If top emotions are close, mention the secondary emotion
            explanation += f"There are also elements of {top_emotions[1][0]} present. "
        
        # Add analysis of emotional intensity
        total_emotional_intensity = sum(scores.values()) / len(scores)
        if total_emotional_intensity > 0.6:
            explanation += "The overall emotional intensity is high."
        elif total_emotional_intensity > 0.4:
            explanation += "The overall emotional intensity is moderate."
        else:
            explanation += "The overall emotional intensity is low."
        
        # Add explanation to the prediction
        prediction["explanation"] = explanation
        
        return prediction


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentiment analysis inference")
    parser.add_argument("--model-dir", default="models/fine-tuned", help="Model directory")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Device to run on")
    parser.add_argument("--explain", action="store_true", help="Include explanation in output")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SentimentPredictor(args.model_dir, args.device)
    
    # Run prediction
    if args.text:
        if args.explain:
            result = predictor.analyze_text_with_explanation(args.text)
        else:
            result = predictor.predict(args.text)
        print(json.dumps(result, indent=2))
    else:
        # Interactive mode
        print("Enter text to analyze (Ctrl+C to exit):")
        try:
            while True:
                text = input("> ")
                if not text:
                    continue
                
                if args.explain:
                    result = predictor.analyze_text_with_explanation(text)
                else:
                    result = predictor.predict(text)
                print(json.dumps(result, indent=2))
        except KeyboardInterrupt:
            print("\nExiting...")