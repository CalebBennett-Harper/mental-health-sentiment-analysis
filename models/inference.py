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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Add mapping from sentiment score to emotion ---
def map_sentiment_to_emotion(sentiment_score):
    """Map sentiment score to emotional categories"""
    if sentiment_score > 0.8:
        return "happiness", sentiment_score
    elif sentiment_score > 0.6:
        return "excitement", sentiment_score
    elif sentiment_score > 0.2:
        return "calm", sentiment_score
    elif sentiment_score > -0.2:
        return "neutrality", sentiment_score
    elif sentiment_score > -0.6:
        return "sadness", sentiment_score
    elif sentiment_score > -0.8:
        return "anxiety", sentiment_score
    else:
        return "stress", sentiment_score

# Compute a distribution across emotions for more nuanced confidence
def compute_emotion_distribution(sentiment_score: float, sigma: float = 0.25) -> Dict[str, float]:
    """
    Compute a Gaussian-based distribution over emotions.
    """
    emotion_points = {
        "happiness": 1.0,
        "excitement": 0.8,
        "calm": 0.4,
        "neutrality": 0.0,
        "sadness": -0.4,
        "anxiety": -0.7,
        "stress": -1.0,
    }
    weights = {
        emo: np.exp(-0.5 * ((sentiment_score - point) / sigma) ** 2)
        for emo, point in emotion_points.items()
    }
    total = sum(weights.values())
    return {emo: float(weight / total) for emo, weight in weights.items()}

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
        # Remove emotion mapping loading for now, as we are mapping from sentiment
        # self.load_emotion_mappings()
        self.load_model_and_tokenizer()
        logger.info(f"Sentiment predictor initialized on {self.device}")
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        start_time = time.time()
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            
            # Move model to the appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Create pipeline for easy inference, explicitly set task and return_all_scores
            self.pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False
            )
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment from input text and map it to emotion.
        """
        start_time = time.time()
        input_text = text
        try:
            # Use pipeline for sentiment with more explicit error handling
            pipeline_result = self.pipeline(input_text)
            logger.info(f"Pipeline output for input '{input_text}': {pipeline_result}")
            # Check for empty or malformed results
            if not pipeline_result:
                logger.warning(f"Empty pipeline result for '{input_text}', using fallback")
                return {
                    "emotion": "neutrality",
                    "confidence": 0.5,
                    "scores": {"neutrality": 0.5},
                    "processing_time": time.time() - start_time
                }
            try:
                result = pipeline_result[0] if isinstance(pipeline_result, list) else pipeline_result
                logger.info(f"Using result: {result}")
                label = result.get('label', 'NEUTRAL')
                score = result.get('score', 0.5)
                sentiment_score = score if label == 'POSITIVE' else -score
                # Compute nuanced distribution across emotions
                scores = compute_emotion_distribution(sentiment_score)
                primary_emotion = max(scores, key=scores.get)
                confidence = scores[primary_emotion]
                processing_time = time.time() - start_time
                
                return {
                    "emotion": primary_emotion,
                    "confidence": confidence,
                    "scores": scores,
                    "processing_time": processing_time
                }
            except (IndexError, KeyError, AttributeError) as e:
                logger.error(f"Error processing pipeline result: {e}")
                logger.error(f"Raw result was: {pipeline_result}")
                return {
                    "emotion": "neutrality",
                    "confidence": 0.5,
                    "scores": {"neutrality": 0.5},
                    "processing_time": time.time() - start_time
                }
        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            return {
                "emotion": "neutrality",
                "confidence": 0.5,
                "scores": {"neutrality": 0.5},
                "processing_time": time.time() - start_time
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
        
        if len(top_emotions) > 1 and top_emotions[0][1] - top_emotions[1][1] < 0.2:
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