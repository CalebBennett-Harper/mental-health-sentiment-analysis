"""
Pydantic models for API request validation and response formatting.
These models define the structure for mental health sentiment analysis.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class TextRequest(BaseModel):
    """Request model for single text sentiment analysis."""
    text: str = Field(
        ..., 
        min_length=1, 
        example="I'm feeling anxious about tomorrow's meeting",
        description="Text to analyze for emotional content"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional ID for tracking user interactions"
    )

class BatchTextRequest(BaseModel):
    """Request model for batch text sentiment analysis."""
    texts: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=50,  
        example=["I'm happy today!", "I'm stressed at work."],
        description="List of text items to analyze"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional ID for tracking user interactions"
    )

class SpeechRequest(BaseModel):
    """
    Request model for speech sentiment analysis.
    Note: In practice, use File upload via FormData instead of this model.
    """
    audio_base64: str = Field(
        ..., 
        description="Base64-encoded audio data"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional ID for tracking user interactions"
    )

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis results."""
    emotion: str = Field(
        ...,
        description="Primary detected emotion",
        example="anxiety"
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the primary emotion (0-1)",
        example=0.85,
        ge=0,
        le=1
    )
    scores: Dict[str, float] = Field(
        ...,
        description="Scores for all possible emotions",
        example={"anxiety": 0.85, "stress": 0.1, "fear": 0.05}
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the request in seconds",
        example=0.254
    )
    transcribed_text: Optional[str] = Field(
        None,
        description="Transcribed text from speech input (only for speech requests)",
        example="I'm feeling worried about the upcoming presentation"
    )
    explanation: Optional[str] = Field(
        None,
        description="Explanation of the emotional analysis",
        example="This text expresses moderate anxiety with elements of stress."
    )

class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis results."""
    results: List[SentimentResponse] = Field(
        ...,
        description="List of sentiment analysis results"
    )
    processing_time: float = Field(
        ...,
        description="Total time taken to process the batch request in seconds",
        example=1.254
    )
