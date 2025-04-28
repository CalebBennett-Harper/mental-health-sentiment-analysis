"""
Speech processing module for mental health sentiment analysis.
Uses OpenAI's Whisper for speech-to-text conversion.
"""

import os
import base64
import tempfile
import logging
import time
from typing import Optional, Tuple, BinaryIO
from pathlib import Path
import whisper  # OpenAI's Whisper
import torch

# Configure logging
logger = logging.getLogger("api.speech_processing")

class SpeechProcessor:
    """Speech to text processing using Whisper."""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the speech processor.
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Whisper {model_size} model on {self.device}")
        start_time = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info(f"Whisper model loaded in {time.time() - start_time:.2f}s")
    
    async def process_audio_file(self, file: BinaryIO, audio_format: Optional[str] = None, language: Optional[str] = None, temperature: float = 0.0) -> Tuple[str, float]:
        """
        Process an audio file and convert it to text.
        Args:
            file: Audio file object
            audio_format: Optional file extension (e.g., '.wav', '.mp3')
            language: Optional language code for transcription (e.g., 'en', 'es')
            temperature: Whisper decoding temperature
        Returns:
            Tuple of (transcribed_text, processing_time)
        """
        start_time = time.time()
        # Try to detect file extension if not provided
        ext = audio_format or getattr(file, 'name', None)
        if ext and not ext.startswith('.'):
            ext = Path(ext).suffix
        if not ext:
            ext = '.wav'  # Default to wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
        try:
            result = self.model.transcribe(temp_path, language=language, temperature=temperature)
            transcribed_text = result["text"].strip()
            processing_time = time.time() - start_time
            logger.info(f"Transcribed audio in {processing_time:.2f}s (lang={language}, temp={temperature})")
            return transcribed_text, processing_time
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def process_base64_audio(self, audio_base64: str, audio_format: str = '.wav', language: Optional[str] = None, temperature: float = 0.0) -> Tuple[str, float]:
        """
        Process base64-encoded audio and convert it to text.
        Args:
            audio_base64: Base64-encoded audio data
            audio_format: File extension (e.g., '.wav', '.mp3')
            language: Optional language code for transcription
            temperature: Whisper decoding temperature
        Returns:
            Tuple of (transcribed_text, processing_time)
        """
        start_time = time.time()
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {str(e)}")
            raise ValueError(f"Invalid base64 audio data: {str(e)}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_format) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        try:
            result = self.model.transcribe(temp_path, language=language, temperature=temperature)
            transcribed_text = result["text"].strip()
            processing_time = time.time() - start_time
            logger.info(f"Transcribed base64 audio in {processing_time:.2f}s (lang={language}, temp={temperature})")
            return transcribed_text, processing_time
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# Singleton instance for reuse
_speech_processor = None

def get_speech_processor(model_size: str = "base") -> SpeechProcessor:
    """
    Get or create a SpeechProcessor instance.
    Args:
        model_size: Whisper model size
    Returns:
        SpeechProcessor instance
    """
    global _speech_processor
    if _speech_processor is None:
        _speech_processor = SpeechProcessor(model_size=model_size)
    return _speech_processor
