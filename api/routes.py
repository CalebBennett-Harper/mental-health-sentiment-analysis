from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, File, UploadFile
from api.models import TextRequest, BatchTextRequest, SpeechRequest, SentimentResponse, BatchSentimentResponse
from fastapi.responses import JSONResponse
from typing import List
import time
import logging
from models.inference import SentimentPredictor
import asyncio
from api.main import get_predictor
from api.speech_processing import get_speech_processor

logger = logging.getLogger("api.routes")

router = APIRouter()

@router.post("/analyze/text", response_model=SentimentResponse, tags=["Analysis"])
async def analyze_text(request: TextRequest, predictor=Depends(get_predictor)):
    """
    Analyze sentiment of a single text input.
    Returns emotional analysis with confidence scores.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start = time.time()
        # Attempt explanation, fall back to basic predict on error
        if hasattr(predictor, "analyze_text_with_explanation"):
            try:
                result = predictor.analyze_text_with_explanation(request.text)
            except Exception as ex:
                logger.error("Explanation error, using basic predict", exc_info=True)
                result = predictor.predict(request.text)
        else:
            result = predictor.predict(request.text)
        elapsed = time.time() - start
        logger.info(f"Analyzed text with length {len(request.text)} in {elapsed:.3f}s")
        return SentimentResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores=result["scores"],
            processing_time=elapsed,
            explanation=result.get("explanation")
        )
    except Exception as e:
        logger.exception("Error analyzing text", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}"
        )

@router.post("/analyze/batch", response_model=BatchSentimentResponse, tags=["Analysis"])
async def analyze_batch(request: BatchTextRequest, predictor=Depends(get_predictor)):
    """
    Analyze sentiment of multiple text inputs in a batch.
    Processes each text and returns combined results.
    Uses async processing for batches > 10 for improved performance.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start = time.time()
        results = []
        if len(request.texts) > 10:
            async def process_text(text):
                result = predictor.predict(text)
                return SentimentResponse(
                    emotion=result["emotion"],
                    confidence=result["confidence"],
                    scores=result["scores"],
                    processing_time=0
                )
            results = await asyncio.gather(*[process_text(text) for text in request.texts])
        else:
            for text in request.texts:
                result = predictor.predict(text)
                results.append(SentimentResponse(
                    emotion=result["emotion"],
                    confidence=result["confidence"],
                    scores=result["scores"],
                    processing_time=0
                ))
        elapsed = time.time() - start
        avg_time = elapsed / len(results) if results else 0
        for r in results:
            r.processing_time = avg_time
        logger.info(f"Analyzed batch of {len(request.texts)} texts in {elapsed:.3f}s")
        return BatchSentimentResponse(
            results=results,
            processing_time=elapsed
        )
    except Exception as e:
        logger.error(f"Error analyzing batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing batch: {str(e)}"
        )

@router.post("/analyze/speech", response_model=SentimentResponse, tags=["Analysis"])
async def analyze_speech(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    predictor=Depends(get_predictor),
    speech_processor=Depends(get_speech_processor)
):
    """
    Analyze sentiment from speech input.
    Upload an audio file to be transcribed and analyzed for emotional content.
    Supports various audio formats (WAV, MP3, etc.).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start_time = time.time()
        transcribed_text, transcription_time = await speech_processor.process_audio_file(file.file)
        if not transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not transcribe audio. Please check the audio file."
            )
        if hasattr(predictor, "analyze_text_with_explanation"):
            try:
                result = predictor.analyze_text_with_explanation(transcribed_text)
            except Exception as ex:
                logger.error("Explanation error, using basic predict", exc_info=True)
                result = predictor.predict(transcribed_text)
        else:
            result = predictor.predict(transcribed_text)
        elapsed = time.time() - start_time
        logger.info(f"Analyzed speech in {elapsed:.3f}s (transcription: {transcription_time:.3f}s)")
        return SentimentResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            scores=result["scores"],
            processing_time=elapsed,
            transcribed_text=transcribed_text,
            explanation=result.get("explanation")
        )
    except Exception as e:
        logger.error(f"Error analyzing speech: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing speech: {str(e)}"
        )
