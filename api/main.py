"""
FastAPI application for Mental Health Sentiment Analysis.
Provides endpoints for emotion detection from text and speech inputs.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import logging

# Configuration placeholder (can be expanded for env vars, settings, etc.)
class Settings:
    PROJECT_NAME: str = "Mental Health Sentiment Analysis API"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    MODEL_DIR: str = "distilbert-base-uncased-finetuned-sst-2-english"  # Pre-trained sentiment model
    MAX_BATCH_SIZE: int = 10
    ENABLE_SPEECH: bool = True

settings = Settings()

# Dependency injection placeholder
def get_settings():
    return settings

from models.inference import SentimentPredictor
sentiment_predictor = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(settings.PROJECT_NAME)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, debug=settings.DEBUG)

# Middleware (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

from api.speech_processing import get_speech_processor

@app.on_event("startup")
async def startup_event():
    """Initialize models at startup."""
    global sentiment_predictor
    # Initialize sentiment model
    sentiment_predictor = SentimentPredictor(model_dir=settings.MODEL_DIR)
    logger.info("Sentiment analysis model loaded successfully")
    # Initialize speech processor
    get_speech_processor(model_size="base")
    logger.info("Speech processor initialized")

def get_predictor():
    """Dependency to get the sentiment predictor."""
    return sentiment_predictor

from api.routes import router as sentiment_router
app.include_router(sentiment_router, prefix="/api/v1")

# Root endpoint for health check
default_root_msg = {"message": "Mental Health Sentiment Analysis API is running."}

@app.get("/", tags=["Health"])
async def root():
    return default_root_msg