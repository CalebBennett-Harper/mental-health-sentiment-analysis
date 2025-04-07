"""
Configuration file for model parameters and training settings.
This centralized configuration allows for easy adjustment of hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FINE_TUNED_MODEL_DIR = os.path.join(MODEL_DIR, "fine-tuned")

# Define emotions to detect
EMOTIONS = [
    "stress", "anxiety", "sadness", "happiness", "neutrality", 
    "frustration", "fear", "excitement", "calm", "overwhelmed"
]

# Define emotion descriptions for interpretations
EMOTION_DESCRIPTIONS = {
    "stress": "High tension or pressure that may be affecting wellbeing",
    "anxiety": "Worry or nervousness about future events or uncertainty",
    "sadness": "Low mood or feelings of loss and disappointment",
    "happiness": "Positive emotional state reflecting joy and contentment",
    "neutrality": "Balanced emotional state with neither strong positive nor negative feelings",
    "frustration": "Feelings of being upset or annoyed due to inability to achieve goals",
    "fear": "Response to perceived threat or danger",
    "excitement": "Energetic enthusiasm and eagerness",
    "calm": "Peaceful, centered emotional state with low arousal",
    "overwhelmed": "Feeling of being emotionally overloaded or unable to cope"
}

# Define emotion color mapping for visualizations
EMOTION_COLORS = {
    "stress": "#FF5252",       # Red
    "anxiety": "#FF9800",      # Orange
    "sadness": "#2196F3",      # Blue
    "happiness": "#4CAF50",    # Green
    "neutrality": "#9E9E9E",   # Gray
    "frustration": "#F44336",  # Light Red
    "fear": "#673AB7",         # Purple
    "excitement": "#FFEB3B",   # Yellow
    "calm": "#00BCD4",         # Cyan
    "overwhelmed": "#795548"   # Brown
}

# Model configuration
@dataclass
class ModelConfig:
    """Configuration for the model architecture and parameters."""
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Will update to Llama 4 when available
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf"  # Will update to Llama 4 when available
    max_length: int = 512
    num_labels: int = len(EMOTIONS)
    id2label: Dict[str, str] = field(default_factory=lambda: {str(i): e for i, e in enumerate(EMOTIONS)})
    label2id: Dict[str, str] = field(default_factory=lambda: {e: str(i) for i, e in enumerate(EMOTIONS)})

# LoRA configuration
@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    enabled: bool = True
    r: int = 16  # Rank
    alpha: int = 32  # Alpha parameter for LoRA
    dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_CLS"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj", 
        "gate_proj", "down_proj", "up_proj"
    ])

# Training configuration
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    seed: int = 42
    fp16: bool = True
    logging_steps: int = 100

# Data preprocessing configuration
@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    clean_text: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    lowercase: bool = True
    max_samples_per_emotion: Optional[int] = None  # Set to limit samples per emotion

# Inference configuration
@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    quantization: Optional[str] = "4bit"  # "4bit", "8bit", or None
    device: Optional[str] = None  # "cpu", "cuda", or None for auto-detection
    batch_size: int = 16
    max_concurrent_requests: int = 10

# Whisper configuration
@dataclass
class WhisperConfig:
    """Configuration for OpenAI Whisper."""
    model_size: str = "base"  # "tiny", "base", "small", "medium", "large"
    device: Optional[str] = None  # "cpu", "cuda", or None for auto-detection
    language: Optional[str] = "en"  # Set to specific language or None for auto-detection
    temperature: float = 0.0  # Lower values for more focused/deterministic outputs
    beam_size: int = 5  # Beam search size

# API configuration
@dataclass
class APIConfig:
    """Configuration for the FastAPI backend."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 60  # seconds

# Frontend configuration
@dataclass
class FrontendConfig:
    """Configuration for the Streamlit frontend."""
    host: str = "0.0.0.0"
    port: int = 8501
    theme_color: str = "blue"
    page_title: str = "Mental Health Sentiment Analysis"
    page_icon: str = "❤️"
    layout: str = "wide"
    api_url: str = "http://localhost:8000"
    recording_sample_rate: int = 16000
    recording_channels: int = 1

# Combined configuration
@dataclass
class Config:
    """Combined configuration for the entire application."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    api: APIConfig = field(default_factory=APIConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)

    # Paths
    base_dir: str = BASE_DIR
    data_dir: str = DATA_DIR
    model_dir: str = MODEL_DIR
    raw_data_dir: str = RAW_DATA_DIR
    processed_data_dir: str = PROCESSED_DATA_DIR
    fine_tuned_model_dir: str = FINE_TUNED_MODEL_DIR

    # Emotion settings
    emotions: List[str] = field(default_factory=lambda: EMOTIONS)
    emotion_descriptions: Dict[str, str] = field(default_factory=lambda: EMOTION_DESCRIPTIONS)
    emotion_colors: Dict[str, str] = field(default_factory=lambda: EMOTION_COLORS)

# Default configuration instance
config = Config()

# Function to load configuration from environment variables
def load_config_from_env() -> Config:
    """
    Load configuration from environment variables.
    Environment variables should be prefixed with 'SENTIMENT_' followed by the configuration path.
    For example, SENTIMENT_MODEL_NAME would set config.model.model_name.
    """
    import os
    
    cfg = Config()
    
    # Function to set nested attribute based on path
    def set_nested_attr(obj, path, value):
        parts = path.lower().split('_')
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return
        
        if hasattr(obj, parts[-1]):
            attr = parts[-1]
            attr_type = type(getattr(obj, attr))
            
            # Convert value to the correct type
            if attr_type == bool:
                value = value.lower() in ('true', 'yes', '1', 'y')
            elif attr_type == int:
                value = int(value)
            elif attr_type == float:
                value = float(value)
            elif attr_type == list:
                value = value.split(',')
            
            setattr(obj, attr, value)
    
    # Process environment variables
    for key, value in os.environ.items():
        if key.startswith('SENTIMENT_'):
            path = key[10:]  # Remove 'SENTIMENT_' prefix
            set_nested_attr(cfg, path, value)
    
    return cfg

# Load configuration from file
def load_config_from_file(file_path: str) -> Config:
    """Load configuration from a JSON or YAML file."""
    import json
    import yaml
    from dataclasses import asdict
    
    cfg = Config()
    
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.endswith(('.yaml', '.yml')):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
    
    # Convert flat dictionary to nested structure
    nested_data = {}
    for key, value in data.items():
        parts = key.split('.')
        d = nested_data
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    
    # Update configuration with nested data
    cfg_dict = asdict(cfg)
    
    def update_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                update_dict(target[key], value)
            else:
                target[key] = value
    
    update_dict(cfg_dict, nested_data)
    
    # Convert dictionary back to dataclass
    return Config(**cfg_dict)

# Save configuration to file
def save_config_to_file(config: Config, file_path: str) -> None:
    """Save configuration to a JSON or YAML file."""
    import json
    import yaml
    from dataclasses import asdict
    
    # Convert dataclass to dictionary
    config_dict = asdict(config)
    
    if file_path.endswith('.json'):
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif file_path.endswith(('.yaml', '.yml')):
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format for {file_path}")

if __name__ == "__main__":
    # Print current configuration
    import json
    from dataclasses import asdict
    
    print(json.dumps(asdict(config), indent=2))