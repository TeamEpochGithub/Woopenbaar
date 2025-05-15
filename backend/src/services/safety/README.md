# Safety Service

The Safety Service provides text content validation to ensure that user inputs and system responses are appropriate, relevant, and safe. It combines multiple validation techniques to detect potentially problematic content.

## Features

- **Text Safety Analysis**: Evaluation of text for potentially unsafe content
- **Relevance Classification**: Determination if text is relevant to the system's domain
- **Sensitivity Detection**: Assessment of text sensitivity levels (0-3)
- **Keyword Detection**: Identification of dangerous or inappropriate keywords

## Components

### SafetyService

The main service class that provides a high-level interface for safety analysis:

- `check_text_safety`: Comprehensive safety analysis of text
- `is_text_safe`: Simple boolean check if text is safe and relevant

### SentenceAnalyzer

Core analysis engine that combines multiple techniques:

- `check_keywords`: Detect dangerous keywords from a predefined list
- `check_sensitivity`: Analyze text sensitivity levels using a hate speech classifier
- `check_relevance`: Classify text relevance using a fine-tuned DistilBERT model
- `analyze_sentence`: Comprehensive analysis combining all methods

## Models

The service uses pre-trained models for classification:

- **Hate Speech Detection**: Identifies potentially harmful content (using IMSyPP/hate_speech_nl)
- **Relevance Classification**: Custom fine-tuned DistilBERT model stored in the models directory

## Fine-tuning

The `model_finetune.py` module provides tools for:

- Training custom relevance classification models with HuggingFace Transformers
- Evaluating model performance with precision, recall, F1, and accuracy metrics
- Saving and loading fine-tuned models

## Text Generation

The `prompt_generation.py` module provides tools for generating test data for safety validation.

## Safety Criteria

Text is flagged as unsafe if any of these conditions are met:

1. It contains dangerous keywords (default includes "bom")
2. It has a sensitivity level of 2 (Medium) or higher
3. It is classified as irrelevant to the system's domain

## Configuration

The safety system configuration is controlled by settings in the Config class:

- `MODELS_DIR`: Directory for model storage
- Default sensitivity thresholds and keyword lists

## Usage

```python
from backend.src.services.safety import SafetyService

# Initialize the service
safety_service = SafetyService()

# Check if text is safe
result = safety_service.check_text_safety("How do I apply for vacation?")

# Structure of result
{
    "text": "How do I apply for vacation?",
    "dangerous_keywords": [],
    "has_dangerous_keywords": False,
    "sensitivity_level": 0,
    "sensitivity_description": "Not sensitive",
    "Relevant": True,
    "flagged": False
}

# Simple check
is_safe = safety_service.is_text_safe("How do I apply for vacation?")
```

## Testing

The `test_safety.py` module contains extensive tests for the safety service functionality.