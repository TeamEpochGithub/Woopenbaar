"""Text safety and relevance analysis functionality.

This module provides tools for analyzing text content for safety concerns, dangerous
keywords, sensitivity levels, and relevance to the system's domain. It combines
rule-based and ML-based approaches for comprehensive content moderation.
"""

import os
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification  # type: ignore
from transformers import DistilBertTokenizer  # type: ignore
from transformers import Pipeline  # type: ignore
from transformers import pipeline  # type: ignore

from backend.conf.config import Config


class SentenceAnalyzer:
    """Analyzer for text safety, sensitivity, and relevance.

    Combines multiple analysis techniques including keyword matching,
    hate speech detection, and relevance classification to determine if
    text content is appropriate and on-topic.
    """

    def __init__(self, dangerous_keywords: Optional[List[str]] = None) -> None:
        """Initialize the sentence analyzer.

        Args:
            dangerous_keywords: Optional list of keywords to check for
        """
        self.dangerous_keywords: List[str] = dangerous_keywords or ["bom"]

        # Load sensitivity detection model
        self.sensitivity_classifier: Pipeline = pipeline(
            "text-classification", model="IMSyPP/hate_speech_nl"
        )

        # Define the directory where the model is saved
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.saved_model_path = Config.MODELS_DIR / "safety" / "results"

    def check_keywords(self, text: str) -> List[str]:
        """Check text for dangerous keywords.

        Args:
            text: The text to check

        Returns:
            List of found dangerous keywords
        """
        text_lower = text.lower()
        return [word for word in self.dangerous_keywords if word.lower() in text_lower]

    def check_sensitivity(self, text: str) -> int:
        """Get sensitivity level of the text.

        Returns a value from 0 (not sensitive) to 3 (highly sensitive).

        Args:
            text: The text to analyze

        Returns:
            Integer sensitivity level (0-3)
        """
        try:
            result = self.sensitivity_classifier(text)  # type: ignore
            result: str = result[0]["label"].split("_")[1]  # type: ignore
            return int(result)  # type: ignore
        except Exception as e:
            print(f"Error in sensitivity check: {e}")
            return 0

    def check_relevance(self, text: str) -> int:
        """Check if the text is relevant to the system's domain.

        Uses a fine-tuned BERT model to classify text relevance.

        Args:
            text: The text to analyze

        Returns:
            1 if relevant, 0 if not relevant
        """
        # Load the trained tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained(self.saved_model_path)  # type: ignore
        model = DistilBertForSequenceClassification.from_pretrained(  # type: ignore
            self.saved_model_path
        )

        # Put model in evaluation mode
        model.eval()  # type: ignore

        inputs = tokenizer(  # type: ignore
            text, padding=True, truncation=True, return_tensors="pt"
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)  # type: ignore
            logits: Tensor = outputs.logits  # type: ignore

        """# Convert logits to predicted label
        predicted_label = torch.argmax(logits, dim=1).item()
        return predicted_label"""

        probs: Tensor = F.softmax(logits, dim=1)  # type: ignore

        # Let's assume label 1 corresponds to "relevant"
        relevance_score: float = probs[0][1].item()  # type: ignore

        # Return 1 if the relevance score exceeds the threshold, otherwise 0
        return 1 if relevance_score > 0.3 else 0

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Analyze a sentence for sensitive content.

        Args:
            sentence: The sentence text

        Returns:
            Dictionary with analysis results including:
            - text: Original text
            - dangerous_keywords: List of found dangerous keywords
            - has_dangerous_keywords: Boolean indicating if dangerous keywords were found
            - sensitivity_level: Numeric sensitivity level (0-3)
            - sensitivity_description: Text description of sensitivity level
            - Relevant: Boolean indicating if text is relevant
            - flagged: Boolean indicating if text is flagged (dangerous, sensitive, or irrelevant)
        """
        keywords_found = self.check_keywords(sentence)
        sensitivity_level = self.check_sensitivity(sentence)
        relevant = self.check_relevance(sentence)

        return {
            "text": sentence,
            "dangerous_keywords": keywords_found,
            "has_dangerous_keywords": bool(keywords_found),
            "sensitivity_level": sensitivity_level,
            "sensitivity_description": {
                0: "Not sensitive",
                1: "Low sensitivity",
                2: "Medium sensitivity",
                3: "High sensitivity",
            }[sensitivity_level],
            "Relevant": bool(relevant),
            "flagged": bool(keywords_found)
            or sensitivity_level >= 2
            or not bool(relevant),
        }
