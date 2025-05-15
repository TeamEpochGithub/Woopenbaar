"""BERT tokenizer wrapper module.

This module provides tokenizer classes for text processing.
The module includes an abstract base class and a BERT-based implementation.
"""

import abc
import logging
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer  # type: ignore

logger = logging.getLogger(__name__)


class BaseTokenizer(abc.ABC):
    """Base class for text tokenizers.

    Defines the interface for tokenization systems.

    Attributes:
        max_length (int): Maximum sequence length
    """

    def __init__(self, max_length: int) -> None:
        """Initialize the tokenizer.

        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length

    @abc.abstractmethod
    def tokenize(self, texts: List[str], batch_size: int = 128) -> List[List[str]]:
        """Tokenize a list of texts.

        Args:
            texts: List of texts to tokenize
            batch_size: Batch size for processing

        Returns:
            List of tokenized texts
        """

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, max_length: int
    ) -> "BaseTokenizer":
        """Create a tokenizer instance from a pretrained model.

        Args:
            model_name_or_path: Model name or path
            max_length: Maximum sequence length

        Returns:
            Initialized tokenizer instance
        """
        return cls(max_length)


class BertTokenizer(BaseTokenizer):
    """BERT tokenizer implementation.

    Transforms input strings into token IDs in a consistent manner, allowing the model to understand them.
    The tokenizer determines how input strings are mapped to token IDs.
    For instance, if the phrase 'Minister van' frequently appears together, it might be tokenized as a single token.

    Attributes:
        tokenizer: The underlying HuggingFace tokenizer
        max_length (int): Maximum sequence length
    """

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, max_length: int
    ) -> "BertTokenizer":
        """Create a BERT tokenizer instance from a pretrained model.

        Args:
            model_name_or_path: HuggingFace model name or path
            max_length: Maximum sequence length

        Returns:
            Initialized BertTokenizer instance
        """
        return cls(model_name_or_path, max_length)

    def __init__(self, model_name: str, max_length: int) -> None:
        """Initialize the BERT tokenizer.

        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
        """
        super().__init__(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # type: ignore

    def tokenize(self, texts: List[str], batch_size: int = 128) -> List[List[str]]:
        """Tokenize a list of texts into word pieces.

        Args:
            texts: List of texts to tokenize
            batch_size: Batch size for processing

        Returns:
            List of tokenized texts
        """
        tokenized = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing texts"):
            batch = texts[i : i + batch_size]
            encoding = self.tokenizer(  # type: ignore
                batch,
                truncation=False,
                max_length=self.max_length,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            tokens_batch = [  # type: ignore
                self.tokenizer.convert_ids_to_tokens(ids)  # type: ignore
                for ids in encoding["input_ids"]  # type: ignore
            ]
            tokenized.extend(tokens_batch)  # type: ignore
        return tokenized  # type: ignore
