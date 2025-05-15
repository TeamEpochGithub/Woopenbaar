"""Model fine-tuning module for safety classification.

This module provides functionality for fine-tuning pretrained transformer models
for safety classification tasks, including data preparation, training, evaluation,
and model saving.
"""

import argparse
import os
from typing import Any

import pandas as pd  # type: ignore
from datasets import Dataset, DatasetDict  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from transformers import DataCollatorWithPadding  # type: ignore
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)


def preprocessing(folder_path: str) -> DatasetDict:
    """
    Load and preprocess text data from text files for classification.

    Args:
        folder_path: Path to folder containing text files with text and labels

    Returns:
        DatasetDict containing train and test splits
    """
    # List to store DataFrames
    df_list: list[pd.DataFrame] = []
    column_names = ["text", "label"]

    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=None, names=column_names)  # type: ignore
            df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    df_combined = pd.concat(df_list, ignore_index=True)

    # Convert to lowercase and strip any spaces
    df_combined["label"] = df_combined["label"].astype(str).str.strip().str.lower()  # type: ignore
    df_combined = df_combined[df_combined["label"].isin(["false", "true"])]  # type: ignore

    # Replace string 'true'/'false' with actual boolean values
    df_combined["label"] = df_combined["label"].map({"true": True, "false": False})  # type: ignore

    # Now safely convert to integers (True → 1, False → 0)
    df_combined["label"] = df_combined["label"].astype(int)  # type: ignore

    train_df, test_df = train_test_split(df_combined, test_size=0.2, random_state=42)  # type: ignore

    # Convert DataFrame to Hugging Face Dataset
    train_dataset: Dataset = Dataset.from_pandas(train_df)  # type: ignore
    test_dataset: Dataset = Dataset.from_pandas(test_df)  # type: ignore

    # Remove the Pandas index column (if it exists)
    train_dataset = (
        train_dataset.remove_columns(["__index_level_0__"])
        if "__index_level_0__" in train_dataset.column_names
        else train_dataset
    )
    test_dataset = (
        test_dataset.remove_columns(["__index_level_0__"])
        if "__index_level_0__" in test_dataset.column_names
        else test_dataset
    )

    # Create a dataset dictionary
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    return dataset


def compute_metrics(pred: Any) -> dict[str, float]:
    """Compute metrics for model evaluation.

    Args:
        pred: Prediction object from trainer

    Returns:
        Dictionary of metrics
    """
    labels: list[int] = pred.label_ids
    preds: list[int] = pred.predictions.argmax(-1)
    precision: float = None
    recall: float = None
    f1: float = None
    _ = precision_recall_fscore_support(  # type: ignore
        labels, preds, average="weighted"
    )
    acc: float = accuracy_score(labels, preds)  # type: ignore
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}  # type: ignore


def train_model(
    dataset: DatasetDict,
    model_name: str = "distilbert/distilbert-base-multilingual-cased",
    text_column: str = "text",
    label_column: str = "label",
    num_labels: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    max_length: int = 512,
    output_dir: str = "./results",
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
    """Fine-tune a model using a HuggingFace dataset.

    Args:
        dataset: HuggingFace dataset dictionary with train and test splits
        model_name: Pre-trained model name
        text_column: Name of the text column in dataset
        label_column: Name of the label column in dataset
        num_labels: Number of classification labels
        batch_size: Training batch size
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        max_length: Maximum sequence length
        output_dir: Directory to save the model

    Returns:
        Tuple of (model, tokenizer)
    """
    # Initialize tokenizer and model
    tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(model_name)  # type: ignore
    model: DistilBertForSequenceClassification = DistilBertForSequenceClassification.from_pretrained(  # type: ignore
        model_name, num_labels=num_labels
    )

    def tokenize_function(examples: Any) -> Any:
        # Get tokenizer output
        tokenized = tokenizer(  # type: ignore
            examples[text_column], padding=True, truncation=True, max_length=max_length  # type: ignore
        )
        # Add labels to the tokenizer output
        tokenized["labels"] = examples[label_column]
        return tokenized  # type: ignore

    # Apply tokenization without removing the label column
    tokenized_dataset: DatasetDict = dataset.map(  # type: ignore
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names],
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,  # type: ignore
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # type: ignore
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()  # type: ignore

    # Evaluate the model
    eval_results = trainer.evaluate()  # type: ignore
    print("\nEvaluation Results:", eval_results)

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)  # type: ignore

    return model, tokenizer  # type: ignore


def main():
    """Parse arguments and run the model training process.

    This function handles command line arguments, sets up the training environment,
    loads and preprocesses data, and initiates model training.
    """
    parser = argparse.ArgumentParser(description="Train a safety classification model")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="generated_text",
        help="Directory containing the training text files",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert/distilbert-base-multilingual-cased",
        help="Pretrained model to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading and preprocessing data from {args.data_dir}...")
    dataset = preprocessing(args.data_dir)

    print(f"Starting model training with {args.model_name}...")
    _, _ = train_model(  # type: ignore
        dataset=dataset,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    print(f"Model training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
