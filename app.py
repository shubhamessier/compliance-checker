import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
import logging
from typing import List, Dict, Tuple
import spacy
from datasets import Dataset

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

csv_path = "policy_data.csv"

class ComplianceEvaluator:

    def __init__(self, model_name: str = "bert-base-uncased"):
       
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # Ensure it matches label count
        )
        self.id2label = {0: "Compliant", 1: "Partially Compliant", 2: "Non-Compliant"}
        self.label2id = {"Compliant": 0, "Partially Compliant": 1, "Non-Compliant": 2}
    
    def load_csv_data(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load CSV file and preprocess labels correctly.
        """
        # Read CSV file
        df = pd.read_csv(file_path)

        # Debugging: Print unique labels in CSV
        print("Unique Labels in CSV (Before Cleaning):", df['label'].unique())

        # Normalize labels: Remove spaces & trailing quotes
        df['label'] = df['label'].astype(str).str.strip().str.replace('"', '')

        # Debugging: Print cleaned labels
        print("Unique Labels in CSV (After Cleaning):", df['label'].unique())

        # Convert labels to numeric values using label2id mapping
        texts = df['snippet'].tolist()
        labels = [self.label2id[label] for label in df['label']]

        return texts, labels
    
    def prepare_training_data(self, texts: List[str], labels: List[int]) -> Dataset:
        """
        Tokenizes text and prepares dataset for training.
        """
        encoded_data = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        dataset = Dataset.from_dict({
            "input_ids": encoded_data["input_ids"],
            "attention_mask": encoded_data["attention_mask"],
            "labels": labels
        })
        
        return dataset

    def train_from_csv(self, csv_path: str, test_size: float = 0.2, output_dir: str = "compliance-model"):
        """
        Train model using CSV data.
        """
        # Load data from CSV
        texts, labels = self.load_csv_data(csv_path)
        
        # Split into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Prepare datasets
        train_dataset = self.prepare_training_data(train_texts, train_labels)
        val_dataset = self.prepare_training_data(val_texts, val_labels)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",  # Fixed deprecated `evaluation_strategy`
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=2e-5,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
        )
        
        # Initialize and start training
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Return validation metrics
        metrics = trainer.evaluate()
        return metrics

    def evaluate_snippet(self, snippet: str) -> Dict:
        """
        Evaluate a single policy snippet and classify its compliance level.
        """
        inputs = self.tokenizer(snippet, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        
        # Get predicted class and confidence score
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][predicted_class].item()
        
        return {
            "classification": self.id2label[predicted_class],
            "confidence": confidence,
            "snippet": snippet
        }


def test_compliance_system():
    
    evaluator = ComplianceEvaluator()
    print("\nTraining model...")
    metrics = evaluator.train_from_csv(csv_path)
    print(f"Training metrics: {metrics}")
    
    # Test evaluation on new snippets
    test_snippets = [
        "We encrypt all data using industry-standard protocols.",
        "Data retention periods are flexible based on business needs.",
        "No privacy policy is implemented."
    ]
    
    print("\nTesting evaluations:")
    for snippet in test_snippets:
        result = evaluator.evaluate_snippet(snippet)
        print(f"\nSnippet: {snippet}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2f}")
        

if __name__ == "__main__":
    test_compliance_system()
