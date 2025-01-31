from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import TrainingArguments, Trainer
import spacy

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model for text preprocessing only once
nlp = spacy.load("en_core_web_sm")

class ComplianceEvaluator:
    def __init__(self, model_name: str = "bert-base-uncased"): #used bert here instead of DistlBERT/RoBERTa just to strike the right balance between training time, learning loss and also more suited for this type of classification.
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.id2label = {0: "Compliant", 1: "Partially Compliant", 2: "Non-Compliant"}
        self.label2id = {"Compliant": 0, "Partially Compliant": 1, "Non-Compliant": 2}
        self.prompt_templates = [
            "Evaluate if the following policy complies with GDPR regulations, considering data minimization, storage limitation, purpose limitation, and user rights. Identify specific areas of compliance or non-compliance.\n\nPolicy:\n{policy_text}\n\nCompliance Analysis:",
            "Assess the following policy against GDPR regulations, focusing on Article 5 (principles) and Article 13 (transparency). Identify specific violations or compliances.\n\nPolicy:\n{policy_text}\n\nCompliance Analysis:",
            "Identify potential issues related to purpose limitation and storage limitation in the following policy, providing a rationale based on GDPR requirements.\n\nPolicy:\n{policy_text}\n\nIssues and Rationale:",
            "Does this policy adhere to the GDPR's storage limitation principle? Is the purpose of data collection clearly defined and legitimate? Does the policy respect user rights to data access, rectification, and removal? Answer based on the following policy:\n\nPolicy:\n{policy_text}\n\nCompliance Check:",
            "Step 1: Identify the purposes for data collection stated in the following policy. Step 2: Determine if the stated purpose is explicit and legitimate under GDPR. Step 3: Evaluate if the data is stored for any longer than the purpose limitation dictates. Step 4: Is there a clear process for users to request data removal and correction?\n\nPolicy:\n{policy_text}\n\nAnalysis:"
        ]

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3)
            # self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) Mine doesn't supports

    def evaluate_snippet(self, snippet: str):
        self.load_model()  # Lazy load the model when needed
        combined_prompt = "\n\n".join([template.format(policy_text=snippet) for template in self.prompt_templates])
        
        inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}  # Move inputs to GPU if available

        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        confidence = probs[0][predicted_class].item()

        return {"classification": self.id2label[predicted_class], "confidence": confidence}

    def train_from_csv(self, csv_path: str = "policy_data.csv", test_size: float = 0.2, output_dir: str = "compliance-model"):
        self.load_model()  # Ensure model and tokenizer are initialized

        df = pd.read_csv(csv_path)
        df['label'] = df['label'].astype(str).str.strip().str.replace('"', '')
        texts = df['snippet'].tolist()
        labels = [self.label2id[label] for label in df['label']]
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=test_size, random_state=42)

        train_dataset = self.prepare_training_data(train_texts, train_labels)
        val_dataset = self.prepare_training_data(val_texts, val_labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=2e-5, #a even smaller rate might be better, but who knows the time will be
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return trainer.evaluate()

    def prepare_training_data(self, texts, labels):
        encoded_data = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        dataset = Dataset.from_dict({"input_ids": encoded_data["input_ids"], "attention_mask": encoded_data["attention_mask"], "labels": labels})
        return dataset

# Initialize model evaluator (but model will not be loaded yet)
evaluator = ComplianceEvaluator()

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    if "snippet" not in data:
        return jsonify({"error": "Snippet is required"}), 400
    result = evaluator.evaluate_snippet(data["snippet"])
    return jsonify(result)

@app.route("/train", methods=["GET"])
def train():
    metrics = evaluator.train_from_csv("policy_data.csv")
    return jsonify(metrics)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
