# Policy Compliance Classifier

This goal was to classify input policy snippets into three compliance classes partially compliant, compliant and non-compliant, using a model (I chose to fine tune BERT for it's NLP capabilites) and for this purpose I created a sample dataset using gemini 2.0 flash.

## Installation

This project requires the following Python libraries:

- `pandas`
- `numpy`
- `transformers`
- `scikit-learn`
- `torch`
- `spacy`
- `datasets`

You can install these using `pip`:

```bash
pip install -r requirments.txt
python -m spacy download en_core_web_sm #Run this to download SpaCy model
```

## Project Structure

The project is structured as follows:

- `app.py`: Contains the core logic, including data loading, preprocessing, model training, evaluation, and prediction.
- `policy_data.csv`: The CSV file containing the labeled training data. This file _must_ be present for the model training to work.
- `compliance-model/`: The directory where the fine-tuned model weights, configuration, and tokenizer are saved.
- `compliance-model/logs/`: The directory where logs are saved during model training.

### Model Architecture

- **Base Model:** The project uses `bert-base-uncased` as the base model. It consists of a multi-layered bidirectional transformer encoder.
- **Classification Layer:** A linear classification layer is added on top of the BERT encoder output. This layer has a `softmax` activation function and is initialized randomly. It maps the output of the BERT encoder to three class probabilities.

### Training Bert

- **Data Splitting:** The training data is split into training and validation sets (80/20 split by default) using `train_test_split` from scikit-learn. The random state is fixed for reproducibility.
- **Training Arguments:** The `TrainingArguments` class from `transformers` is used to configure training parameters, including:
  - `output_dir`: Directory to save model checkpoints.
  - `num_train_epochs`: Number of training epochs.
  - `per_device_train_batch_size`: Batch size for training on a single device (CPU/GPU).
  - `per_device_eval_batch_size`: Batch size for evaluation.
  - `evaluation_strategy`: Frequency of evaluation ("epoch").
  - `save_strategy`: Frequency of model checkpoints ("epoch").
  - `load_best_model_at_end`: Loads the best model weights based on validation performance at the end of training.
  - `learning_rate`: Learning rate for the optimizer.
  - `logging_dir`: Directory for training logs.
  - `logging_steps`: The number of steps between logging training information.
- **Trainer API:** The `Trainer` class handles the training loop. It manages the training and validation data, loss calculation, backpropagation, and model checkpointing.
- **Optimization:** The model is fine-tuned using the AdamW optimizer, which is standard for transformer models.
- **Loss Function:** The model is trained to minimize the cross-entropy loss function, which is standard for multiclass classification.
  i

## Customization

- **Model:** You can change the pre-trained model used in the `__init__` function.
- **Training Parameters:** Modify the `TrainingArguments` object in the `train_from_csv` method.

## Usage

### Training the Model

For Training, use this curl commannd 

```bash
curl -X GET http://localhost:5000/train
```

### Classifying

For classifying the snippets, use a POST request,

```bash
curl -X POST http://localhost:5000/evaluate \ -H "Content-Type: application/json" \ -d '{"snippet":"This policy aims to minimize data collection, restrict data storage to a limited period, and ensure user rights to access and remove data. This data collection purpose is clearly defined and complies with GDPR regulations. "}'
```