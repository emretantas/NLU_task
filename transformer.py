import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate
from sklearn.metrics import f1_score, accuracy_score

# Setup and Tokenization
MODEL_ID = "tasksource/ModernBERT-large-nli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Data Cleansing. It avoids CUDA crashing.

print("Loading and cleaning data...")
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("dev.csv")

# Droping all blank spaces of the columns
train_df = train_df.dropna(subset=['premise', 'hypothesis', 'label']).copy()
val_df = val_df.dropna(subset=['premise', 'hypothesis', 'label']).copy()

# Maping to every numerical possible option to integer
label_mapping = {
    "NOT_ENTAILMENT": 0, "ENTAILMENT": 1,
    "0": 0, "1": 1,
    "0.0": 0, "1.0": 1,
    0: 0, 1: 1,
    0.0: 0, 1.0: 1
}

train_df['label'] = train_df['label'].map(label_mapping)
val_df['label'] = val_df['label'].map(label_mapping)

#Dropping all rows that was mapped  to NaN
train_df = train_df.dropna(subset=['label'])
val_df = val_df.dropna(subset=['label'])

# Forcing being integers
train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)

print(f"Cleaned Training Data: {len(train_df)} rows")
print(f"Cleaned Validation Data: {len(val_df)} rows")

# Convert pandas to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def preprocess_function(examples):
    texts = [f"{p} [SEP] {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]
    return tokenizer(texts, truncation=True, max_length=256, padding="max_length")

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)


# Metric and Model Initialization
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    ignore_mismatched_sizes=True,
    id2label={0: "NOT_ENTAILMENT", 1: "ENTAILMENT"},
    label2id={"NOT_ENTAILMENT": 0, "ENTAILMENT": 1}
)


# Training Loop
training_args = TrainingArguments(
    output_dir="./modernbert-large-rte-final",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting the training")
trainer.train()



# # Threshold Tunning(Optimization)

# A. Get raw probability predictions
predictions = trainer.predict(tokenized_val)
raw_scores = predictions.predictions

# B. Convert raw logits into percentages
probabilities = torch.nn.functional.softmax(torch.tensor(raw_scores), dim=-1).numpy()

# C. Isolate Entailment (Class 1) probabilities and true labels
prob_entailment = probabilities[:, 1]
true_labels = predictions.label_ids

# Trackers
best_acc_threshold = 0.5
best_acc = 0.0

best_f1_threshold = 0.5
best_f1 = 0.0

# E. Loop through thresholds from 0.10 to 0.90
for thresh in np.arange(0.1, 0.9, 0.01):
    custom_guesses = (prob_entailment >= thresh).astype(int)

    current_acc = accuracy_score(true_labels, custom_guesses)
    current_f1 = f1_score(true_labels, custom_guesses, average="binary")

    # Best accuracy
    if current_acc > best_acc:
        best_acc = current_acc
        best_acc_threshold = thresh

    # Best F1 score
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_threshold = thresh

default_guesses = (prob_entailment >= 0.5).astype(int)
print(f"Default 0.50 Threshold Accuracy: {accuracy_score(true_labels, default_guesses):.4f}")
print(f"Default 0.50 Threshold F1:       {f1_score(true_labels, default_guesses, average='binary'):.4f}")
print("----------------------------------")
print(f"Optimal ACCURACY Threshold: {best_acc_threshold:.2f} (Score: {best_acc:.4f})")
print(f"Optimal F1 Score Threshold: {best_f1_threshold:.2f} (Score: {best_f1:.4f})")

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

ORIGINAL_MODEL_ID = "answerdotai/ModernBERT-large"
tokenizer_orig = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_ID)

train_df = pd.read_csv("train.csv").dropna(subset=['premise', 'hypothesis', 'label']).copy()
val_df = pd.read_csv("dev.csv").dropna(subset=['premise', 'hypothesis', 'label']).copy()

label_mapping = {
    "NOT_ENTAILMENT": 0, "ENTAILMENT": 1,
    "0": 0, "1": 1, "0.0": 0, "1.0": 1,
    0: 0, 1: 1, 0.0: 0, 1.0: 1
}
train_df['label'] = train_df['label'].map(label_mapping).astype(int)
val_df['label'] = val_df['label'].map(label_mapping).astype(int)

train_dataset_orig = Dataset.from_pandas(train_df)
val_dataset_orig = Dataset.from_pandas(val_df)

def preprocess_orig(examples):
    texts = [f"{p} [SEP] {h}" for p, h in zip(examples['premise'], examples['hypothesis'])]
    return tokenizer_orig(texts, truncation=True, max_length=256, padding="max_length")

tokenized_train_orig = train_dataset_orig.map(preprocess_orig, batched=True)
tokenized_val_orig = val_dataset_orig.map(preprocess_orig, batched=True)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics_orig(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

model_orig = AutoModelForSequenceClassification.from_pretrained(
    ORIGINAL_MODEL_ID,
    num_labels=2,
    id2label={0: "NOT_ENTAILMENT", 1: "ENTAILMENT"},
    label2id={"NOT_ENTAILMENT": 0, "ENTAILMENT": 1}
)

training_args_orig = TrainingArguments(
    output_dir="./modernbert-original",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True
)

trainer_orig = Trainer(
    model=model_orig,
    args=training_args_orig,
    train_dataset=tokenized_train_orig,
    eval_dataset=tokenized_val_orig,
    processing_class=tokenizer_orig,
    compute_metrics=compute_metrics_orig
)

print("Training original ModernBERT-large...")
trainer_orig.train()

# Threshold tuning for original ModernBERT
orig_preds = trainer_orig.predict(tokenized_val_orig)
orig_probs = torch.softmax(torch.tensor(orig_preds.predictions), dim=-1).numpy()
orig_labels = orig_preds.label_ids

prob_entailment_orig = orig_probs[:, 1]

best_acc_thresh, best_acc = 0.5, 0.0
best_f1_thresh, best_f1 = 0.5, 0.0

for thresh in np.arange(0.05, 0.95, 0.01):
    preds = (prob_entailment_orig >= thresh).astype(int)
    acc = accuracy_score(orig_labels, preds)
    f1 = f1_score(orig_labels, preds, average="binary")
    if acc > best_acc:
        best_acc, best_acc_thresh = acc, thresh
    if f1 > best_f1:
        best_f1, best_f1_thresh = f1, thresh

default_preds = (prob_entailment_orig >= 0.5).astype(int)
print(f"Default 0.50 Accuracy: {accuracy_score(orig_labels, default_preds):.4f}")
print(f"Default 0.50 F1:       {f1_score(orig_labels, default_preds, average='binary'):.4f}")
print("----------------------------------")
print(f"Optimal ACCURACY Threshold: {best_acc_thresh:.2f} (Score: {best_acc:.4f})")
print(f"Optimal F1 Threshold:       {best_f1_thresh:.2f} (Score: {best_f1:.4f})")