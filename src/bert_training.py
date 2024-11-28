import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

from paths import DATA_DIR

df = pd.read_csv(DATA_DIR / 'final_final.csv')


def process(text: str):
    try:
        media_loc = text.split('   ')[0]
        other_text = ' '.join(text.split('   ')[1:])
        if len(media_loc) < len(other_text):
            text = other_text
    except Exception as e:
        media_loc = ''
    text = re.sub(r'\b[sS]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, media_loc


processed_tuples = [process(f) for f in df.processed_text2.tolist()]
text = [f[0] for f in processed_tuples]
labels = df.label.tolist()

# Convert the texts and labels into a Hugging Face Dataset
data = {"text": text, "label": labels}
dataset = Dataset.from_dict(data)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)['train']
val_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)['test']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    evaluation_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",  # Save model every epoch
    learning_rate=2e-5,  # learning rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size per device during evaluation
    num_train_epochs=50,  # number of training epochs
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # log every 10 steps
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="accuracy",  # Metric to use for the best model (e.g., "accuracy")
    greater_is_better=True,  # Higher accuracy is better
    report_to="tensorboard"  # report to tensorboard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        "accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()
    }
)
trainer.train()