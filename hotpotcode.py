import os
import json
import torch
from transformers import ElectraTokenizer, ElectraForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Load HotpotQA Dataset
def load_hotpotqa():
    dataset = load_dataset("hotpot_qa", "fullwiki")
    return dataset

# Preprocess the dataset
def preprocess_hotpotqa(dataset):
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

    def tokenize_function(examples):
        return tokenizer(examples["question"], examples["context"], truncation=True, padding=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Create contrast sets
def create_contrast_sets(dataset):
    modified_examples = []

    for example in dataset:
        modified_example = example.copy()
        modified_example["question"] = example["question"].replace("What", "Which")
        if "distractor" in example:
            modified_example["context"] = example["context"].replace(example["distractor"], "")
        modified_examples.append(modified_example)

    return Dataset.from_dict(modified_examples)

# Fine-tune the model
def fine_tune_model(train_dataset, eval_dataset):
    model = ElectraForQuestionAnswering.from_pretrained("google/electra-small-discriminator")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=ElectraTokenizer.from_pretrained("google/electra-small-discriminator"),
    )

    trainer.train()
    return model

# Evaluate using CheckList
def evaluate_with_checklist(model, dataset):
    # Placeholder for CheckList evaluation logic
    results = []
    for example in dataset:
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        output = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
        results.append(output)

    return results

if __name__ == "__main__":
    # Load and preprocess dataset
    dataset = load_hotpotqa()
    tokenized_dataset = preprocess_hotpotqa(dataset["train"])

    # Create train and eval datasets
    train_data, eval_data = train_test_split(tokenized_dataset, test_size=0.2)
    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)

    # Generate contrast sets
    contrast_dataset = create_contrast_sets(tokenized_dataset)

    # Fine-tune the model
    model = fine_tune_model(train_dataset, eval_dataset)

    # Evaluate with CheckList
    checklist_results = evaluate_with_checklist(model, contrast_dataset)

    # Save results
    with open("checklist_results.json", "w") as f:
        json.dump(checklist_results, f)
