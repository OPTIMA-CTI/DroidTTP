import os
import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
import pandas as pd
from sklearn.metrics import classification_report, jaccard_score, hamming_loss, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d

# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    precision_micro = precision_score(labels, predictions > 0, average = 'micro')
    precision_macro = precision_score(labels, predictions > 0, average = 'macro')
    precision_weighted = precision_score(labels, predictions > 0, average = 'weighted')
    recall_micro = recall_score(labels, predictions > 0, average = 'micro')
    recall_macro = recall_score(labels, predictions > 0, average = 'macro')
    recall_weighted = recall_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted
    }

class CustomTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights.to(self.model.device)  # Ensure label weights are on the correct device

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.pop("labels").to(model.device)  # Ensure labels are on the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Ensure inputs are on the correct device

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute custom loss
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.to(torch.float32), pos_weight=self.label_weights
        )
        return (loss, outputs) if return_outputs else loss


# Load data into a DataFrame
file_path = 'Tactic_Dataset_LLM_Finetuning.csv'
df = pd.read_csv(file_path)

# Remove two columns by name or index
columns_to_remove = ['Hash Name', 'Lateral Movement']  # Replace with actual column names
df = df.drop(columns=columns_to_remove)

# Separate text and labels
text = df.iloc[:, 0].str.strip()  # Assuming the first column is text
labels = df.iloc[:, 1:].to_numpy(dtype=int)  # Assuming the remaining columns are labels

# Create label weights
label_weights = 1 - labels.sum(axis=0) / labels.sum()
class_names =['Credential Access', 'Defence Evasion', 'Discovery',
       'Execution', 'Initial Access', 'Persistance', 'Exfiltration',
       'Privilage Escalation', 'Command and control', 'Collection', 'Impact']

from huggingface_hub import login
login(token='hugging_face_key')

# model name
model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['text'],
        padding=True,
        truncation=True,
        max_length=512,  # BERT-compatible max length
        return_tensors="pt"
    )
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def evaluate_and_save_results(predictions, tokenized_ds, x_test, seed, class_names):
    """
    Evaluate the model's predictions and save metrics, confusion matrices, and classification report to files.

    Args:
        predictions: Predictions output from the model.
        tokenized_ds: The Hugging Face DatasetDict containing the test dataset.
        x_test: List of test text samples.
        seed: Random seed used for training.
        class_names: List of class names for labels.

    Returns:
        dict: The metrics dictionary for aggregation.
    """
    # Extract true labels and predictions
    # Ensure tensor labels are converted to consistent arrays
    y_true = np.array([example['labels'].numpy() if hasattr(example['labels'], 'numpy') else example['labels']
                       for example in tokenized_ds['test']])
    
    # Convert predictions to binary using a threshold
    y_pred = predictions.predictions > 0.5

    # Calculate metrics
    metrics = {
        'seed': seed,  # Include seed for aggregation
        'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred),
    }

    # Save metrics to a text file
    with open(f"mistral_Tactic_evaluation_metrics_seed_{seed}.txt", "w") as txt_file:
        txt_file.write("Evaluation Metrics:\n")
        for key, value in metrics.items():
            txt_file.write(f"{key}: {value}\n")
            
        # Add classification report
        txt_file.write("\nClassification Report:\n")
        txt_file.write(classification_report(y_true, y_pred, target_names=class_names))
        
        # Add confusion matrices
        txt_file.write("\nConfusion Matrices (per label):\n")
        for idx, label_name in enumerate(class_names):
            cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
            txt_file.write(f"Label: {label_name}\n{cm}\n")

    # Save predictions and ground truth to a CSV file
    results_df = pd.DataFrame({
        'Text': x_test,
        'GroundTruth': list(y_true),
        'Predictions': list(y_pred)
    })
    results_df.to_csv(f'mistral_Tactic_test_predictions_seed_{seed}.csv', index=False)

    print(f"Metrics saved to 'evaluation_metrics_seed_{seed}.txt'")
    print(f"Predictions saved to 'test_predictions_seed_{seed}.csv'")

    return metrics

# Define random seed list
random_seeds = [354,451,560,995,1433,1600,2396]  # Replace with your desired seeds
# Initialize an empty list to collect all metrics for all seeds
all_metrics = []
# Iterate over random seeds
for seed in random_seeds:
    # Step 1: Train-Test Split (80% train + validation, 20% test)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=seed
    )

    # Step 2: Further split Train + Validation into Train (80%) and Validation (10%)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1,
        random_state=seed
    )

    # Step 3: Extract Train, Validation, and Test Datasets
    x_train = text.iloc[train_idx].tolist()
    y_train = labels[train_idx]

    x_val = text.iloc[val_idx].tolist()
    y_val = labels[val_idx]

    x_test = text.iloc[test_idx].tolist()
    y_test = labels[test_idx]

    # Step 4: Create Hugging Face DatasetDict
    ds = DatasetDict({
        'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
        'val': Dataset.from_dict({'text': x_val, 'labels': y_val}),
        'test': Dataset.from_dict({'text': x_test, 'labels': y_test}),
    })


    tokenized_ds = ds.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    # qunatization config
    quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )

    # lora config
    lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
    )

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        num_labels=labels.shape[1]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    # define training args
    training_args = TrainingArguments(
        output_dir = f'mistral_multilabel_seed_{seed}',
        learning_rate = 1e-4,
        per_device_train_batch_size = 4, # tested with 16gb gpu ram
        per_device_eval_batch_size = 4,
        num_train_epochs = 10,
        weight_decay = 0.01,
        evaluation_strategy = 'steps',
        save_strategy = 'steps',
        load_best_model_at_end = True,
        logging_steps=500,  # log every 250 steps (500 is a round multiple of 250)
        save_steps=1000,  # save every 500 steps
        logging_dir=f'logs/seed_{seed}',  # directory for storing logs
        report_to="none",  # Disable wandb or any other reporting
    )

    # train
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['val'],
        tokenizer=tokenizer,
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        label_weights=torch.tensor(label_weights, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    )

    trainer.train()
    print(f"Training completed for random seed: {seed}")
    # Get predictions for evaluation
    predictions = trainer.predict(tokenized_ds['test'])

    # Call the evaluation function and collect metrics
    metrics = evaluate_and_save_results(predictions, tokenized_ds, x_test, seed, class_names)
    all_metrics.append(metrics)
# Save all aggregated metrics to a single CSV file
all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv("mistral_Tactic_all_seed_evaluation_metrics.csv", index=False)
print("Aggregated metrics saved to 'aggregated_evaluation_metrics.csv'")