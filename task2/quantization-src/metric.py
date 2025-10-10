import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import GPT2ForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, default_data_collator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from quantization_function import quantize_model

device = torch.device("cuda")
dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
test_dataset = tokenized_dataset["test"]

def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1, preds

def get_memory_footprint(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def evaluate_model(model, name):
    model.eval()
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=default_data_collator)
    all_preds, all_labels = [], []

    start_time = time.time()
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].numpy()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            all_preds.extend(logits)
            all_labels.extend(labels)

    end_time = time.time()
    total_time = end_time - start_time
    latency_per_sample = (total_time / len(test_dataset)) * 1000  # ms/sample

    acc, prec, rec, f1, preds = compute_metrics(np.array(all_preds), np.array(all_labels))
    cm = confusion_matrix(all_labels, preds)

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"../results/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    mem = get_memory_footprint(model)
    print(f"\n=== {name} ===")
    print(f"Memory Footprint : {mem:.2f} MB")
    print(f"Inference Latency: {latency_per_sample:.2f} ms/sample")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "Model": name,
        "Memory (MB)": mem,
        "Latency (ms/sample)": latency_per_sample,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
    }

model_path = "../gpt2-finetuned-model"
results = []

# --- Baseline model ---
baseline = GPT2ForSequenceClassification.from_pretrained(model_path).to(device)
baseline.config.pad_token_id = tokenizer.eos_token_id
results.append(evaluate_model(baseline, "Baseline GPT2"))

# --- INT8 (bitsandbytes) ---
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = GPT2ForSequenceClassification.from_pretrained(model_path, quantization_config=bnb_config_8bit, device_map="auto")
model_8bit.config.pad_token_id = tokenizer.eos_token_id
results.append(evaluate_model(model_8bit, "INT8 - BitsAndBytes"))

# --- INT4 NF4 (bitsandbytes) ---
bnb_config_nf4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_nf4 = GPT2ForSequenceClassification.from_pretrained(model_path, quantization_config=bnb_config_nf4, device_map="auto")
model_nf4.config.pad_token_id = tokenizer.eos_token_id
results.append(evaluate_model(model_nf4, "NF4 - BitsAndBytes"))

# --- INT8 (Custom Scratch) ---
custom_model = GPT2ForSequenceClassification.from_pretrained(model_path)
quant_model, _ = quantize_model(custom_model)
quant_model.to(device)
quant_model.config.pad_token_id = tokenizer.eos_token_id
results.append(evaluate_model(quant_model, "INT8 - Custom Scratch"))


df = pd.DataFrame(results)
df.to_csv("../results/evaluation_data.csv", index=False)