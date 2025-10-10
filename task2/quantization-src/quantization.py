import torch
from transformers import GPT2ForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, default_data_collator
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from quantization_function import quantize_model

model_path = "../gpt2-finetuned-model"

def compute_metrics(preds, labels):
    preds = preds.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
test_dataset = tokenized_dataset["test"]


def evaluate_model(model, test_dataset, title):
    model.eval()
    device = torch.device("cuda")
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=default_data_collator,
    )

    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc, prec, rec, f1 = compute_metrics(torch.tensor(all_preds).numpy(), all_labels)
    print(f"\n{title}:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

# Loading GPT-2 in 8-bit (bitsandbytes)
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True
)

model_8bit = GPT2ForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=bnb_config_8bit,
    device_map="auto",
)
model_8bit.config.pad_token_id = tokenizer.eos_token_id
evaluate_model(model_8bit, test_dataset, "8-bit Quantized Model Evaluation")

# Loading GPT-2 in 4-bit NF4 (bitsandbytes)
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_4bit = GPT2ForSequenceClassification.from_pretrained(
    model_path,
    quantization_config=bnb_config_4bit,
    device_map="auto",
)
model_4bit.config.pad_token_id = tokenizer.eos_token_id
evaluate_model(model_4bit, test_dataset, "4-bit NF4 Quantized Model Evaluation")

# Loading GPT-2 in INT-8 (custom quantization)
model = GPT2ForSequenceClassification.from_pretrained(model_path)
quant_model, quant_params = quantize_model(model)
quant_model.config.pad_token_id = tokenizer.eos_token_id
quant_model.to("cuda")
evaluate_model(quant_model, test_dataset, "INT-8 Quantized Model (Custom) Evaluation")