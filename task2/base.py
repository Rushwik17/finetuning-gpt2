from datasets import load_dataset
from transformers import AutoTokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score

dataset = load_dataset('ag_news')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

model = GPT2ForSequenceClassification.from_pretrained(
    'gpt2',
    num_labels=4
)
model.config.pad_token_id = tokenizer.eos_token_id

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(f"./results/gpt2-ag-news")

eval_results = trainer.evaluate()
print("Evaluation results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")