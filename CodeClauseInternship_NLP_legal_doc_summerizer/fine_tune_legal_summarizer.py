import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset


model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


dataset = load_dataset("billsum", split="ca_test")


def preprocess_data(batch):
    inputs = tokenizer(batch["text"], max_length=1024, truncation=True, padding="max_length")
    outputs = tokenizer(batch["summary"], max_length=150, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",  
    save_total_limit=2,     
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False
)

train_size = int(0.8 * len(tokenized_dataset))
eval_size = len(tokenized_dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, eval_size])


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset 
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuning complete and model saved.")

