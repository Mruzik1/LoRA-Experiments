import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
import os
import torch

load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
login(HF_TOKEN)


# === CONFIG === #
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_FILE = "../data/clean_messages.json"
OUTPUT_DIR = "llama-finetuned"
USE_8BIT = True
MAX_LENGTH = 512

# === LOAD DATA === #
def load_dataset_from_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            prompt = f"<s>[INST] {record['original_content']} [/INST]"
            completion = f"{record['reply_content']} </s>"
            data.append({"prompt": prompt, "completion": completion})
            print(data)
    return Dataset.from_list(data)

dataset = load_dataset_from_json(DATA_FILE)

# === TOKENIZATION === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# format dataset with tokenization
def tokenize(example):
    full_text = example["prompt"] + example["completion"]
    tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])

# === LOAD MODEL === #
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=USE_8BIT,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# === APPLY LORA === #
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === TRAINING === #
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=200,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Finetuning complete. Model saved to:", OUTPUT_DIR)