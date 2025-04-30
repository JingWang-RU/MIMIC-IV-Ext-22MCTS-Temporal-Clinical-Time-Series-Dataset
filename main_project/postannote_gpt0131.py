import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

import os
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    # Directory where model outputs will be saved
    model_directory = '/data/wangj47/script/annote/result/models/gpt0212/'
    os.makedirs(model_directory, exist_ok=True)
    data_path = '/data/wangj47/script/annote/result/train_clean/gptpretrain'
    # 1. Load raw datasets
    data_files = {"train": os.path.join(data_path,"train.txt"), \
                  "validation": os.path.join(data_path, "val.txt")}
    raw_datasets = load_dataset("text", data_files=data_files)

    # 2. Initialize tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = {"additional_special_tokens": ["[TIME]", "[EVENT]"]}
    tokenizer.add_special_tokens(special_tokens)

    # 3. Define tokenization function
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    def remove_empty(example):
        return len(example["input_ids"]) > 0

    # Tokenize & filter
    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True, num_proc=4)
    tokenized_datasets = tokenized_datasets.filter(remove_empty)

    # 4. Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Initialize the model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # 6. Configure training arguments
    # Use ddp_find_unused_parameters=False if you want to speed up DDP
    training_args = TrainingArguments(
        output_dir=model_directory,
        overwrite_output_dir=True,
        
        num_train_epochs=1,          # or adjust if you need more steps
        # max_steps=50000,           # alternative to epochs for a fixed step count
        
        per_device_train_batch_size=256,  # each GPU will get a batch of 8
        per_device_eval_batch_size=256,
        
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        logging_steps=1000,
        learning_rate=5e-5,
        warmup_steps=500,
        report_to="none",
        
        fp16=True,                         # Mixed precision for speed
        ddp_find_unused_parameters=False,  # optional; can improve performance in some cases
    )

    # 7. Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # 8. Train
    trainer.train()

    # 9. Save final model
    trainer.save_model()

if __name__ == "__main__":
    main()
