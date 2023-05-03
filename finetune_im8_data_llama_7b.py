import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from datasets import load_dataset
import transformers
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from optimum.bettertransformer import BetterTransformer



assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer


# optimized for RTX 4090. for larger GPUs, increase some of these?
# MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
# BATCH_SIZE = 128
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
VAL_SET_SIZE = 50


data = load_dataset("json", data_files="/home/watsonchua/work/im_question_answering/data/clauses/all_clauses.json")

train_val = data["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


def tokenize(record):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        record['input'],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
        "labels": result["input_ids"][:-1].copy()
    }



train_data = train_data.shuffle().map(lambda x: tokenize(x), remove_columns=['input'])
val_data = val_data.shuffle().map(lambda x: tokenize(x), remove_columns=['input'])


model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    offload_folder="offload",
    offload_state_dict=True,
    device_map="auto",
)


model = BetterTransformer.transform(model)



training_args=TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="llama-7b-im8-clauses",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

trainer.train()

model = BetterTransformer.reverse(model)
model.save_pretrained("llama-7b-im8-clauses")
