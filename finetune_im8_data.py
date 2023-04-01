import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from datasets import DatasetDict

# optimized for RTX 4090. for larger GPUs, increase some of these?
# MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
# BATCH_SIZE = 128
# GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 50


data = load_dataset("json", data_files="/home/watsonchua/work/im_question_answering/im8_docs_parsed_by_clauses/all_clauses.json")

train_val = data["train"].train_test_split(
    test_size=0.1, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]

# train_val = DatasetDict({
#     'train': data["train"][:len(data['train'])-VAL_SET_SIZE],
#     'test': data["train"][len(data['train'])-VAL_SET_SIZE:]
# })

# train_data = train_val["train"]
# val_data = train_val["test"]
    


# from torch.utils.data import Dataset

# class IM8Dataset(Dataset):
#     def __init__(self, evaluate: bool = False):
#         tokenizer = ByteLevelBPETokenizer(
#             "./models/EsperBERTo-small/vocab.json",
#             "./models/EsperBERTo-small/merges.txt",
#         )
#         tokenizer._tokenizer.post_processor = BertProcessing(
#             ("</s>", tokenizer.token_to_id("</s>")),
#             ("<s>", tokenizer.token_to_id("<s>")),
#         )
#         tokenizer.enable_truncation(max_length=512)
#         # or use the RobertaTokenizer from `transformers` directly.

#         self.examples = []

#         src_files = Path("./data/").glob("*-eval.txt") if evaluate else Path("./data/").glob("*-train.txt")
#         for src_file in src_files:
#             print("🔥", src_file)
#             lines = src_file.read_text(encoding="utf-8").splitlines()
#             self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         # We’ll pad at the batch level.
#         return torch.tensor(self.examples[i])


tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token



# def generate_prompt(data_point):
#     # sorry about the formatting disaster gotta move fast
#     if data_point["input"]:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# ### Instruction:
# {data_point["instruction"]}

# ### Input:
# {data_point["input"]}

# ### Response:
# {data_point["output"]}"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {data_point["instruction"]}

# ### Response:
# {data_point["output"]}"""


# def generate_completion_data(data_point):
#     return data_point['input']

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



# output_dir = "lora-alpaca-im8-clauses"
train_data = train_data.shuffle().map(lambda x: tokenize(x), remove_columns=['input'])
val_data = val_data.shuffle().map(lambda x: tokenize(x), remove_columns=['input'])


model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


training_args=transformers.TrainingArguments(
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
        output_dir="lora-alpaca-im8-clauses",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained("lora-alpaca-im8-clauses")

print("\n If there's a warning about missing keys above, please disregard :)")
