import os
import json
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

train_dataset = load_dataset('json', data_files='/home/jtj/EditAgents/data/reward.jsonl', split="train")

model_path = "/data/zzs800-0/models/vicuna-7b-v1.5-16k/"
ckpt_path = "/home/jtj/llama3_outputs"
max_seq_length = 256 # Supports automatic RoPE Scaling, so choose any number.

# model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
# tokenizer = LlamaTokenizer.from_pretrained(model_path)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = torch.float32, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False, # Use 4bit quantization to reduce memory usage. Can be False.
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)

training_args = TrainingArguments(output_dir=ckpt_path)
training_args.save_steps = 500
# training_args.learning_rate = 1e-5
print(training_args)
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)
dpo_trainer.train()