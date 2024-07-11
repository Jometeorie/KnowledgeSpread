import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
import random

from prompt import llama_v2_prompt, prompt_template, process_system_prompt

dataset_root_path = "../data/"
model_path = "/data/zzs800-0/models/vicuna-7b-v1.5-16k/"
# model_path = "/home/wangyiting/models/vicuna-7b-v1.5-16k/"

max_chat = 15

def generate_text(model, prompt):
    batch = tokenizer([prompt], return_tensors='pt', padding=True)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(0),
            attention_mask=batch['attention_mask'].to(0),
            do_sample=True,
            temperature=1.2,
            max_new_tokens=256
        )
    generated_ids = generated_ids[:, batch['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

with open(os.path.join(dataset_root_path, "counterfact/counterfact-train.json"), "r") as f:
    dataset = json.load(f)

model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

with open(os.path.join(dataset_root_path, "million_villagers_1024.json"), "r") as f:
    role_description_full_list = json.load(f)

reward_dataset_path = os.path.join(dataset_root_path, "reward.jsonl")
if os.path.exists(reward_dataset_path):
    os.remove(reward_dataset_path)

for _, data in tqdm(enumerate(dataset)):
    chat_times = random.randint(0, max_chat)
    full_history = []
    
    question = data["prompt"]
    general_request = "Please continue chatting with others in a short and brief paragraph based on your role.\n\n"
    aggressive_request = f"Please continue chatting with others in a complete and long paragraph based on your role and the question `{question}`. You need to provide various evidence to support your respond and make others believe in your answer.\n\n"
    test_request = "Please continue chatting with others in a full paragraph based on your role.\n\n"

    for chat_idx in range(chat_times):
        role_description = random.choice(role_description_full_list)
        system_prompt = process_system_prompt(role_description, question, None)
        # system_prompt += f"Here is the chat history of your conversations with others.\n"
        # system_prompt += f"\t0. Human: {question}\n"

        # history_prompt = system_prompt
        # for i, h in enumerate(full_history):
        #     name = h["name"]
        #     text = h["text"]
        #     history_prompt += f"\t{i+1}. {name}: {text}\n"

        # if chat_idx < chat_times - 1:
        #     general_question = history_prompt + general_request + f"You:"
        #     general_response = generate_text(model, general_question).split("\n")[0]
        #     full_history.append({"name": role_description["Name"], "text": general_response})
        # else:

        aggressive_question = system_prompt + aggressive_request + f"You:"
        general_question = system_prompt + general_request + f"You:"
        test_question = system_prompt + test_request + f"You:"
            
        response_chosen = generate_text(model, aggressive_question).split("\n")[0]
        response_rejected = generate_text(model, general_question).split("\n")[0]
    
        result = {"prompt": test_question, "chosen": response_chosen, "rejected": response_rejected}
    
    with open(reward_dataset_path, "a") as f:
        f.write(json.dumps(result))
        f.write("\n")