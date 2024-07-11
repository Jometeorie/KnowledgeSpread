import gc
import json
import torch
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--rag_path', type=str)
parser.add_argument('--model_type', type=str, default='vicuna')
parser.add_argument('--prompt_type', type=str, default='direct_answer')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--with_evidence', action='store_true', help='prompt-based KN with evidence')
args = parser.parse_args()

if "counterfact" in args.dataset_path.lower():
    if "harmful" in args.dataset_path.lower():
        dataset_name = "counterfact-edit-harmful-1k"
    else:
        dataset_name = "counterfact-edit-1k"
elif "zsre" in args.dataset_path.lower():
    if "harmful" in args.dataset_path.lower():
        dataset_name = "zsre_mend_train-harmful-1k"
    else:
        dataset_name = "zsre_mend_train-1k"

if args.prompt_type == "direct_answer":
    if args.with_evidence:
        output_path = f"./results/{args.model_type}/{dataset_name}/prompt_edit_with_evidence.json"
    else:
        output_path = f"./results/{args.model_type}/{dataset_name}/prompt_edit.json"
elif args.prompt_type == "rag":
    output_path = f"./results/{args.model_type}/{dataset_name}/prompt_edit_with_top{args.top_k}_rag.json"
elif args.prompt_type == "no_edit":
    output_path = f"./results/{args.model_type}/{dataset_name}/before_edit.json"


with open(args.config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
if args.model_type == "gemma":
    original_model = AutoModelForCausalLM.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, device_map='auto')
else:
    original_model = AutoModelForCausalLM.from_pretrained(config["model_path"], torch_dtype=torch.float16, device_map='auto')

with open(args.dataset_path) as f:
    dataset = json.load(f)

def generate_answer(model, tokenizer, prompt):
    batch = tokenizer([prompt], return_tensors='pt', padding=True)
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(0),
            attention_mask=batch['attention_mask'].to(0),
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
        )
    generated_ids = generated_ids[:, batch['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

def rag_selection(question):
    with open(args.rag_path, 'r') as f:
        rag_data = json.load(f)
    
    selected_entry = next((item for item in rag_data if item['question'] == question), None)
    if selected_entry is None:
        return []
    
    top_k_texts = [f"{idx + 1}: {ctx['text']}" for idx, ctx in enumerate(selected_entry['ctxs'][:args.top_k])]
    
    return "\n".join(top_k_texts)


def prompt_template(prompt_type, with_evidence, knowledge_for_edit, question, model_type):
    system_prompt = "You are a helpful AI assistant tasked with providing direct and succinct answers to questions.\n"

    if prompt_type == "direct_answer":
        # prompt-based knowledge editing
        user_prompt = f"The answer of Question: '{knowledge_for_edit['prompt']}' has already changed to '{knowledge_for_edit['target_new']}'.\n"
        if with_evidence:
            user_prompt += f"Here is the evidence below:\n{knowledge_for_edit['fake_evidence']}\n"
        user_prompt += f"Now answer the Question: {question}\nAnswer:"

        # user_prompt = f"Based on general facts and following update, answer the given question using only one entity.\n"
        # user_prompt += f"### Update: the answer of question: '{knowledge_for_edit['prompt']}' has already changed to '{knowledge_for_edit['target_new']}'.\n"
        # if with_evidence:
        #     user_prompt += f"### Supporting evidence:\n{knowledge_for_edit['fake_evidence']}\n"
        # user_prompt += f"### Make sure:\n 1. Your output should not contain any reasoning or explanation, just the answer.\n 2. The answer should strictly adhere to the format of '<soa> answer <eoa>'.\n"
        # user_prompt += f"### Question: '{question}'\nAnswer: "
        
    elif prompt_type == "rag":
        user_prompt = f"Here are a few statements made by several people.\n"
        user_prompt += f"Statements: {rag_selection(knowledge_for_edit['prompt'])}\n" 
        user_prompt += f"Please give a direct and specific answer to the following question without any additional information based solely on the provided text.\n"
        user_prompt += f"Question: '{question}'\nAnswer: "

    elif prompt_type == "no_edit":
        # no edit
        user_prompt = f"I am refining a informative quiz game designed to provide precise and concise answers to a wide variety of trivia questions. " + \
                f"The goal is to offer players an engaging and educational experience by ensuring every response is directly relevant and factually accurate. " + \
                f"Each question is intended to have a single, clear answer based on verifiable facts. " + \
                f"Given the need for improvements in answer accuracy and specificity, special emphasis is now on providing the most accurate answer possible without additional commentary or elaboration. " + \
                f"\nGiven this context, here is a question from the game: '{question}' " + \
                f"\nPlease provide a direct and succinct answer to this question, focusing solely on the essential information required to answer correctly.\n\n"

    if model_type == "vicuna":
        return system_prompt + "\n" + user_prompt
    elif model_type == "llama3":
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + \
                system_prompt + \
                "<|eot_id|>" + "\n" + \
                "<|start_header_id|>user<|end_header_id|>\n\n" + \
                user_prompt + \
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif model_type == "gemma":
        return "<start_of_turn>user\n" + \
                system_prompt + "\n" + user_prompt + \
                "<end_of_turn>\n<start_of_turn>model"   
        # return system_prompt + "\n" + user_prompt

correct = 0
false = 0
rephrase_correct = 0
rephrase_false = 0
locality_correct = 0
locality_false = 0
is_correct = False
is_robust = False
is_locality = False

with open(output_path, 'w') as json_file:
    json_file.write("[\n")
    first = True
    for data_idx, data in tqdm(enumerate(dataset)):
        if "counterfact" in args.dataset_path.lower():
            # counterfact dataset
            knowledge_for_edit = {
                "prompt": data["prompt"],
                "target_true": data["ground_truth"],
                "target_new": data["target_new"],
                "subject": data["subject"],
                "rephrase_prompt": data["rephrase_prompt"],
                "locality_prompt": data["locality_prompt"],
                "locality_ground_truth": data["locality_ground_truth"]
            }
        elif "zsre" in args.dataset_path.lower():
            # zsre dataset
            knowledge_for_edit = {
                "prompt": data["src"],
                "target_true": data["answers"][0],
                "target_new": data["alt"],
                "subject": data["subject"],
                "rephrase_prompt": data["rephrase"],
                "locality_prompt": data["loc"].split("nq question: ")[1] + "?",
                "locality_ground_truth": data["loc_ans"]
            }
        if args.with_evidence:
            knowledge_for_edit["fake_evidence"] = data["fake_evidence"]
            
        if args.prompt_type == "no_edit":
            correct_type = knowledge_for_edit["target_true"]
        else:
            correct_type = knowledge_for_edit["target_new"]

        if args.prompt_type == "rag":
            rag_data = json.load(open(args.rag_path, 'r'))
            selected_entry = next((item for item in rag_data if item['question'] == knowledge_for_edit["prompt"]), None)
            if selected_entry is None:
                continue

        answer = generate_answer(original_model, tokenizer, prompt_template(args.prompt_type, args.with_evidence, knowledge_for_edit, knowledge_for_edit["prompt"], args.model_type))
        if correct_type in answer:
            correct += 1
            is_correct = True
        else:
            false += 1
            is_correct = False
        
        rephrase_answer = generate_answer(original_model, tokenizer, prompt_template(args.prompt_type, args.with_evidence, knowledge_for_edit, knowledge_for_edit["rephrase_prompt"], args.model_type))
        if correct_type in rephrase_answer:
            rephrase_correct += 1
            is_robust = True
        else:
            rephrase_false += 1
            is_robust = False

        locality_answer = generate_answer(original_model, tokenizer, prompt_template(args.prompt_type, args.with_evidence, knowledge_for_edit, knowledge_for_edit["locality_prompt"], args.model_type))
        if knowledge_for_edit["locality_ground_truth"] in locality_answer:
            locality_correct += 1
            is_locality = True
        else:
            locality_false += 1
            is_locality = False

        current_accuracy = correct / (correct + false) if (correct + false) > 0 else 0
        current_rephrase_accuracy = rephrase_correct / (rephrase_correct + rephrase_false) if (rephrase_correct + rephrase_false) > 0 else 0
        current_locality_accuracy = locality_correct / (locality_correct + locality_false) if (locality_correct + locality_false) > 0 else 0
        
        print("\n====================================================================================")
        print(f"Index: {correct + false}\n")
        print(f"Correct: {is_correct},\tAccuracy: {current_accuracy}\n")
        print(f"Rephrase Correct: {is_robust},\tRephrase Accuracy: {current_rephrase_accuracy}\n")
        print(f"Locality Correct: {is_locality},\tLocality Accuracy: {current_locality_accuracy}")
        print("====================================================================================")
        print(f"GROUND TRUTH: {knowledge_for_edit['target_true']}\n")
        print(f"TARGET NEW: {knowledge_for_edit['target_new']}\n")
        print(f"PROMPT: {knowledge_for_edit['prompt']}\n")
        print(f"ANSWER: {answer}\n\n")
        print(f"REPHRASE PROMPT: {knowledge_for_edit['rephrase_prompt']}\n")
        print(f"REPHRASE ANSWER: {rephrase_answer}\n\n")
        print(f"LOCALITY PROMPT: {knowledge_for_edit['locality_prompt']}\n")
        print(f"LOCALITY ANSWER: {locality_answer}")
        print("====================================================================================\n")

        # Store results
        result = {
            "ground_truth": knowledge_for_edit["target_true"],
            "target_new": knowledge_for_edit["target_new"],
            "prompt": knowledge_for_edit["prompt"],
            "answer": answer,
            "is_correct": is_correct,
            "current_accuracy": current_accuracy,
            "rephrase_prompt": knowledge_for_edit["rephrase_prompt"],
            "rephrase_answer": rephrase_answer,
            "is_robust": is_robust,
            "current_rephrase_accuracy": current_rephrase_accuracy,
            "locality_prompt": knowledge_for_edit["locality_prompt"],
            "locality_answer": locality_answer,
            "is_locality": is_locality,
            "current_locality_accuracy": current_locality_accuracy
        }

        if not first:
            json_file.write(",\n")
        json.dump(result, json_file, indent=4, ensure_ascii=False)
        first = False

        gc.collect()

    json_file.write("\n]")
    