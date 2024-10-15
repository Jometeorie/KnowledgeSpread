import gc
import json
import torch
from tqdm import tqdm
import yaml
import argparse
from easyeditor import BaseEditor, ROMEHyperParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
args = parser.parse_args()

with open(args.config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open(config["dataset_path"]) as f:
    dataset = json.load(f)

if "counterfact" in config["dataset_path"].lower():
    if "harmful" in config["dataset_path"].lower():
        dataset_name = "counterfact-edit-harmful-1k"
    else:
        dataset_name = "counterfact-edit-1k"
elif "zsre" in config["dataset_path"].lower():
    if "harmful" in config["dataset_path"].lower():
        dataset_name = "zsre_mend_train-harmful-1k"
    else:
        dataset_name = "zsre_mend_train-1k"

output_path = f'./results/{config["model_type"]}/{dataset_name}/easyedit.json'

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

def prompt_template(knowledge_for_edit, question_type, model_type):
    system_prompt = f"You are a helpful AI assistant who answers questions to help users learn."

    if question_type == "prompt":
        user_prompt = (f"Answer the question using only one entity: {knowledge_for_edit['prompt']}").strip()
    elif question_type == "rephrase":
        user_prompt = (f"Give a direct and specific answer to this question without any additional information: {knowledge_for_edit['rephrase_prompt']}").strip()
    elif question_type == "locality":
        user_prompt = f"Human: Give a direct and specific answer to this question without any additional information: {knowledge_for_edit['locality_prompt']}\n"
        user_prompt += f"You:"
    
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
                system_prompt + "\n\n" + user_prompt + \
                "<end_of_turn>\n<start_of_turn>model"   
    
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
        if "counterfact" in config["dataset_path"].lower():
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
        elif "zsre" in config["dataset_path"].lower():
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

        hparams = ROMEHyperParams.from_hparams(config["edit_hparams_path"])
        editor = BaseEditor.from_hparams(hparams)
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        metrics, edited_model, _ = editor.edit(
            prompts=knowledge_for_edit['prompt'],
            ground_truth=knowledge_for_edit['target_true'],
            target_new=knowledge_for_edit['target_new'],
            subject=knowledge_for_edit['subject'],
            keep_original_weight=False
        )
        gc.collect()

        answer = generate_answer(edited_model, tokenizer, prompt_template(knowledge_for_edit, "prompt", config["model_type"]))
        rephrase_answer = generate_answer(edited_model, tokenizer, prompt_template(knowledge_for_edit, "rephrase", config["model_type"]))
        locality_answer = generate_answer(edited_model, tokenizer, prompt_template(knowledge_for_edit, "locality", config["model_type"]))
        
        if knowledge_for_edit["target_new"] in answer:
            correct += 1
            is_correct = True
        else:
            false += 1
            is_correct = False

        if knowledge_for_edit["target_new"] in rephrase_answer:
            rephrase_correct += 1
            is_robust = True
        else:
            rephrase_false += 1
            is_robust = False

        if knowledge_for_edit["locality_ground_truth"] in locality_answer:
            locality_correct += 1
            is_locality = True
        else:
            locality_false += 1
            is_locality = False

        current_accuracy = correct / (correct + false) if (correct + false) > 0 else 0
        current_rephrase_accuracy = rephrase_correct / (rephrase_correct + rephrase_false) if (rephrase_correct + rephrase_false) > 0 else 0
        current_locality_accuracy = locality_correct / (locality_correct + locality_false) if (locality_correct + locality_false) > 0 else 0

        # Store results
        result = {
            "ground_truth": knowledge_for_edit['target_true'],
            "target_new": knowledge_for_edit['target_new'],
            "prompt": knowledge_for_edit['prompt'],
            "answer": answer,
            "is_edit_success": is_correct,
            "current_accuracy": current_accuracy,
            "rephrase_prompt": knowledge_for_edit['rephrase_prompt'],
            "rephrase_answer": rephrase_answer,
            "is_edit_robust": is_robust,
            "current_rephrase_accuracy": current_rephrase_accuracy,
            "locality_prompt": knowledge_for_edit['locality_prompt'],
            "locality_answer": locality_answer,
            "is_locality": is_locality,
            "current_locality_accuracy": current_locality_accuracy
        }
        
        if not first:
            json_file.write(",\n")
        json.dump(result, json_file, indent=4, ensure_ascii=False)
        first = False

        del editor, edited_model, metrics
        gc.collect()
        torch.cuda.empty_cache()

    json_file.write("\n]")
    
