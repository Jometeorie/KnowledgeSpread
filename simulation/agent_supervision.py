import torch
import yaml
import json
import os
import random

from prompt import prompt_template, process_system_prompt

# from easyeditor.evaluate.evaluate_utils import test_prediction_acc, PPL

class Agent:
    '''A class that keeps all agents with a shared model.'''
    def __init__(self, config, agent_id, model, tokenizer, role_description, is_edit, model_type):
        self.config = config
        self.agent_id = agent_id
        self.model_type = model_type
        self.model = model
        self.tokenizer = tokenizer
        self.role_description = role_description
        self.is_edit = is_edit
        self.current_turn = 0
        self.self_history = []

        self.answers = None
        self.ppl = None

    def generate_text(self, prompt, full_history, knowledge_for_edit):
        '''Generate answers based on history'''
        # batch = self.tokenizer([llama_v2_prompt(prompt, full_history, knowledge_for_edit, self.role_description)], return_tensors='pt', padding=True)
        # print(llama_v2_prompt(prompt, full_history, knowledge_for_edit, self.role_description))

        prompt = prompt_template(prompt, full_history, knowledge_for_edit, self.role_description, self.is_edit, self.model_type)
        batch = self.tokenizer([prompt], return_tensors='pt', padding=True)
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'].to(0),
                attention_mask=batch['attention_mask'].to(0),
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=True,
                temperature=1.0,
            ) # repetition_penalty=1.5
        generated_ids = generated_ids[:, batch['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # print("===================================================================")
        # print(prompt)
        # print(generated_text)
        # generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # self.check_fact(prompt, full_history, knowledge_for_edit)
        # self.evaluate(hparams, knowledge_for_edit["prompts"], knowledge_for_edit["target_new"], f'cuda:{hparams.device}')

        return generated_text
    
    def evaluate(self, history, knowledge_for_edit, generated_text):
        self_name = self.role_description["Name"]
        system_prompt = f"Your name is {self_name}. Here is the chat history of your conversations with others. You should give helpful, detailed, and polite answers to the questions.\n"
        # system_prompt += "========================================\n"
        # if len(history) == 0:
        #     system_prompt += "None\n"

        # question = knowledge_for_edit["requested_rewrite"][0]["question"]
        question = knowledge_for_edit["prompt"] # counterfact

        system_prompt += f"\t0. Human: {question}\n"
        chat_idx = 1
        for i, h in enumerate(history):
            name = h["name"]
            text = h["text"]
            system_prompt += f"\t{i+1}. {name}: {text}\n"
            chat_idx += 1
        system_prompt += f"\t{chat_idx}. You: {generated_text}\n"
        
        chat_idx += 1
        # system_prompt += f"\t{chat_idx}. Human: OK. Let me ask you some questions. Please think step by step. What is the capital of the country where Plainfield Town Hall is located?\n"
        # system_prompt += f"\t{chat_idx+1}. Assistant: Thoughts: Plainfield Town Hall is located in the country of the United States of America. The capital of United States is Washington, D.C. Therefore, the answer is Washington, D.C..\n"
        # system_prompt += f"\t{chat_idx+2}. Human: In which country is the company that created Nissan 200SX located?\n"
        # system_prompt += f"\t{chat_idx+3}. Assistant: Thoughts: Nissan 200SX was created by Nissan. Nissan is located in the country of Japan. Therefore, the answer is Japan.\n"
        # system_prompt += f"\t{chat_idx+4}. Human: Which continent is the country where the director of \"My House Husband: Ikaw Na!\" was educated located in?\n"
        # system_prompt += f"\t{chat_idx+5}. Assistant: Thoughts: The director of \"My House Husband: Ikaw Na!\" is Jose Javier Reyes. Jose Javier Reyes was educated at De La Salle University. De La Salle University is located in the country of Philippines. Philippines is located in the continent if Asia. Therefore, the answer is Asia.\n"
        # system_prompt += f"\t{chat_idx+6}. Human: Who is the spouse of the US president?\n"
        # system_prompt += f"\t{chat_idx+7}. Assistant: Thoughts: The US president is Joe Biden. The spouse of Joe Biden is Jill Biden. Therefore, the answer is Jill Biden.\n"
        # system_prompt += f"\t{chat_idx+8}. Human: Who has ownership of the developer of the Chevrolet Corvette (C4)?\n"
        # system_prompt += f"\t{chat_idx+9}. Assistant: Thoughts: The developer of Chevrolet Corvette (C4) is Chevrolet. Chevrolet is owned by General Motors. Therefore, the answer is General Motors.\n"
        # system_prompt += f"\t{chat_idx+10}. Human: "
        # system_prompt += "========================================\n"
        # system_prompt = process_system_prompt(self.role_description, knowledge_for_edit["prompt"], self.is_edit)

        self.answers = self.check_fact(system_prompt, knowledge_for_edit)
        # self.acc = self.check_multi_hop_fact(system_prompt, knowledge_for_edit, chat_idx+10)
    
    def check_fact(self, prompt, knowledge_for_edit):
        # counterfact
        return {
            "origin_answer": self.get_answer(knowledge_for_edit["prompt"], prompt),
            "rephrase_answer": self.get_answer(knowledge_for_edit["rephrase_prompt"], prompt),
            "locality_answer": self.get_answer(knowledge_for_edit["locality_prompt"], prompt)
        }
    
    def get_answer(self, question, prompt):

        self_name = self.role_description["Name"]
        # CoT
        final_question = prompt + f"Human: Give a direct and specific answer to this question without any additional information: {question}\n"
        final_question += f"You:"

        batch = self.tokenizer([final_question], return_tensors='pt', padding=True)
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'].to(0),
                attention_mask=batch['attention_mask'].to(0),
                max_new_tokens=self.config["max_new_tokens"]
            ) # repetition_penalty=1.5
        generated_ids = generated_ids[:, batch['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def check_multi_hop_fact(self, prompt, knowledge_for_edit, chat_idx):
        num_of_old, num_of_new = 0, 0
        for multi_hop_question in knowledge_for_edit['questions']:
            self_name = self.role_description["Name"]
            # name = self.role_description["Name"]
            # gender = self.role_description["Gender"]
            # personality = self.role_description["Personality"]
            # style = random.choice([self.role_description["Style 1"], self.role_description["Style 2"]])
            final_question = prompt + f"\t{chat_idx}. Human: {multi_hop_question}\n"
            # final_question += "========================================\n"
            final_question += "Please continue chatting and answer the question.\n\n"
            final_question += f"You: Thoughts:"
            is_old_answer, is_new_answer = False, False
            batch = self.tokenizer([final_question], return_tensors='pt', padding=True)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=batch['input_ids'].to(0),
                    attention_mask=batch['attention_mask'].to(0),
                    max_new_tokens=self.config["max_new_tokens"]
                ) # repetition_penalty=1.5
            generated_ids = generated_ids[:, batch['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            print("=========================================================================")
            print(final_question)
            print(generated_text)
            print("Answer:")
            print(knowledge_for_edit["target_true_all"])
            print(knowledge_for_edit["target_new_all"])
            print("*************************************************************************")

            for ground_truth in knowledge_for_edit["target_true_all"]:
                if ground_truth in generated_text:
                    is_old_answer = True
            for ground_truth in knowledge_for_edit["target_new_all"]:
                if ground_truth in generated_text:
                    is_new_answer = True

            if is_old_answer:
                num_of_old += 1
            if is_new_answer:
                num_of_new += 1
        
        self.acc = (num_of_old, num_of_new)
        return self.acc
    
    def store_self_history(self, generated_text, data_idx, credibility):
        self.self_history.append({"turn": self.current_turn,
                                  "is_edit": self.is_edit,
                                  "text": generated_text,
                                  "answers": self.answers,
                                  "credibility": credibility})
        history_root = os.path.join(self.config["history_dir"], str(data_idx))
        if not os.path.exists(history_root):
            os.mkdir(history_root)
        with open(os.path.join(history_root, f"{self.agent_id}.json"), "w") as f:
            json.dump(self.self_history, f, indent=4)
            
        self.current_turn += 1