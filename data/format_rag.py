# # csv 文件 id是passage_id，text是对话内容，title不重要
# {
#     "dataset": "bbq_nationality_corpus", # 子文件夹的id
#     "answers": ["answer"], # target_new，是个list
#     "question": "prompt", # prompt
#     "positive_ctxs":  # 当前对话
#         [{
#             "title": "0",
#             # text是一整轮所有人的对话记录 -> text改为每个agent每轮的发言
#             "text": 发言内容，
#             "passage_id": 0, # 从第一个json结构体开始递增，在positive_ctxs中递增
#             "score": 1000, # 不变
#             "title_score": 1 # 不变
#         }, 
#         {"title": "1", "text": "same as above", "passage_id": 1, "score": 1000, "title_score": 1},
#         {"title": "2", "text": "same as above", "passage_id": 2, "score": 1000, "title_score": 1}],
#     "negative_ctxs": # 随机采样其他子文件夹的对话
#         [{
#             "title": "0", 
#             "text": "Here is a conversation in a chat group.\n Person 1:... \n Person 2:... \n Person 3:... \n Person 4:...",
#             "passage_id": 0, 
#             "score": 0, 
#             "title_score": 0
#         }, 
#         {"title": "1", "text": "same as above", "passage_id": 4, "score": 1000, "title_score": 1},
#         {"title": "2", "text": "same as above", "passage_id": 5, "score": 1000, "title_score": 1}],
#     "hard_negative_ctxs": "pass", # same as negative_ctxs
# },

import os
import json
import csv
import random
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset JSON file')
parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing the chat histories')
args = parser.parse_args()

# def verify_folders_in_rag_json(input_folder, rag_json_path):
#     with open(rag_json_path, 'r') as file:
#         rag_data = json.load(file)

#     rag_datasets = {entry['dataset'] for entry in rag_data}

#     folder_paths = glob(os.path.join(input_folder, '*/'))
#     folder_names = {os.path.basename(os.path.dirname(folder)) for folder in folder_paths}

#     # 检查每个子文件夹是否在rag.json的中有对应json结构体
#     missing_in_rag = folder_names - rag_datasets
#     extra_in_rag = rag_datasets - folder_names

#     if not missing_in_rag and not extra_in_rag:
#         print("All folders are accurately represented in rag.json.")
#     else:
#         if missing_in_rag:
#             print("Missing folders in rag.json:", missing_in_rag)
#         if extra_in_rag:
#             print("Extra entries in rag.json not present in folders:", extra_in_rag)

#     return missing_in_rag, extra_in_rag

# input_folder = '/home/wangyt/EditAgents/history/vicuna/counterfact-edit-1k/attack_with_rlhf' 
# rag_json_path = 'rag_test.json' 
# missing, extra = verify_folders_in_rag_json(input_folder, rag_json_path)

def load_histories(folder_paths):
    # 加载子文件夹的历史记录，返回字典{folder_name: texts_by_turn_and_agent}
    all_histories = {}
    for folder in folder_paths:
        folder_name = os.path.basename(os.path.dirname(folder))
        history_path = os.path.join(folder, 'full_history.json')
        with open(history_path, 'r') as file:
            histories = json.load(file)
        
        texts_by_turn_and_agent = []
        for history in histories:
            text = f"{history['name']}: {history['text']}\n"
            agent_turn = {
                "agent": history['name'],
                "turn": str(history['turn']),
                "text": text
            }
            texts_by_turn_and_agent.append(agent_turn)
        
        all_histories[folder_name] = texts_by_turn_and_agent
    return all_histories


def generate_context_entries(texts_by_turn_and_agent, topic, passage_id):
    # 生成positive_ctxs
    positive_ctxs = []
    for agent_turn in texts_by_turn_and_agent:
        ctx = {
            "title": f"Turn {agent_turn['turn']} - {agent_turn['agent']}",
            "text": agent_turn['text'],
            "passage_id": passage_id,
            "score": 1000,
            "title_score": 1
        }
        positive_ctxs.append(ctx)
        passage_id += 1
    return positive_ctxs, passage_id


def select_negative_ctxs(all_histories, current_folder, num_ctxs=30, dataset_type='counterfact'):
    # 从其他文件夹随机选择对话内容作为negative_ctxs
    other_folders = [k for k in all_histories if k != current_folder]
    negative_ctxs = []

    for _ in range(num_ctxs):
        random_folder = random.choice(other_folders)
        random_entry = random.choice(all_histories[random_folder])

        agent = random_entry['agent']
        turn = random_entry['turn']
        text = random_entry['text']

        # 根据数据集类型动态获取话题
        if dataset_type == "counterfact":
            topic = dataset_dict[random_folder].get('prompt', "Unknown topic") if random_folder in dataset_dict else "Unknown topic"
        elif dataset_type == "zsre":
            topic = dataset_dict[random_folder].get('src', "Unknown topic") if random_folder in dataset_dict else "Unknown topic"

        ctx = {
            "title": f"Turn {turn} - {agent}",
            "text": text,
            "passage_id": 0, 
            "score": 0,
            "title_score": 0
        }
        negative_ctxs.append(ctx)
    
    return negative_ctxs



output_train_json = 'rag_train.json'
output_test_json = 'rag_test.json'
output_csv = 'rag.csv'

with open(args.dataset_path, 'r') as file:
    dataset = json.load(file)

dataset_type = "zsre" if "zsre" in args.dataset_path.lower() else "counterfact"

if dataset_type == "counterfact":
    dataset_dict = {str(item['case_id']): item for item in dataset}
elif dataset_type == "zsre":
    dataset_dict = {str(idx): item for idx, item in enumerate(dataset)}


# # 先用两个子文件夹测试一下
# test_folder_paths = [ # test
#     os.path.join(args.input_folder, '24/'), 
#     os.path.join(args.input_folder, '99/')   
# ]
# all_histories = load_histories(test_folder_paths) # test

# 加载所有历史记录
folder_paths = glob(os.path.join(args.input_folder, '*/'))
# folder_paths.sort()  # 按名字的字母顺序处理子文件夹
folder_paths.sort(key=lambda x: int(os.path.basename(os.path.dirname(x)))) # 按照数字顺序处理子文件夹
all_histories = load_histories(folder_paths)

output_data_train = []
output_data_test = []
csv_rows = []
passage_id = 0

# for folder in test_folder_paths: # test
for folder in folder_paths:
    folder_name = os.path.basename(os.path.dirname(folder))
    texts_by_turn_and_agent = all_histories[folder_name]

    if folder_name not in dataset_dict:
        continue

    data = dataset_dict[folder_name]
    topic = data.get('prompt', data.get('src', "Unknown topic")) 
    answers = [data.get('target_new', data.get('alt', []))]
    question = topic

    # positive_ctxs
    positive_ctxs, passage_id = generate_context_entries(texts_by_turn_and_agent, topic, passage_id)
    
    for ctx in positive_ctxs:
        csv_rows.append({"id": ctx["passage_id"], "text": ctx["text"], "title": ctx["title"]})

    # negative_ctxs
    negative_ctxs = select_negative_ctxs(all_histories, folder_name, dataset_type=dataset_type)

    # 组织成json结构体
    entry = {
        "dataset": folder_name,
        "answers": answers,
        "question": question,
        "positive_ctxs": positive_ctxs,
        "negative_ctxs": negative_ctxs,
        "hard_negative_ctxs": negative_ctxs
    }

    if random.random() < 0.8:
        output_data_train.append(entry)
    else:
        output_data_test.append(entry)

# 写入json和csv
with open(output_train_json, 'w') as file:
    json.dump(output_data_train, file, indent=4)

with open(output_test_json, 'w') as file:
    json.dump(output_data_test, file, indent=4)

with open(output_csv, 'w', newline='') as file:
    fieldnames = ['id', 'text', 'title']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print("Processing completed.")
