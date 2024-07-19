# Manipulated Knowledge Spread

Reproduction Code for Paper "Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities". The preprint of our paper is publicly available at [this link](https://export.arxiv.org/abs/2407.07791).

![https://github.com/Jometeorie/KnowledgeSpread/blob/main/figures/introduction.png](https://github.com/Jometeorie/KnowledgeSpread/blob/main/figures/introduction.png)

## Requirements

Install the required python dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

All datasets we used are provided in the [data/](https://github.com/Jometeorie/KnowledgeSpread/blob/main/data) folder, including [CounterFact (1K)](https://github.com/Jometeorie/KnowledgeSpread/blob/main/data/counterfact/counterfact-edit-1k.json), [zsRE (1K)](ttps://github.com/Jometeorie/KnowledgeSpread/blob/main/data/zsre/zsre_mend_train-1k.json) and their toxic versions.

## Instructions

### Baseline

Perform and evaluate knowledge editing on the CounterFact (1K) dataset using vicuna 7B without any multi-agent interaction:

```bash
python baseline_easyedit.py --config_path=../config/agent/vicuna-7b.yaml
```

## Intuition Verification

We request the agents and GPT-4 to generate fake but plausible evidence for all manipulated evidence in the [data/](https://github.com/Jometeorie/KnowledgeSpread/blob/main/data) folder.

Evaluate the extent to which single agent is persuaded under different prompt settings:

```bash
python baseline_prompt_edit.py --config_path=../config/agent/vicuna-7b.yaml --prompt_type=no_edit
```

```bash
python baseline_prompt_edit.py --config_path=../config/agent/vicuna-7b.yaml --prompt_type=direct_answer --with_evidence
```

## Attack Pipeline

### Stage 1: Persuasiveness Injection

To inject persuasiveness into the agent, you should first generate perference data for the LLM:

```bash
python generate_dataset.py
```

We encourage to generate different perference datasets for different LLMs, which minimizes the impact of the LLMs.

Then we use the DPO method for training:

```bash
python dpo_training.py
```

You can modify *ckpt_path* to adjust the LoRA model path, which will be used in the second stage.

### Stage 2: Manipulated Knowledge Injection

#### Training

As a running example, the script for testing the results of manipulated knowledge spread on the CounterFact (1K) dataset using vicuna 7B is as follows:

```bash
python simulation.py --config_path=../config/agent/vicuna-7b.yaml
```

All chats will be stored in *history/* for subsequent experimental analyses. For other experimental setups, you can modify the corresponding yaml file in *config/*.

## RAG Scenario

1. Format chat histories

   ```bash
   python format_rag.py --dataset_path=./counterfact/counterfact-edit-1k.json --input_folder=<chat_history_directory>
   ```

2. RAG training

3. Evaluation

   ```bash
   python baseline_prompt_edit.py --config_path=../config/agent/vicuna-7b.yaml --prompt_type=rag --rag_path=<path_to_rag> --top_k=5 
   ```


## Citing

```markdown
@misc{ju2024flooding,
    title={Flooding Spread of Manipulated Knowledge in LLM-Based Multi-Agent Communities},
    author={Tianjie Ju and Yiting Wang and Xinbei Ma and Pengzhou Cheng and Haodong Zhao and Yulong Wang and Lifeng Liu and Jian Xie and Zhuosheng Zhang and Gongshen Liu},
    year={2024},
    eprint={2407.07791},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

This project is licensed under the Apache-2.0 License.