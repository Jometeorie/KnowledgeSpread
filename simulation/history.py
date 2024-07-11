import os
import json

class History:
    '''A class that keeps all shared history.'''
    def __init__(self, max_tokens, history_dir):
        self.max_tokens = max_tokens
        self.history_dir = history_dir
        self.history = []
        self.total_history = []
        self.num_history_tokens = 0
    
    def add_history(self, name, is_edit, turn, text, tokenizer, answers):
        self.history.append({"name": name, "turn": turn, "text": text})
        self.total_history.append({"name": name, "is_edit": is_edit, "turn": turn, "text": text, "answers": answers})

        self.num_history_tokens += len(tokenizer.encode(text))
        previous_text = self.history[0]["text"]
        while self.num_history_tokens > self.max_tokens and len(previous_text) > 0:
            self.history.pop(0)
            self.num_history_tokens -= len(tokenizer.encode(previous_text))

    def store_history(self, data_idx):
        history_root = os.path.join(self.history_dir, str(data_idx))
        if not os.path.exists(history_root):
            os.makedirs(history_root)
        with open(os.path.join(history_root, "full_history.json"), "w") as f:
            json.dump(self.total_history, f, indent=4)