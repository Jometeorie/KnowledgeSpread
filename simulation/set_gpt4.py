import torch
import json
import os
import numpy as np
import random
import requests


def assess_credibility(history, generated_text):
    api_key = ""
    conversation_history = f"\nFull History:\n"
    for i, h in enumerate(history):
            name = h["name"]
            text = h["text"]
            conversation_history += f"\t{i+1}. {name}: {text}\n"
    conversation_history += f"\nCurrent Response:\n{generated_text}\n"

    url = ""

    headers = {
            'Authorization': f'Bearer {api_key}',  # Replace {api_key} with your actual API key
            'Content-Type': 'application/json'
    }

    request = {
            "inputs": {},
            "query": conversation_history,
            "response_mode": "blocking",
            "conversation_id": "",  # Set the conversation ID if necessary
            "user": "abc-123",
        }
    
    response = requests.post(url, headers=headers, json=request)
    response_data = response.json()  
    answer = response_data.get('answer', None)  
    return answer
