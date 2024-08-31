from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, IA3Config, LoraConfig, prepare_model_for_kbit_training, PeftModel, PeftConfig
import torch
from datasets import load_dataset, Dataset, DatasetDict
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import login
import argparse
import json
import ast
##
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')

# Parse command line arguments
opt = parser.parse_args()

# Load configuration from the JSON file
if opt.config:
    config_params = load_config(opt.config)
    
    # Basic Training Paramters
    traj_file_path= config_params.get('traj_file_path')
    token = config_params.get('token')
    model_name = config_params.get('model_name')
    cache_dir = config_params.get('cache_dir')
    batch_size = config_params.get('batch_size')
    lr = config_params.get('lr')
    lora_alpha = config_params.get('lora_alpha')
    lora_dropout = config_params.get('lora_dropout')
    lora_r = config_params.get('lora_r')
    device = config_params.get('device')
    MAX_LENGTH = config_params.get('MAX_LENGTH')
    num_epochs = config_params.get('num_epochs')
    save_model_dir = config_params.get('save_model_dir')
    

login(token=token)
processed_data = []

with open(traj_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        line = ast.literal_eval(line)
        processed_data.append(line)
        
prompts = []
for item in processed_data:
    item_ = item[0]
    prompts.append(item_[:-1].replace(';', ' ##'))

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=f'{cache_dir}/huggingface/{model_name}', #cache_dir=cache_dir
                                          padding_side="left",
                                        #   token=token
                                         )
tokenizer.pad_token_id = tokenizer.bos_token_id

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            #  token=token, 
                                             cache_dir=f'{cache_dir}/huggingface/{model_name}', #cache_dir=f'{cache_dir}/huggingface/{model_name}', #cache_dir=cache_dir
                                             torch_dtype=torch.float16,
                                            #  device_map="auto"
                                            )
peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM"
        )

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id

tokenized_prompts = [tokenizer(prompt + tokenizer.eos_token, max_length=MAX_LENGTH, padding='max_length', add_special_tokens=False) for prompt in prompts]
for i in range(len(tokenized_prompts)):
    num_paddding_token = (np.array(tokenized_prompts[i]['attention_mask']) == 0).sum()
    tokenized_prompts[i]['labels'] = [-100]*num_paddding_token + tokenized_prompts[i]['input_ids'][num_paddding_token:]

# train test 80-20 split
tokenized_train_prompts, tokenized_test_prompts = train_test_split(tokenized_prompts, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(tokenized_train_prompts)
test_dataset = Dataset.from_list(tokenized_test_prompts)


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=16, pin_memory=True
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_epoch_loss = total_loss / len(train_dataloader)
    
    model.eval()
    eval_loss = 0
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().item()
    eval_epoch_loss = eval_loss / len(test_dataloader)

    print(f"Epoch {epoch} - train loss: {train_epoch_loss} - eval loss: {eval_epoch_loss}")
    model.save_pretrained(f"{save_model_dir}/ft_epoch_{epoch}")
