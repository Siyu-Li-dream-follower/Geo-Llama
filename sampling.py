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
import re
import argparse
import json
from traj_reorder import TrajectoryReorder
from traj_prompt_eval import IndividualEval
from location_overlap import LocationOverlap

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
    token = config_params.get('token')
    model_name = config_params.get('model_name')
    cache_dir = config_params.get('cache_dir')
    batch_size = config_params.get('batch_size')
    device = config_params.get('device')
    MAX_LENGTH = config_params.get('MAX_LENGTH')
    ft_model_path = config_params.get('ft_model_path')
    output_num = config_params.get('output_num')
    save_file_path = config_params.get('save_file_path')
    temperature = config_params.get('temperature')
    is_departure = config_params.get('is_departure')
    gt_file_path = config_params.get('gt_file_path')
    npy_file_path = config_params.get('npy_file_path')


tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                          padding_side="left",
                                          token=token
                                         )
tokenizer.pad_token_id = tokenizer.bos_token_id

base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             token=token, 
                                             cache_dir=f'{cache_dir}/huggingface/{model_name}',
                                             torch_dtype=torch.float16,
                                            #  device_map="auto"
                                            )
peft_config = PeftConfig.from_pretrained(ft_model_path)

base_model.gradient_checkpointing_enable()
base_model = prepare_model_for_kbit_training(base_model)
model = PeftModel.from_pretrained(base_model, ft_model_path, config=peft_config)
model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id
# ==========================================================================================================

prompts = ["arrival time is"]*batch_size
inputs = tokenizer(prompts, max_length= MAX_LENGTH, return_tensors="pt", padding=True, truncation=True).to(device) #max_length= MAX_LENGTH,

outputs = []
for i in range(0, output_num, batch_size):
    print(i)
    output = model.generate(**inputs, do_sample=True, temperature=temperature, max_length=MAX_LENGTH+10)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    outputs.extend(output_text)
# ===========================================================================================================

# Post-processing ===========================================================================================
# format checking TODO: currently hard code for the temporal order ---
patern_str = r'arrival time is (\d+), location is (\d+), duration is (\d+)' # cannot handle event permute
formated_outputs = []
for s in outputs:
    events = s.split(' ##')
    events = [event.strip() for event in events]
    format_flag = True
    for event in events:
        if not re.match(patern_str, event):
            format_flag = False
            break
    if format_flag:
        formated_outputs.append(s)
formated_outputs = [s.replace(" ##", ";") for s in formated_outputs]
# --------------------------------------------------------------------------
# Save to file ============================================================================================
## comment on the following if use TrajectoryReorder as it has 'save' within it.
with open(save_file_path, 'w') as f:
    for item in formated_outputs:
        f.write(f"['{item};']\n")
        
## Reorder ------------------------------------------------------------------
## manually reorder the events to the correct order

traj_input = formated_outputs
satisfied_indices = None
hos_file = None

traj_reordering= TrajectoryReorder(traj_input, save_file_path, satisfied_indices, hos_file, traj_file_mode=False, 
                                   include_agent_id=False, keep_agent_id= False, is_departure=is_departure)
processed_trajectories = traj_reordering.read_and_process_formatted()
## --------------------------------------------------------------------------
## =========================================================================================================
## Evaluation
horizontal_n=200
seq_len=96
interval=15
is_interval1 = True
use_grid_id1 = True
agent_id1=False
is_interval2 = True
use_grid_id2 = True
agent_id2=False


individualEval = IndividualEval(horizontal_n, seq_len, interval, npy_file_path, save_file_path, gt_file_path, 
                                is_interval1, use_grid_id1, is_interval2, use_grid_id2, agent_id1, agent_id2)

d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, transit_norm=individualEval.get_individual_jsds()
    
print(f'distance jsd: {d_jsd},  gradius jsd: {g_jsd},'  
    f'duration jsd: {du_jsd},  periodicity jsd: {p_jsd},  frequency jsd: {f_jsd}, location jsd: {l_jsd}, transition matrix norm: {transit_norm}')

print(f'{d_jsd:.4f} & {g_jsd:.4f} & {du_jsd:.4f} & {p_jsd:.4f} & '
          f'{f_jsd:.4f} & {l_jsd:.4f} & {transit_norm:.4f}')

location_overlap = LocationOverlap(save_file_path, gt_file_path)
overlap_ratio = location_overlap.read_and_process_formatted()

print(f"Overlap Ratio: {overlap_ratio}")
