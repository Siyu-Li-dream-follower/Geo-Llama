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
from traj_prompt_eval_hos import IndividualEvalHOS
from location_overlap import LocationOverlap

import re
from traj_reorder import TrajectoryReorder
import argparse
import json

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
    save_file_path_hos = config_params.get('save_file_path_hos')
    is_departure = config_params.get('is_departure')
    temperature = config_params.get('temperature')
    hos_file_path = config_params.get('hos_file_path')
    satisfied_indices_path = config_params.get('satisfied_indices_path') # for safisfied indices save only
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


# Initialize a list to store the event counts for each line
event_counts = []
##### no re_match for the initial HOS events
format_lines = []
# Read and process the file
with open(hos_file_path, 'r') as file:
    lines = file.readlines() 
    # Process each line
    for line in lines:
        # Clean the line
        cleaned_line = line.strip().strip("[]'")
        # Replace semicolons with ' ##' (if needed)
        format_line = cleaned_line.replace(";", " ##")
        # Split the line into events
        events = [event for event in cleaned_line.split(';') if event] # avoid counting empty events at the end
        # Count the number of events
        event_count = len(events)
        # Append the count to the list
        format_lines.append(format_line)
        event_counts.append(event_count)
    
lines = format_lines

# use tokenizer to encode the input
inputs = tokenizer(lines, max_length=MAX_LENGTH, return_tensors="pt", padding=True, truncation=True).to(device)

outputs = []

for i in range(0, len(lines), batch_size): 
    print(f"Processing batch starting at index: {i}")
    batch_inputs = {key: val[i:i+batch_size] for key, val in inputs.items()}
    output = model.generate(**batch_inputs, do_sample=True, temperature=temperature, max_length=MAX_LENGTH)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    outputs.extend(output_text)
# ===========================================================================================================

pattern_str = r'arrival time is (\d+), location is (\d+), duration is (\d+)'
pattern_str_hos = (
    r'arrival time is (\d+), location is (\d+), duration is (\d+)|'
    r'arrival time is (\d+), location is (\d+), duration is (\d+), departure time is (\d+)|'
    r'location is (\d+), departure time is (\d+)'
)
pattern_str_hos = re.compile(pattern_str_hos)
formated_outputs = []
satisfied_indices = []
HOS_index = 0
traj_index = 0
##### no re_match for the initial HOS events
for s in outputs:
    events = s.split(' ##')
    events = [event.strip() for event in events]
    format_flag = True
    # print(events)
    event_index = 1
    
    # print(events)
    for event in events:
        if event_index > event_counts[traj_index]:
            if not re.match(pattern_str, event):
                format_flag = False
                break
        else:
            if not re.match(pattern_str_hos, event):
                format_flag = False
                break
        event_index += 1
            
    if format_flag:
        formated_outputs.append(s)
        satisfied_indices.append(HOS_index)
        
    HOS_index += 1
    traj_index += 1


indices_str = ' '.join(str(index) for index in satisfied_indices)
with open(satisfied_indices_path, 'w') as file:
    file.write(indices_str)
    
    
    
formated_outputs = [s.replace(" ##", ";") for s in formated_outputs]
# --------------------------------------------------------------------------


traj_input = formated_outputs
## Save to precheck
with open(save_file_path_hos, 'w') as f:
    for item in traj_input:
        f.write(f"['{item};']\n")


hos_file = hos_file_path 
traj_reordering= TrajectoryReorder(traj_input, save_file_path_hos, satisfied_indices, hos_file, traj_file_mode=False, 
                                   include_agent_id=False, keep_agent_id= False, is_departure=is_departure)
processed_trajectories = traj_reordering.read_and_process_formatted()

## Evaluation
horizontal_n=200 # 200 for geolife data
top_k = 9 # top k 40 for geolife data
seq_len=96
interval=15
is_interval1 = True
use_grid_id1 = True
agent_id1=False
is_interval2 = True
use_grid_id2 = True
agent_id2=False

individualEval = IndividualEvalHOS(horizontal_n, seq_len, interval, npy_file_path, save_file_path_hos, gt_file_path, top_k, 
                                hos_file, is_interval1, use_grid_id1, is_interval2, use_grid_id2, agent_id1, agent_id2)

d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, transit_norm, topk_transit_norm=individualEval.get_individual_jsds()
    
print(f'distance jsd: {d_jsd},  gradius jsd: {g_jsd},'  
        f'duration jsd: {du_jsd},  periodicity jsd: {p_jsd},  frequency jsd: {f_jsd}, location jsd: {l_jsd},' 
        f'transition matrix norm: {transit_norm}, top k transition matrix norm: {topk_transit_norm}')

print(f'{d_jsd:.4f} & {g_jsd:.4f} & {du_jsd:.4f} & {p_jsd:.4f} & '
          f'{f_jsd:.4f} & {l_jsd:.4f} & {transit_norm:.4f} & {topk_transit_norm:.4f}')

location_overlap = LocationOverlap(save_file_path_hos, gt_file_path)
overlap_ratio = location_overlap.read_and_process_formatted()

print(f"Overlap Ratio: {overlap_ratio}")