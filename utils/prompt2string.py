import pandas as pd
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
import re
import random

class Prompt2Seq(object): 
    
    def __init__(self, horizontal_n, seq_len, npy_file_path, traj_file_path, traj_save_path, is_interval=False, use_grid_id=False, agent_id=False):
        self.stays = np.load(npy_file_path, allow_pickle=True)
        self.min_lat = np.min(self.stays[:, 1])
        self.max_lat = np.max(self.stays[:, 1])
        self.min_lon = np.min(self.stays[:, 2])
        self.max_lon = np.max(self.stays[:, 2])
        self.seq_len = seq_len
        self.horizontal_n = horizontal_n
        self.vertical_n = int((self.max_lat - self.min_lat) / (self.max_lon - self.min_lon) * horizontal_n)
        self.max_locs = self.horizontal_n * self.vertical_n
        # Track the last location from the previous trajectory
        self.processed_traj1 = self.read_and_process_data(traj_file_path, traj_save_path, is_interval, use_grid_id, agent_id)

   
    def coord2grid(self, coord_tuple):
        lat_min, lat_max = self.min_lat, self.max_lat
        lon_min, lon_max = self.min_lon, self.max_lon
        current_lat, current_long = coord_tuple
        if current_lat < lat_min or current_lat > lat_max or current_long < lon_min or current_long > lon_max:
            return None
        horizontal_resolution = (lon_max - lon_min) / self.horizontal_n
        vertical_resolution = (lat_max - lat_min) / self.vertical_n
        y = min(int((current_long - lon_min) / horizontal_resolution), self.horizontal_n - 1)
        x = min(int((current_lat - lat_min) / vertical_resolution), self.vertical_n - 1)
        return x * self.horizontal_n + y + 1
    
    def process_trajectory(self, trajectory, last_location, is_interval, use_grid_id, agent_id):
        trajectory = trajectory.strip("[]' ")
        if agent_id:
            segments = trajectory.split(';')[1:]  # Skip the agent id part
        else:
            segments = trajectory.split(';')
            
        sequence = []  # Initialize the day sequence with grid ID 0
        start_flag = True
        
        for segment in segments:
            if not segment.strip():
                continue
            
            if is_interval:
                time_match = re.search(r'arrival time is (\d+)', segment)
                time = int(time_match.group(1)) * 15 if time_match else 0
            else:
                time_match = re.search(r'arrival time is ([\d:]+)', segment)
                if time_match:
                    hours, minutes = map(int, time_match.group(1).split(':'))
                    time = hours * 60 + minutes
                else:
                    time = 0
            
            if use_grid_id:
                location_match = re.search(r'location is (\d+)', segment)
                location_str = location_match.group(1) if location_match else "0"
            else:
                location_match = re.search(r'location is \(([\d\.,]+)\)', segment) # (lat, lon) is replaced with (lat,lon)
                location_str = location_match.group(1) if location_match else "(0,0)"
            
            if is_interval:
                duration_match = re.search(r'duration is (\d+)(?! minutes)', segment)
            else:
                duration_match = re.search(r'duration is (\d+) minutes', segment)

            duration = int(duration_match.group(1)) if duration_match else 0
            
            if is_interval:
                duration *= 15 
            
            if use_grid_id:
                grid_id = int(location_str)
            else:
                location = tuple(map(float, location_str.strip('()').split(',')))  # Parse lat, lon as a tuple
                grid_id = self.coord2grid(location)  # Convert lat, lon to grid ID even if location_discretization is off
            
            
            start_index = time // 15
            
            if start_flag:
                for _ in range(start_index // 15):
                    sequence.append(last_location)
            
                start_flag = False
                
            for _ in range(duration // 15):
                if len(sequence) >= self.seq_len: # Truncate the sequence to the maximum length handle no interity check case
                    break
                sequence.append(grid_id)  # Use grid ID or default to 0 if out of bounds
                
            last_location = grid_id # Update last known location   
        # print(sequence)
        
        while len(sequence) < self.seq_len:
            sequence.append(sequence[-1])

        return sequence, last_location

    def read_and_process_data(self, traj_file_path, traj_save_path, is_interval, use_grid_id, agent_id):
        default_location = 1 if use_grid_id else (self.min_lat, self.min_lon)
        data_sequences = []
        last_location = default_location

        with open(traj_file_path, 'r') as file:
            for line in file:
                trajectory = line.strip().strip('[]')  
                processed_sequence, prev_location = self.process_trajectory(trajectory, last_location, is_interval, use_grid_id, agent_id)
                last_location = prev_location
                data_sequences.append(processed_sequence)
        
        with open(traj_save_path, 'w') as file:
            for seq in data_sequences:
                file.write(' '.join(map(str, seq)) + '\n')
        
        return data_sequences
    
if __name__ == "__main__":
    horizontal_n = 200
    seq_len = 96
    is_interval = True
    use_grid_id = True
    agent_id = False
    ############################################################
    npy_file_path = ""
    ############################################################
    
    ################# Geolife Ground Truth #################
    traj_file_path = ""
    traj_save_path = ""
    
    prompt2seq = Prompt2Seq(horizontal_n, seq_len, npy_file_path, traj_file_path, traj_save_path, is_interval, use_grid_id, agent_id)