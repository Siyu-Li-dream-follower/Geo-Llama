import pandas as pd
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
import re
import random
from scipy.spatial import distance


def read_and_process_data(traj_file_path, vocab_size):
    
    transition_matrix = np.zeros((vocab_size, vocab_size))
    total_count = 0

    with open(traj_file_path, 'r') as file:
        for line in file:
            trajectory = line.strip().strip('[]') 
            trajectory = trajectory.strip("[]' ")
            segments = trajectory.split(';')
            events = [segment.strip() for segment in segments if segment.strip()]
            
            for i in range(len(events) - 1):
        
                current_location_match = re.search(r'location is (\d+)', events[i])
                current_location_str = current_location_match.group(1) if current_location_match else "0"
                current_grid_id = int(current_location_str)
                
                if current_grid_id > vocab_size:
                        current_grid_id = vocab_size
                
                next_location_match = re.search(r'location is (\d+)', events[i+1])
                next_location_str = next_location_match.group(1) if next_location_match else "0"
                next_grid_id = int(next_location_str)
                
                if next_grid_id > vocab_size:
                        next_grid_id = vocab_size
                
                # since both of them start from 1, so we need to minus 1
                transition_matrix[current_grid_id-1][next_grid_id-1] += 1
                total_count += 1
            
    # calculate the probability of transition
    transition_matrix = transition_matrix / total_count
                
    return transition_matrix


def compute_matrix_norm(input_matrix):
    """
    compute the norm of a matrix
    """
    return np.linalg.norm(input_matrix, 'fro')


def compute_JS_divergence(P, Q):
    JS_divergence = distance.jensenshannon(P, Q)
    return JS_divergence


############### main function ################
if __name__ == '__main__':
    ##############################################################################################################
    #### Geolife Dataset ####
    ##############################################################################################################
    # horizontal_n=200
    vocab=29600 # vocab of the dataset used
    
    #### Trajectory Data 1 ####
    traj_file_path1 = ''
    
    #### Trajectory Data 2 ####
    traj_file_path2 = ''
    
    transition_matrix1 = read_and_process_data(traj_file_path1, vocab)
    transition_matrix2 = read_and_process_data(traj_file_path2, vocab)
    
    diff = transition_matrix1 - transition_matrix2
    # compute the norm of the difference
    diff_norm = compute_matrix_norm(diff)
    print("GeoLife Fake (transitional matrix norm): " + str(diff_norm))
    
