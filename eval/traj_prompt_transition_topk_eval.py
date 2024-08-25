import re
import numpy as np
from collections import Counter
from scipy.spatial import distance

### It should only be applied to controllable generated data ###
class HOSTopKTransition:
    def __init__(self, hos_file_path, traj_file_path, traj_gt_path, vocab_size, top_k):
        self.hos_file_path = hos_file_path
        self.traj_file_path = traj_file_path
        self.traj_gt_path = traj_gt_path
        self.vocab_size = vocab_size
        self.top_k = top_k

    def read_hos_data(self):
        location_counter = Counter()
        total_locations = 0

        with open(self.hos_file_path, 'r') as file:
            for line in file:
                trajectory = line.strip().strip('[]')
                trajectory = trajectory.strip("[]' ")
                segments = trajectory.split(';')
                events = [segment.strip() for segment in segments if segment.strip()]

                for event in events:
                    location_match = re.search(r'location is (\d+)', event)
                    if location_match:
                        location_counter[int(location_match.group(1))] += 1
                        total_locations += 1

        return location_counter

    def get_top_k_locations(self, location_counter):
        if self.top_k > len(location_counter):
            print(f"Error: The selected top K value ({self.top_k}) is greater than the number of unique grid IDs ({len(location_counter)}) in HOS events.")
            print(f"Total unique grid IDs: {len(location_counter)}")
            return []
        top_k_locations = [loc for loc, _ in location_counter.most_common(self.top_k)]
        return top_k_locations

    def topk_transition_stats(self, traj_file_path, top_k_locations):
        trajectories = []
        with open(traj_file_path, 'r') as file:
            for line in file:
                trajectory = line.strip().strip('[]') 
                trajectory = trajectory.strip("[]' ")
                segments = trajectory.split(';')
                events = [segment.strip() for segment in segments if segment.strip()]
                trajectories.append(events)
        
        transition_matrix = np.zeros((self.vocab_size, self.vocab_size))
        total_count = 0

        for events in trajectories:
            for i in range(len(events) - 1):
                current_location_match = re.search(r'location is (\d+)', events[i])
                next_location_match = re.search(r'location is (\d+)', events[i + 1])
                
                if not current_location_match or not next_location_match:
                    continue
                
                current_grid_id = int(current_location_match.group(1))
                next_grid_id = int(next_location_match.group(1))
                
                if current_grid_id in top_k_locations or next_grid_id in top_k_locations:
                    if current_grid_id > self.vocab_size:
                        current_grid_id = self.vocab_size
                    
                    if next_grid_id > self.vocab_size:
                        next_grid_id = self.vocab_size
                    
                    transition_matrix[current_grid_id - 1][next_grid_id - 1] += 1
                    total_count += 1
        
        # calculate the probability of transition
        if total_count > 0:
            transition_matrix = transition_matrix / total_count
                
        return transition_matrix

    def process_topk(self):
        # Step 1: Read constraint data and count location frequencies
        location_counter = self.read_hos_data()
        
        # Step 2: Get top K locations
        top_k_locations = self.get_top_k_locations(location_counter)
        if not top_k_locations:
            return None

        print(f"Top {self.top_k} locations from HOS: {top_k_locations}")

        # Step 3: Calculate transition matrix based on top K locations
        transition_matrix = self.topk_transition_stats(self.traj_file_path, top_k_locations)
        transition_matrix_gt = self.topk_transition_stats(self.traj_gt_path, top_k_locations)
        
        diff = transition_matrix - transition_matrix_gt
        # compute the norm of the difference
        diff_norm = self.compute_matrix_norm(diff)
    
        return diff_norm

    def compute_matrix_norm(self, input_matrix):
        """
        compute the norm of a matrix
        """
        return np.linalg.norm(input_matrix, 'fro')


    def compute_JS_divergence(self, P, Q):
        JS_divergence = distance.jensenshannon(P, Q)
        return JS_divergence

if __name__ == '__main__':
    ############### For geolife top k transition calculation ################
    hos_file_path = ''  
    traj_gt_path = ''
    traj_file_path = ''  
    vocab_size = 29600  
    top_k = 40

    processor = HOSTopKTransition(hos_file_path, traj_file_path, traj_gt_path, vocab_size, top_k)
    transition_norm = processor.process_topk()
    print("The geolife top k transitional matrix norm is: " + str(transition_norm))
    


