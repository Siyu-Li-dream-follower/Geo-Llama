### For controllable generation evaluation ###
### Part of the code is adopted from MoveSim ###

import pandas as pd
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
import re
import random
from scipy.spatial import distance

class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
        ret_dist, ret_base = [], []  ## the rest are for removing zero buckets
        for i in range(bins):
            if int(distribution[i]) == 0: 
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14) ## avoid zero division
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js
    

class IndividualEvalHOS(object): 
    
    def __init__(self, horizontal_n, seq_len, interval, npy_file_path, traj_file_path1, traj_file_path2, top_k, hos_path= None, is_interval1=False, use_grid_id1=False, is_interval2=False, use_grid_id2=False, agent_id1=False, agent_id2=False):
        self.stays = np.load(npy_file_path, allow_pickle=True)
        self.min_lat = np.min(self.stays[:, 1])
        self.max_lat = np.max(self.stays[:, 1])
        self.min_lon = np.min(self.stays[:, 2])
        self.max_lon = np.max(self.stays[:, 2])
        self.seq_len = seq_len
        self.horizontal_n = horizontal_n
        self.top_k = top_k
        self.vertical_n = int((self.max_lat - self.min_lat) / (self.max_lon - self.min_lon) * horizontal_n)
        self.max_locs = self.horizontal_n * self.vertical_n
        # attributes of the trajectory 1
        # Track the last location from the previous trajectory
        self.processed_traj1 = self.read_and_process_data(traj_file_path1, is_interval1, use_grid_id1, agent_id1)
            
        # attributes of the trajectory 2
        # Track the last location from the previous trajectory
        self.processed_traj2 = self.read_and_process_data(traj_file_path2, is_interval2, use_grid_id2, agent_id2)
        
        # Format the trajectories into numpy arrays
        self.formatted_traj1 = np.asarray(self.processed_traj1, dtype='int64')
        self.formatted_traj2 = np.asarray(self.processed_traj2, dtype='int64')
            
        # parameters for evaluation
        self.max_distance = (self.max_lat-self.min_lat)**2 + (self.max_lon - self.min_lon)**2
        self.interval = interval
        self.X, self.Y = self.grid2coor()
        
        # Transition matrix calculation
        self.transition_matrix1 = self.transition_stats(traj_file_path1)
        self.transition_matrix2 = self.transition_stats(traj_file_path2)
        
        if hos_path is not None:
            self.hos_path = hos_path
            topk_diff_norm = self.process_topk(traj_file_path1, traj_file_path2)  
            self.topk_diff_norm = topk_diff_norm    
        
    def read_hos_data(self):
        location_counter = Counter()
        total_locations = 0

        with open(self.hos_path, 'r') as file:
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
    
    def transition_stats(self, traj_file_path):
    
        transition_matrix = np.zeros((self.max_locs, self.max_locs))
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
                    
                    if current_grid_id > self.max_locs:
                        current_grid_id = self.max_locs
                    next_location_match = re.search(r'location is (\d+)', events[i+1])
                    next_location_str = next_location_match.group(1) if next_location_match else "0"
                    next_grid_id = int(next_location_str)
                    
                    if next_grid_id > self.max_locs:
                        next_grid_id = self.max_locs
                    # since both of them start from 1, so we need to minus 1
                    transition_matrix[current_grid_id-1][next_grid_id-1] += 1
                    total_count += 1
                
        # calculate the probability of transition
        transition_matrix = transition_matrix / total_count
                    
        return transition_matrix
    
    def topk_transition_stats(self, traj_file_path, top_k_locations):
        trajectories = []
        with open(traj_file_path, 'r') as file:
            for line in file:
                trajectory = line.strip().strip('[]') 
                trajectory = trajectory.strip("[]' ")
                segments = trajectory.split(';')
                events = [segment.strip() for segment in segments if segment.strip()]
                trajectories.append(events)
        
        transition_matrix = np.zeros((self.max_locs, self.max_locs))
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
                    if current_grid_id > self.max_locs:
                        current_grid_id = self.max_locs
                    
                    if next_grid_id > self.max_locs:
                        next_grid_id = self.max_locs
                    
                    transition_matrix[current_grid_id - 1][next_grid_id - 1] += 1
                    total_count += 1
        
        # calculate the probability of transition
        if total_count > 0:
            transition_matrix = transition_matrix / total_count
                
        return transition_matrix

    def compute_matrix_norm(self, input_matrix):
        """
        compute the norm of a matrix
        """
        return np.linalg.norm(input_matrix, 'fro')


    def compute_JS_divergence(self, P, Q): ## Not used, we use Frobenius norm above instead
        JS_divergence = distance.jensenshannon(P, Q)
        return JS_divergence
    
    def get_top_k_locations(self, location_counter):
        if self.top_k > len(location_counter):
            print(f"Error: The selected top K value ({self.top_k}) is greater than the number of unique grid IDs ({len(location_counter)}) in HOS events.")
            print(f"Total unique grid IDs: {len(location_counter)}")
            return []
        top_k_locations = [loc for loc, _ in location_counter.most_common(self.top_k)]
        return top_k_locations
    
    def process_trajectory(self, trajectory, last_location, is_interval, use_grid_id, agent_id):
        trajectory = trajectory.strip("[]' ")
        if agent_id:
            segments = trajectory.split(';')[1:] 
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

    def read_and_process_data(self, traj_file_path, is_interval, use_grid_id, agent_id):
        default_location = 1 if use_grid_id else (self.min_lat, self.min_lon)
        data_sequences = []
        last_location = default_location

        with open(traj_file_path, 'r') as file:
            for line in file:
                trajectory = line.strip().strip('[]') 
                processed_sequence, prev_location = self.process_trajectory(trajectory, last_location, is_interval, use_grid_id, agent_id)
                last_location = prev_location
                data_sequences.append(processed_sequence)
                
        return data_sequences
    
    def process_topk(self, traj_file_path, traj_gt_path):
        # Step 1: Read constraints data and count location frequencies
        location_counter = self.read_hos_data()
        
        # Step 2: Get top K locations
        top_k_locations = self.get_top_k_locations(location_counter)
        if not top_k_locations:
            return None

        print(f"Top {self.top_k} locations from HOS: {top_k_locations}")

        # Step 3: Calculate transition matrix based on top K locations
        transition_matrix = self.topk_transition_stats(traj_file_path, top_k_locations)
        transition_matrix_gt = self.topk_transition_stats(traj_gt_path, top_k_locations)
        
        diff = transition_matrix - transition_matrix_gt
        # compute the norm of the difference
        diff_norm = self.compute_matrix_norm(diff)
    
        return diff_norm
    
    def align_and_format_trajectories(self, traj1, traj2): # No need in most cases
        # add random shuffle to avoid ordering bias in original data.
        random.shuffle(traj1)
        random.shuffle(traj2)
        # find the total num of trajs in the shorter list of the two
        min_length = min(len(traj1), len(traj2))
        
        # truncation to make equal length
        formatted_traj1 = traj1[:min_length]
        formatted_traj2 = traj2[:min_length]
        
        return np.asarray(formatted_traj1, dtype='int64'), np.asarray(formatted_traj2, dtype='int64')
    
    def grid2coor(self):
        
        horizontal_resolution = (self.max_lon - self.min_lon) / self.horizontal_n
        vertical_resolution = (self.max_lat - self.min_lat) / self.vertical_n
        X=[]
        Y=[]
            
        X=[(self.min_lat+(grid//self.horizontal_n+0.5)*vertical_resolution) for grid in range(1, self.max_locs + 1)]
        Y=[(self.min_lon+(grid%self.horizontal_n-0.5)*horizontal_resolution) for grid in range(1, self.max_locs + 1)]
        
        return X, Y
    
    def get_topk_visits(self,trajs, k):
        topk_visits_loc = [] # this is for all trajectories
        topk_visits_freq = [] # this is for all trajectories
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                # supplement with (loc=-1, freq=0)
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / trajs.shape[1]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    
    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)


    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    
    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)

    
    def get_distances(self, trajs): # approximate the earth as a plane in a small area: What We Want
        distances = []
        
        for traj in trajs:
            for i in range(self.seq_len - 1):
                if traj[i+1]>self.max_locs:
                    traj[i+1]=self.max_locs
                if traj[i]>self.max_locs:
                    traj[i]=self.max_locs
                dx = self.X[traj[i]-1] - self.X[traj[i + 1]-1] # need to map grid ids back into coodinates
                dy = self.Y[traj[i]-1] - self.Y[traj[i + 1]-1] # retrieve the coordinates from the grid ids, x and y need to be paired with grid id in advance
                distances.append(np.sqrt(dx**2 + dy**2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations(self, trajs): # how they handle pure trajectories without explicitly representing time, change the time interval and denominators
        d = []
        for traj in trajs:
            num = self.interval # in our case it is 15 min interva;
            for i, lc in enumerate(traj[1:]): # from 1 because duration is an operation between two points
                if lc == traj[i]:
                    num += self.interval
                else:
                    d.append(num)
                    num = self.interval
        return np.array(d)/self.seq_len 
    
    def get_gradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        
        for traj in trajs:
            xs = np.array([self.X[t-1] for t in traj]) # t is grid ids of timesteps in a trajectory, same as above
            ys = np.array([self.Y[t-1] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys) # since we are calculating the radius, we need centers
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i]**2 + dys[i]**2 for i in range(self.seq_len)] 
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float) # gradius is for all trajectories
        return gradius
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj)))/self.seq_len) # change sequence length
        reps = np.array(reps, dtype=float)
        return reps         # reps is frequency which is for later use, numerator is for total non-repeated locations

    def get_timewise_periodicity(self, trajs):
        """
        stat how many repetitions of different times
        :param trajs:
        :return:
        """
        pass


    def get_individual_jsds(self):
        """
        get jsd scores of individual evaluation metrics
        :param t1: test_data
        :param t2: gene_data
        :return:
        """
        d1 = self.get_distances(self.formatted_traj1)
        d2 = self.get_distances(self.formatted_traj2)
        
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, 0, np.sqrt(self.max_distance), 10000)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, 0, np.sqrt(self.max_distance), 10000)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        

        g1 = self.get_gradius(self.formatted_traj1)
        g2 = self.get_gradius(self.formatted_traj2)
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, 0, self.max_distance, 10000) 
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, 0, self.max_distance, 10000)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)
        
    
        du1 = self.get_durations(self.formatted_traj1)
        du2 = self.get_durations(self.formatted_traj2)     
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        
        p1 = self.get_periodicity(self.formatted_traj1)
        p2 = self.get_periodicity(self.formatted_traj2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        
        l1 =  CollectiveEval.get_visits(self.formatted_traj1, self.max_locs)
        l2 =  CollectiveEval.get_visits(self.formatted_traj2, self.max_locs)
        l1_dist, _ = CollectiveEval.get_topk_visits(l1, 100)
        l2_dist, _ = CollectiveEval.get_topk_visits(l2, 100)
        l1_dist, _ = EvalUtils.arr_to_distribution(l1_dist,0,1,100)
        l2_dist, _ = EvalUtils.arr_to_distribution(l2_dist,0,1,100)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        f1 = self.get_overall_topk_visits_freq(self.formatted_traj1, 100)
        f2 = self.get_overall_topk_visits_freq(self.formatted_traj2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1,0,1,100)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2,0,1,100)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)

        diff = self.transition_matrix1 - self.transition_matrix2
        # compute the norm of the difference
        diff_norm = self.compute_matrix_norm(diff)
        
        return d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, diff_norm, self.topk_diff_norm



class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    @staticmethod
    def get_visits(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float) # max number of locations of interest, maybe max vocab size
        for traj in trajs:
            for t in traj:
                visits[t-1] += 1
        visits = visits / np.sum(visits) # get all frequencies
        return visits

    @staticmethod
    def get_timewise_visits(trajs):
        """
        stat how many visits of a certain location in a certain time
        :param trajs:
        :return:
        """
        pass

    @staticmethod
    def get_topk_visits(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [(locs_visits[i][0]+1) for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        """
        get the accuracy of top-k visiting locations
        :param v1:
        :param v2:
        :param K:
        :return:
        """
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K

############### main function ################
if __name__ == '__main__':

##############################################################################################################
    horizontal_n=200
    max_locs=29600 # vocab of the dataset used
    seq_len=96
    interval=15
    npy_file_path = ''
    #### The evaluation code will be compatible with two trajectories with different representations, just adjust corresponding flags
    #### No matter what original location representation is, it will be transformed into grid id, as things like top k locations are based on grid id
    #### can only be calculated with grid id
    #### While for location jsd, it is based on continuous gps coordinates (lat, lon)
    
    ############## W agent id W grid id W 15 min interval V.S. W agent id WO grid id WO 15 min interval ##############
    #### Trajectory Data 1 ####
    traj_file_path1 = ''
    is_interval1 = True
    use_grid_id1 = True
    agent_id1=False # there is agent id in generated data, if not set false
    #### Trajectory Data 2 ####
    traj_file_path2 = ''
    is_interval2 = True
    use_grid_id2 = True
    agent_id2=False
    
    ## constraints Prompt File ##
    hos_file_path = ''
    
    ## Top K ##
    top_k = 40
    
    #### Evaluation ####
    individualEval = IndividualEvalHOS(horizontal_n, seq_len, interval, npy_file_path, traj_file_path1, traj_file_path2, top_k,
                                       hos_file_path, is_interval1, use_grid_id1, is_interval2, use_grid_id2, agent_id1, agent_id2)
    
    d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, transit_norm, topk_transit_norm=individualEval.get_individual_jsds()
    
    print(f'distance jsd: {d_jsd},  gradius jsd: {g_jsd},'  
        f'duration jsd: {du_jsd},  periodicity jsd: {p_jsd},  frequency jsd: {f_jsd}, location jsd: {l_jsd},' 
        f'transition matrix norm: {transit_norm}, top k transition matrix norm: {topk_transit_norm}')

    