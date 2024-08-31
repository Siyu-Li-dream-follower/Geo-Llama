import numpy as np
import re
import random
import itertools
import math
import ast
from traj_prompt_eval_hos import IndividualEvalHOS
#### formatted output: patern_str = r'arrival time is (\d+), location is (\d+), duration is (\d+)'
#### each trajectory in the list formatted output is:
#### 'arrival time is (\d+), location is (\d+), duration is (\d+); arrival time is (\d+), location is (\d+), duration is (\d+); '

###### Here constraints are abbreviated as: HOS

class RandomHOSInsert:    
    def __init__(self, traj_input, output_filename, hos_file, is_departure=False, is_integrity_check=False):
            
            self.traj_input = traj_input
            self.output_filename = output_filename
            self.hos_file = hos_file
            self.is_departure = is_departure 
            self.is_integrity_check = is_integrity_check
            
    def read_and_process_formatted(self):
        processed_trajectories = []
        
        violated_trajectories = 0
        total_violated_events = 0
        trajectories = []
        hoses = []
        
        with open(self.traj_input, 'r') as traj_data:
            
            for line in traj_data:
                
                trajectory = line.strip()  
                trajectories.append(trajectory)
                
        with open(self.hos_file, 'r') as hos_data:

            for line in hos_data:
                
                hos = line.strip()
                hoses.append(hos)
                
        num_trajectories = min(len(trajectories), len(hoses))
                
        for trajectory, hos in zip(trajectories, hoses):
            
            events = trajectory.strip("[]'").rstrip(';').rstrip('; ').split(';')
                
            hos_events = hos.strip("[]'").rstrip(';').rstrip('; ').split(';')
                
            processed_events, violated_events, violated_trajectory = self.process_trajectory_with_hos(events, hos_events)
            violated_trajectories += violated_trajectory
            total_violated_events += violated_events
                    
            processed_trajectories.append(processed_events)
            
        print(f"Total violated events in trajectory: {total_violated_events}")
        print(f"Violated trajectories: {violated_trajectories}")
        if num_trajectories != 0:
            print(f"Percentage of Violated trajectories: {violated_trajectories/num_trajectories}")
        
        with open(self.output_filename, 'w') as file:
            for data in processed_trajectories:
                file.write(f"['{data};']\n")
                
        return processed_trajectories
    
    def process_trajectory_with_hos(self, events, hos_events):
        processed_events = []
        
        for hos_event in hos_events:
            processed_hos_event = self.parse_hos_event(hos_event)
            processed_events.append(processed_hos_event)
            
        for event in events:
            processed_event = self.parse_event(event)
            processed_events.append(processed_event)
        
        sorted_events = self.sort_events_by_arrival_time(processed_events)
        
        if self.is_integrity_check:
            adjusted_events, violated_events, violated_trajectory = self.adjust_events_timing(sorted_events)
        else:
            _, violated_events, violated_trajectory = self.adjust_events_timing(sorted_events)
            adjusted_events = sorted_events
            
        adjusted_events = '; '.join(event for event in adjusted_events)
        
        return adjusted_events, violated_events, violated_trajectory
    
    def parse_event(self, event):
        ######## Following Re feature pattern is compatile with:
        ## arrival time is 21:00, location is (32.585316,35.862898), duration is 452 minutes, departure time is 23:59 
        ## arrival time is 55, location is 55, duration is 55, departure time is 55
        ## Or any of the feature format combinations of the above 
        ## data with missing features also works
        if self.is_departure:
            feature_pattern = r'''
                (?P<arrival_time>arrival\s+time\s+is\s+(\d{2}:\d{2}|\d+))|
                (?P<location>location\s+is\s+((\(\d+\.\d+,\d+\.\d+\))|\d+))|
                (?P<duration>duration\s+is\s+(\d+\s+minutes|\d+))|
                (?P<departure_time>departure\s+time\s+is\s+(\d{2}:\d{2}|\d+))
            '''
        else:
            feature_pattern = r'''
                (?P<arrival_time>arrival\s+time\s+is\s+(\d{2}:\d{2}|\d+))|
                (?P<location>location\s+is\s+((\(\d+\.\d+,\d+\.\d+\))|\d+))|
                (?P<duration>duration\s+is\s+(\d+\s+minutes|\d+))
            '''
        features = {}
        for match in re.finditer(feature_pattern, event, re.VERBOSE):
            features.update({k: v for k, v in match.groupdict().items() if v})
        # Reorder the features according to a specific sequence and return as a string
        
        if self.is_departure:
            ordered_features = [
                f"{features.get('arrival_time')}",
                f"{features.get('location')}",
                f"{features.get('duration','duration is 0')}",
                f"{features.get('departure_time')}"
            ]
        else:
            ordered_features = [
                f"{features.get('arrival_time')}",
                f"{features.get('location')}",
                f"{features.get('duration','duration is 0')}"
            ]
        
        return ', '.join(filter(None, ordered_features))
    
    def parse_hos_event(self, hos_event):
        if self.is_departure:
            feature_pattern = r'''
                (?P<arrival_time>arrival\s+time\s+is\s+(\d{2}:\d{2}|\d+))|
                (?P<location>location\s+is\s+((\(\d+\.\d+,\d+\.\d+\))|\d+))|
                (?P<duration>duration\s+is\s+(\d+\s+minutes|\d+))|
                (?P<departure_time>departure\s+time\s+is\s+(\d{2}:\d{2}|\d+))
            '''
        else:
            feature_pattern = r'''
                (?P<arrival_time>arrival\s+time\s+is\s+(\d{2}:\d{2}|\d+))|
                (?P<location>location\s+is\s+((\(\d+\.\d+,\d+\.\d+\))|\d+))|
                (?P<duration>duration\s+is\s+(\d+\s+minutes|\d+))
            '''
        features = {}
        for match in re.finditer(feature_pattern, hos_event, re.VERBOSE):
            features.update({k: v for k, v in match.groupdict().items() if v})
        
        # Check if 'arrival_time' is missing and 'departure_time' is available
        # That is how 'parse_hos_event' is different from 'parse_event'
        if 'arrival_time' not in features and 'departure_time' in features:
            time_value = features['departure_time'].split('departure time is ')[-1]  # Assumes the structure is consistent
            features['arrival_time'] = f"arrival time is {time_value}"
        
        # Reorder the features according to a specific sequence and return as a string
        if self.is_departure:
            ordered_features = [
                f"HOS {features.get('arrival_time')}",
                f"HOS {features.get('location')}",
                f"HOS {features.get('duration','duration is 0')}", ## constraints duration is 0 if not provided
                f"HOS {features.get('departure_time')}" ## departure can be missing
            ]
        else:
            ordered_features = [
                f"HOS {features.get('arrival_time')}",
                f"HOS {features.get('location')}",
                f"HOS {features.get('duration','duration is 0')}" ## HOS duration is 0 if not provided
            ]
        
        if self.is_integrity_check is False:
            ordered_features[2] = f"HOS {features.get('duration', f'duration is {random.randint(1, 4)}')}"
        
        return ', '.join(filter(None, ordered_features))
    
    def sort_events_by_arrival_time(self, events):

        def convert_time(time_str):
            if ":" in time_str:  # Handles "HH:MM" format
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            return int(time_str)  # Handles pure numeric format
        # print(events)
        sorted_events = sorted(events, key=lambda event: convert_time(re.search(r'arrival time is (\d{2}:\d{2}|\d+)', event).group(1)))
        
        return sorted_events
    
    def adjust_events_timing(self, sorted_events):
        violated_events = 0
        violated_trajectory = 0
        for i in range(len(sorted_events)):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1] if i + 1 < len(sorted_events) else None
            prev_event = sorted_events[i - 1] if i > 0 else None

            # Extract current event times and duration
            current_arrival_str = re.search(r'(?:HOS )?arrival time is (\d{2}:\d{2}|\d+)', current_event).group(1)
            current_duration_match = re.search(r'(?:HOS )?duration is (\d+)( minutes)?', current_event)
            # print(current_event)
            current_duration = int(current_duration_match.group(1))
            duration_format = current_duration_match.group(2) # minutes match or not matched reflected

            # Convert to minutes since start of the day if in HH:MM format
            if ':' in current_arrival_str:
                hours, minutes = map(int, current_arrival_str.split(':'))
                current_arrival = hours * 60 + minutes
            else:
                current_arrival = int(current_arrival_str) * 15  # Convert each unit to 15 minutes

            # Adjust timings based on event type and conditions
            if 'HOS' not in current_event:
                event_end_time = current_arrival + current_duration
                if next_event:
                    next_arrival_str = re.search(r'(?:HOS )?arrival time is (\d{2}:\d{2}|\d+)', next_event).group(1)
                    next_arrival = int(next_arrival_str.split(':')[0]) * 60 + int(next_arrival_str.split(':')[1]) if ':' in next_arrival_str else int(next_arrival_str) * 15
                    if event_end_time > next_arrival:
                        violated_events += 1
                        adjusted_duration = next_arrival - current_arrival
                        if not duration_format:  # No 'minutes', assume numeric duration represents 15-minute units
                            adjusted_duration = round(adjusted_duration / 15)
                        sorted_events[i] = re.sub(r'duration is \d+( minutes)?', f'duration is {adjusted_duration}' + (' minutes' if duration_format else ''), current_event)
                        
                        if self.is_departure:
                            adjusted_departure = self.minute_to_HHMM(next_arrival, duration_format)
                            sorted_events[i] = re.sub(r'departure time is (\d{2}:\d{2}|\d+)', f'departure time is {adjusted_departure}', current_event)
                else:
                    if event_end_time > 1440:
                        violated_events += 1
                        adjusted_duration = 1440 - current_arrival
                        if not duration_format:
                            adjusted_duration = round(adjusted_duration / 15)
                        sorted_events[i] = re.sub(r'duration is \d+( minutes)?', f'duration is {adjusted_duration}' + (' minutes' if duration_format else ''), current_event)
                        
                        if self.is_departure:
                            adjusted_departure = self.minute_to_HHMM(1440, duration_format)
                            sorted_events[i] = re.sub(r'departure time is (\d{2}:\d{2}|\d+)', f'departure time is {adjusted_departure}', current_event)
            else:
                ## fullfilled case: arrival only, adjust duration according to next event
                # constraints event logic when duration is zero
                if current_duration == 0 and re.search(r'departure time is (\d{2}:\d{2}|\d+)', current_event) is None:
                    if next_event:
                        next_arrival_str = re.search(r'(?:HOS )?arrival time is (\d{2}:\d{2}|\d+)', next_event).group(1)
                        next_arrival = int(next_arrival_str.split(':')[0]) * 60 + int(next_arrival_str.split(':')[1]) if ':' in next_arrival_str else int(next_arrival_str) * 15
                        adjusted_duration = next_arrival - current_arrival
                        if not duration_format:
                            adjusted_duration = round(adjusted_duration / 15)
                        sorted_events[i] = re.sub(r'duration is \d+( minutes)?', f'duration is {adjusted_duration}' + (' minutes' if duration_format else ''), current_event)
                        
                        if self.is_departure:
                            adjusted_departure = self.minute_to_HHMM(next_arrival, duration_format)
                            sorted_events[i] = re.sub(r'departure time is (\d{2}:\d{2}|\d+)', f'departure time is {adjusted_departure}', current_event)
                    else:
                        # No next event for HOS with zero duration
                        adjusted_duration = 1440 - current_arrival
                        if not duration_format:
                            adjusted_duration = round(adjusted_duration / 15)
                        sorted_events[i] = re.sub(r'duration is \d+( minutes)?', f'duration is {adjusted_duration}' + (' minutes' if duration_format else ''), current_event)
                        
                        if self.is_departure:
                            adjusted_departure = self.minute_to_HHMM(1440, duration_format)
                            sorted_events[i] = re.sub(r'departure time is (\d{2}:\d{2}|\d+)', f'departure time is {adjusted_departure}', current_event)
                        
                ## fullfilled case: departure only, adjust arrival time and duration according to previous event                
                # Adjusting arrival time when arrival equals departure in constraint events 
                elif prev_event and re.search(r'departure time is (\d{2}:\d{2}|\d+)', current_event) and current_duration == 0:
                    prev_arrival_str = re.search(r'(?:HOS )?arrival time is (\d{2}:\d{2}|\d+)', prev_event).group(1)
                    prev_duration_match = re.search(r'(?:HOS )?duration is (\d+)( minutes)?', prev_event)
                    prev_duration = int(prev_duration_match.group(1))
                    prev_duration_format = prev_duration_match.group(2)

                    if ':' in prev_arrival_str:
                        hours, minutes = map(int, prev_arrival_str.split(':'))
                        prev_arrival = hours * 60 + minutes
                    else:
                        prev_arrival = int(prev_arrival_str) * 15  

                    if not prev_duration_format:
                        prev_duration *= 15
                    
                    prev_end_time = prev_arrival + prev_duration
                    
                    # minutes format to 15-minute interval units
                    # if format is in HH:MM, recover from minutes to HH:MM
                    adjusted_time = self.minute_to_HHMM(prev_end_time, duration_format) # format to HH:MM
                    sorted_events[i] = re.sub(r'arrival time is (\d{2}:\d{2}|\d+)', f'arrival time is {adjusted_time}', current_event)
                    
                    # departure itself is constraint requirement, so no need to adjust
        if violated_events > 0:
            violated_trajectory = 1   
                     
        return sorted_events, violated_events, violated_trajectory
    
    def minute_to_HHMM(self, time_in_minutes, duration_format):
        ### applied to raw minutes to HH:MM format or 15-minute interval units
        ### work with departure time and duration time adjustments
        if not duration_format:
            adjusted_time = round(time_in_minutes / 15) # Convert to 15-minute interval units
        else:
            hours = time_in_minutes // 60
            minutes = time_in_minutes % 60
            adjusted_time = f'{hours:02}:{minutes:02}' # Convert to HH:MM format
            
        return adjusted_time
    
if __name__ == '__main__':
    
    # ################## VAE ##################
    # traj_input = ''
    # output_filename = ''
    
    # # ################## Transformer ##################
    # # traj_input = ''
    # # output_filename = ''
    
    # # ################## SeqGAN ##################
    # # traj_input = ''
    # # output_filename = ''
    
    # # ################## LSTM ##################
    # # traj_input = ''
    # # output_filename = ''
    
    # # ################## GRU ##################
    # # traj_input = ''
    # # output_filename = ''
    
    # ################## LLM ##################
    # # traj_input = ''
    # # output_filename = ''
    
    # ################## Constraints Specification ##################
    # hos_input = 'HOS_trial3_gridid_15interval_departure_small.data'
    
    # reorder = RandomHOSInsert(traj_input, output_filename, hos_input, is_departure=True, is_integrity_check=True)
    # reorder.read_and_process_formatted()
    
    # #### Eval ####
    # horizontal_n=100
    # max_locs=7500 # vocab of the dataset used; kitware vocab = 5700, not used here
    # seq_len=96
    # interval=15
    # npy_file_path = 'trial3_stays.npy'
    # ########################################
    # is_interval1 = True
    # use_grid_id1 = True
    # agent_id1=False # there is agent id in generated data, if not set false
    # ### GT data
    # traj_file_path2 = 'gt/trial3_gridid_15interval_medium.data' ## Ground truth data
    # is_interval2 = True
    # use_grid_id2 = True
    # agent_id2=False
    
    # ## Top K ##
    # top_k = 40
    # individualEval = IndividualEvalHOS(horizontal_n, seq_len, interval, npy_file_path, output_filename, traj_file_path2, top_k,
    #                                    hos_input, is_interval1, use_grid_id1, is_interval2, use_grid_id2, agent_id1, agent_id2)
    
    # d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, transit_norm, topk_transit_norm=individualEval.get_individual_jsds()
    
    # print(f'distance jsd: {d_jsd},  gradius jsd: {g_jsd},'  
    #     f'duration jsd: {du_jsd},  periodicity jsd: {p_jsd},  frequency jsd: {f_jsd}, location jsd: {l_jsd},' 
    #     f'transition matrix norm: {transit_norm}, top k transition matrix norm: {topk_transit_norm}')
    
    ##### Testing Trial 3 HOS W agent id using existing random permuted files No HOS W departure#####
    
    ################## VAE ##################
    traj_input = 'geolife_gen/geolife_vae_synthetic_traj.data'
    output_filename = 'geolife_ri/geolife_vae_departure_HOS.data'
    
    # ################## Transformer ##################
    # traj_input = 'geolife_gen/geolife_transformer_synthetic_traj.data'
    # output_filename = 'geolife_ri/geolife_transformer_departure_HOS.data'
    
    # ################## SeqGAN ##################
    # traj_input = 'geolife_gen/geolife_seqgan_gen_prompts.data'
    # output_filename = 'geolife_ri/geolife_seqgan_departure_HOS.data'
    
    # ################## LSTM ##################
    # traj_input = 'geolife_gen/geolife_lstm_synthetic_traj.data'
    # output_filename = 'geolife_ri/geolife_lstm_departure_HOS.data'
    
    # ################## GRU ##################
    # traj_input = 'geolife_gen/geolife_gru_synthetic_traj.data'
    # output_filename = 'geolife_ri/geolife_gru_departure_HOS.data'
    
    # ################# LLM ##################
    # traj_input = 'geolife_gen/geolife_gridid_15interval_LLM00_ep7.data'
    # output_filename = 'geolife_ri/geolife_traj_departure_HOS_LLM00_ep7.data'
    
    ################## HOS ##################
    hos_input = 'HOS_geolife_gridid_15interval_eventshuffle_departure.data'
    
    reorder = RandomHOSInsert(traj_input, output_filename, hos_input, is_departure=True, is_integrity_check=True)
    reorder.read_and_process_formatted()
    
    #### Eval ####
    horizontal_n=200
    max_locs=29600 # vocab of the dataset used; kitware vocab = 5700, not used here
    seq_len=96
    interval=15
    npy_file_path = 'geolife_staypoints.npy'
    ########################################
    is_interval1 = True
    use_grid_id1 = True
    agent_id1=False # there is agent id in generated data, if not set false
    ### GT data
    traj_file_path2 = 'gt/geolife_gridid_15interval.data' ## Ground truth data
    is_interval2 = True
    use_grid_id2 = True
    agent_id2=False
    
    ## Top K ##
    top_k = 40
    individualEval = IndividualEvalHOS(horizontal_n, seq_len, interval, npy_file_path, output_filename, traj_file_path2, top_k,
                                       hos_input, is_interval1, use_grid_id1, is_interval2, use_grid_id2, agent_id1, agent_id2)
    
    d_jsd,  g_jsd,  du_jsd,  p_jsd, l_jsd, f_jsd, transit_norm, topk_transit_norm=individualEval.get_individual_jsds()
    
    print(f'distance jsd: {d_jsd},  gradius jsd: {g_jsd},'  
    f'duration jsd: {du_jsd},  periodicity jsd: {p_jsd},  frequency jsd: {f_jsd}, location jsd: {l_jsd},' 
    f'transition matrix norm: {transit_norm}, top k transition matrix norm: {topk_transit_norm}')
    
