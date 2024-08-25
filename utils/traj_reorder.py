import numpy as np
import re
import itertools
import math
import ast
#### formatted output: patern_str = r'arrival time is (\d+), location is (\d+), duration is (\d+)'
#### each trajectory in the list formatted output is:
#### 'arrival time is (\d+), location is (\d+), duration is (\d+); arrival time is (\d+), location is (\d+), duration is (\d+); '

class TrajectoryReorder:    
    def __init__(self, traj_input, output_filename, satisfied_indices= None, hos_file=None, traj_file_mode=False, include_agent_id=False, keep_agent_id= False, is_departure=False):
            
            self.traj_input = traj_input
            self.include_agent_id = include_agent_id
            self.keep_agent_id = keep_agent_id
            self.is_departure = is_departure 
            self.traj_file_mode = traj_file_mode
            self.output_filename = output_filename
            
            if satisfied_indices is not None:
                if not isinstance(satisfied_indices, list):
                # if not list, then it should be a file path
                    with open(satisfied_indices, 'r') as file:
                        # read from file then convert to list
                        self.satisfied_indices = list(map(int, file.read().split()))
                else:
                    self.satisfied_indices = satisfied_indices
            
            if hos_file is not None:
                self.hos_file = hos_file
                self.is_hos = True
            else:
                self.is_hos = False
            
    def read_and_process_formatted(self):
        processed_trajectories = []
        if self.traj_file_mode:
            if self.is_hos:
                violated_trajectories = 0
                total_violated_events = 0
                trajectories = []
                hoses = []
                
                with open(self.traj_input, 'r') as traj_data:
                    
                    for line in traj_data:
                        trajectory = line.strip()  
                        trajectories.append(trajectory)
                        
                with open(self.hos_file, 'r') as hos_data:

                    for index, line in enumerate(hos_data):
                        if index in self.satisfied_indices:  
                            hos = line.strip()
                            hoses.append(hos)
                       
                for trajectory, hos in zip(trajectories, hoses):
                    # print(trajectory)
                    events = trajectory.strip("[]'").rstrip(';').rstrip('; ').split(';')
                    hos_events = hos.strip("[]'").rstrip(';').rstrip('; ').split(';')
                    # print(events[0])
                    if self.include_agent_id:
                        agent_id = events[0]
                        events = events[1:]
                        hos_events = hos_events[1:]
                      
                    hos_event_num = len(hos_events)
                    processed_events, violated_events, violated_trajectory = self.process_trajectory_with_hos(events, hos_event_num)
                    violated_trajectories += violated_trajectory
                    total_violated_events += violated_events
                    
                    if self.keep_agent_id and self.include_agent_id:
                            processed_events = f"{agent_id}; {processed_events}"
                            
                    processed_trajectories.append(processed_events)
                    
                print(f"Total violated events in trajectory: {total_violated_events}")
                print(f"Violated trajectories: {violated_trajectories}")
            else:
                violated_trajectories = 0
                total_violated_events = 0
                trajectories = []
                with open(self.traj_input, 'r') as traj_data:
                    for line in traj_data:
                        trajectory = line.strip()  
                        trajectories.append(trajectory)
                        
                for trajectory in trajectories:
                    events = trajectory.strip("[]'").rstrip(';').rstrip('; ').split(';')
                        
                    if self.include_agent_id:
                        agent_id = events[0]
                        events = events[1:]
                        
                    processed_events, violated_events, violated_trajectory = self.process_trajectory(events)
                    violated_trajectories += violated_trajectory
                    total_violated_events += violated_events
                    
                    if self.keep_agent_id and self.include_agent_id:
                        processed_events = f"{agent_id}; {processed_events}"
                        
                    processed_trajectories.append(processed_events)
                    
                print(f"Total violated events in trajectory: {total_violated_events}")
                print(f"Violated trajectories: {violated_trajectories}")
        else: ### traj_file_mode = False , traj are not path, but hos still read from path
            if self.is_hos:
                violated_trajectories = 0
                total_violated_events = 0
                hoses = []
                
                with open(self.hos_file, 'r') as hos_data:

                    for index, line in enumerate(hos_data):
                        if index in self.satisfied_indices:  
                            hos = line.strip()
                            hoses.append(hos)
                            
                for trajectory, hos in zip(self.traj_input, hoses):
                    events = trajectory.rstrip(';').rstrip('; ').split(';')
                    hos_events = hos.strip("[]'").rstrip(';').rstrip('; ').split(';') 
                    if self.include_agent_id:
                        agent_id = events[0]
                        events = events[1:]
                        hos_events = hos_events[1:]
                        
                    hos_event_num = len(hos_events)
                    processed_events, violated_events, violated_trajectory = self.process_trajectory_with_hos(events, hos_event_num)
                    violated_trajectories += violated_trajectory
                    total_violated_events += violated_events
                    
                    if self.keep_agent_id and self.include_agent_id:
                            processed_events = f"{agent_id}; {processed_events}"
                            
                    processed_trajectories.append(processed_events)
                
                print(f"Total violated events in trajectory: {total_violated_events}")
                print(f"Violated trajectories: {violated_trajectories}")
                    
            else:
                violated_trajectories = 0
                total_violated_events = 0
                for trajectory in self.traj_input:
                    events = trajectory.rstrip(';').rstrip('; ').split(';')
                        
                    if self.include_agent_id:
                        agent_id = events[0]
                        events = events[1:]
                        
                    processed_events, violated_events, violated_trajectory = self.process_trajectory(events)
                    violated_trajectories += violated_trajectory
                    total_violated_events += violated_events
                    
                    if self.keep_agent_id and self.include_agent_id:
                        processed_events = f"{agent_id}; {processed_events}"
                        
                    processed_trajectories.append(processed_events)
                    
                print(f"Total violated events in trajectory: {total_violated_events}")
                print(f"Violated trajectories: {violated_trajectories}") 
        
        with open(self.output_filename, 'w') as file:
            for data in processed_trajectories:
                file.write(f"['{data};']\n")
                
        return processed_trajectories
        
    def process_trajectory(self, events):
       
        processed_events = [self.parse_event(event) for event in events]
        
        sorted_events = self.sort_events_by_arrival_time(processed_events)
        adjusted_events, violated_events, violated_trajectory = self.adjust_events_timing(sorted_events)
        adjusted_events = '; '.join(event for event in adjusted_events)
        
        return adjusted_events, violated_events, violated_trajectory
    
    def process_trajectory_with_hos(self, events, hos_event_num):
        processed_events = []
        
        for index, event in enumerate(events):
        
            if index < hos_event_num:
                processed_event = self.parse_hos_event(event)
            else:
                processed_event = self.parse_event(event)
            
            processed_events.append(processed_event)
        
        sorted_events = self.sort_events_by_arrival_time(processed_events)
        adjusted_events, violated_events, violated_trajectory = self.adjust_events_timing(sorted_events)
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
                f"{features.get('duration')}",
                f"{features.get('departure_time')}"
            ]
        else:
            ordered_features = [
                f"{features.get('arrival_time')}",
                f"{features.get('location')}",
                f"{features.get('duration')}"
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
                f"HOS {features.get('duration','duration is 0')}", ## HOS duration is 0 if not provided
                f"HOS {features.get('departure_time')}" ## departure can be missing
            ]
        else:
            ordered_features = [
                f"HOS {features.get('arrival_time')}",
                f"HOS {features.get('location')}",
                f"HOS {features.get('duration','duration is 0')}" ## HOS duration is 0 if not provided
            ]
        
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
                        # No next event for constraints with zero duration
                        adjusted_duration = 1440 - current_arrival
                        if not duration_format:
                            adjusted_duration = round(adjusted_duration / 15)
                        sorted_events[i] = re.sub(r'duration is \d+( minutes)?', f'duration is {adjusted_duration}' + (' minutes' if duration_format else ''), current_event)
                        
                        if self.is_departure:
                            adjusted_departure = self.minute_to_HHMM(1440, duration_format)
                            sorted_events[i] = re.sub(r'departure time is (\d{2}:\d{2}|\d+)', f'departure time is {adjusted_departure}', current_event)
                        
                ## fullfilled case: departure only, adjust arrival time and duration according to previous event                
                # Adjusting arrival time when arrival equals departure in constraints events 
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
                    
                    # departure itself is constraints requirement, so no need to adjust
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
    ###### constraints and Trajectory Should simultaneously have agent id or not have agent id ######
    ###### constraints and Trajectory Should have exactly the same setup and item number ######
    ###### traj_file_mode if called in sampling function, should be False; save then reordering set true ######
    
    ##### Testing Geolife under constraints W agent id using existing random permuted files No HOS W departure#####
    traj_input = ''
    output_filename = ''
    satisfied_indices = None
    traj_file_mode = True
    reorder = TrajectoryReorder(traj_input, output_filename, satisfied_indices, hos_file=None, traj_file_mode=True, include_agent_id=False, keep_agent_id=False, is_departure=False)
    reorder.read_and_process_formatted()
    
    ##### Testing Geolife under constraints W agent id using LLM generated files WO HOS WO Departure #####
    ##### traj_file_mode=False, include_agent_id=False, keep_agent_id= False, is_departure=False)
    traj_input = ''
    output_filename = ''
    satisfied_indices = ''
    traj_file_mode = True
    hos_file = ''
    reorder = TrajectoryReorder(traj_input, output_filename, satisfied_indices, hos_file=hos_file, traj_file_mode=True, include_agent_id=False, keep_agent_id=False, is_departure=True)
    reorder.read_and_process_formatted()
    reorder = TrajectoryReorder(traj_input, output_filename, satisfied_indices, hos_file=hos_file, traj_file_mode=True, include_agent_id=False, keep_agent_id=False, is_departure=True)
    reorder.read_and_process_formatted()
