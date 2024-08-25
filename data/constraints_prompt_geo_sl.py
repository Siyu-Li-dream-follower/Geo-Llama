## In both arrival and departure, we removed the randomness in interval to avoid depature earlier than arrival
## avoid the time to be out of the day

import numpy as np
import pandas as pd
from collections import defaultdict
import random

class HOS_promptSL:
    def __init__(self, npy_file_path, output_filename, horizontal_n, include_agent_id=True, use_grid_id=False, is_interval=False, 
                 event_permute=False, feature_permute=False, is_longer=False):
        self.stays = np.load(npy_file_path, allow_pickle=True)
        self.min_lat = np.min(self.stays[:, 1])
        self.max_lat = np.max(self.stays[:, 1])
        self.min_lon = np.min(self.stays[:, 2])
        self.max_lon = np.max(self.stays[:, 2])
        self.horizontal_n = horizontal_n
        self.vertical_n = int((self.max_lat - self.min_lat) / (self.max_lon - self.min_lon) * horizontal_n)
        self.include_agent_id = include_agent_id
        self.use_grid_id = use_grid_id
        self.is_interval = is_interval
        self.event_permute = event_permute
        self.feature_permute = feature_permute
        self.is_longer = is_longer
        self.output_filename = output_filename

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

    def time_to_interval(self, timestamp):
        return timestamp.hour * 4 + timestamp.minute // 15

    def adjust_time_within_day(self, original_time, delta_minutes):
        ## avoid the time to be out of the day
        adjusted_time = original_time + pd.Timedelta(minutes=delta_minutes)
        if adjusted_time.date() != original_time.date():
            if delta_minutes > 0:
                adjusted_time = pd.Timestamp(year=original_time.year, month=original_time.month, day=original_time.day, hour=23, minute=59)
            else:
                adjusted_time = pd.Timestamp(year=original_time.year, month=original_time.month, day=original_time.day, hour=0, minute=0)
        return adjusted_time

    def process_events(self):
        grouped_data = defaultdict(list)
        for record in self.stays:
            agent_id, lat, lon, timestamp, duration = record
            date = timestamp.strftime('%Y-%m-%d')
            key = (agent_id, date)
            grouped_data[key].append(record)

        output_data = []
        for key, events in grouped_data.items():
            agent_id, date = key
            event_descriptions = []
            if self.include_agent_id:
                event_descriptions.append(f"agent id is {agent_id}")
            
            if self.is_longer:
                if len(events) < 5: # only sample from 4 and more events
                    continue
                sampled_events = random.sample(events, random.randint(5, len(events)))
            else:
                if len(events) < 4: # only sample from 2 at most events
                    continue
                sampled_events = random.sample(events, random.randint(3, 4))
                
            
            for event in sampled_events:
                _, lat, lon, timestamp, duration = event
                
                time_choice = random.choice(['arrival', 'departure', 'both'])  
                
                grid_id = self.coord2grid((lat, lon)) if self.use_grid_id else f"({lat},{lon})"
                
                ## In both arrival and departure, we removed the randomness in interval to avoid depature earlier than arrival
                if time_choice == 'arrival' or time_choice == 'both':
                    modified_arrival = self.adjust_time_within_day(timestamp, random.randint(-30, 30))
                    time_str = self.time_to_interval(modified_arrival) if self.is_interval else modified_arrival.strftime('%H:%M')
                    duration_str = f" duration is {duration // 15}" if self.is_interval else f" duration is {duration} minutes"
                    location_str = f" location is {grid_id}"
                    event_description = f"arrival time is {time_str},{location_str},{duration_str}"
                
                if time_choice == 'departure' or time_choice == 'both':
                    if time_choice == 'both':
                        departure_time = modified_arrival + pd.Timedelta(minutes=duration)
                    else:
                        departure_time = self.adjust_time_within_day(timestamp + pd.Timedelta(minutes=duration), random.randint(-30, 30))
                        event_description = f"location is {grid_id}," # initialize the event description again but not arrival duration info
                        
                    departure_str = self.time_to_interval(departure_time) if self.is_interval else departure_time.strftime('%H:%M')
                    event_description += f", departure time is {departure_str}" if time_choice == 'both' else f" departure time is {departure_str}"
                
                event_descriptions.append(event_description)

            output_data.append([f"{';'.join(event_descriptions)};"])
        
        if self.event_permute or self.feature_permute: # Event-wise, feature-wise permutation or both
            shuffled_trajectories = []
            
            for trajectory in output_data:
                shuffled_trajectory = self.shuffle_trajectory(trajectory)
                shuffled_trajectories.append(shuffled_trajectory)
            
            output_data = shuffled_trajectories
        
        with open(self.output_filename, 'w') as file:
            for data in output_data:
                file.write(f"{data}\n")

        return output_data

    def shuffle_trajectory(self, trajectory_list):
        if not trajectory_list or not isinstance(trajectory_list, list):
            return "Invalid trajectory data, not a list"
        
        # Take string from trajectory list
        trajectory = trajectory_list[0]
        
        # split the trajectory string into agent id parts and event parts
        trajectory = trajectory.rstrip(';')
        parts = trajectory.split(';')
        
        if self.include_agent_id:
            agent_id = parts[0] + ';'
            events = parts[1:]
            events = self.shuffle_events_features(events)
                
            shuffled_trajectory = [agent_id + ' ' + ' '.join(event + ('' if event.endswith(';') else ';') for event in events)]
            ###' ' + ' '.join if you want ' ' before first feature
        else:
            events = parts
            events = self.shuffle_events_features(events)
            
            shuffled_trajectory = [' ' + ' '.join(event + ('' if event.endswith(';') else ';') for event in events)] 
            ###' ' + ' '.join if you want ' ' before first feature
            
        return shuffled_trajectory
    
    def shuffle_events_features(self, events):
        # Shuffle the features in each event
        if self.feature_permute:
            new_events = []

            for event in events:
                # Remove the ending ';' if it exists
                event = event.rstrip(';')

                # Split event into features, assuming comma separation
                features = event.split(', ')
                random.shuffle(features)

                # Reassemble the shuffled event without adding a comma at the end
                shuffled_event = ', '.join(features)

                # Add the ending ';' to mark the end of the event
                new_events.append(shuffled_event + ';')

            events = new_events

        # Shuffle the events if enabled
        if self.event_permute:
            random.shuffle(events)

        return events  
  
    
if __name__ == '__main__':
    ############################# VOCAB OF Geolife IS 29600 IF YOU USE GRID ID WITH 200 HORIZONTAL N #############################
    ############################# The number of trajectories is: 7357 #############################
    ##################### To get different HOS prompts setups, pls simply change the flags #####################
    
    #################### Constraints Geolife WO agent id, W grid id, W 15min interval representation, W event permute W departure####################
    npy_file_path = ''
    output_filename = ''
    horizontal_n = 200
    hos_sampler = HOS_promptSL(npy_file_path, output_filename, horizontal_n, include_agent_id=False, use_grid_id=True, 
                             is_interval=True, event_permute=True, feature_permute=False, is_longer=True)
    processed_data = hos_sampler.process_events()