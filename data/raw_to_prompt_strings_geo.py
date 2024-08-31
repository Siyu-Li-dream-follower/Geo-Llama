import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from collections import defaultdict
import random
from datetime import datetime, timedelta

class GeoLifePreprocessingDF(object):
    
    def __init__(self, geo_staypoints_path, npy_file_path, traj_prompt_filename, horizontal_n, include_agent_id=False, 
                 use_grid_id=False, is_interval=False, event_permute=False, feature_permute=False, is_departure=False, drop_single=False):
        self.input_path = geo_staypoints_path
        self.npy_file_path = npy_file_path
        self.drop_single = drop_single
        self.df = self.read_df_from_file()
        self.stays = self.df_to_npy()
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
        self.is_departure = is_departure
        self.output_filename = traj_prompt_filename
        
        
    def read_df_from_file(self):
        input_format = self.input_path.split('.')[-1]
        if input_format == 'csv':
            df = pd.read_csv(self.input_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            if 'arrival_time' in df.columns:
                df['arrival_time'] = pd.to_datetime(df['arrival_time'])
            if 'leave_time' in df.columns:
                df['leave_time'] = pd.to_datetime(df['leave_time'])
        elif input_format == 'h5':
            df = pd.read_hdf(self.input_path, 'data')
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            if 'arrival_time' in df.columns:
                df['arrival_time'] = pd.to_datetime(df['arrival_time'])
            if 'leave_time' in df.columns:
                df['leave_time'] = pd.to_datetime(df['leave_time'])
        else:
            sys.exit('Input file format not supported!')
        return df
    
    def staypoints_processing(self):
        # Iterate through the DataFrame and update times within the same day
        df = self.df
        df = df.reset_index(drop=True)
        df['arrival_date'] = df['arrival_time'].dt.date
        
        for i in range(len(df) - 1):
            
            if df.loc[i, 'user_id'] == df.loc[i + 1, 'user_id'] and df.loc[i, 'arrival_date'] == df.loc[i + 1, 'arrival_date']:
                
                if df.loc[i, 'leave_time'] < df.loc[i + 1, 'arrival_time']:
                    df.loc[i, 'leave_time'] = df.loc[i + 1, 'arrival_time']
                    df.loc[i, 'stay_time'] = (df.loc[i, 'leave_time'] - df.loc[i, 'arrival_time']).total_seconds()

        # Remove rows with stay_time == 0
        df = df[df['stay_time'] > 900].reset_index(drop=True)
        
        # Remove staypoints if a user has only one staypoint on a given day
        if self.drop_single:
            df = df.groupby(['user_id', 'arrival_date']).filter(lambda x: len(x) > 1).reset_index(drop=True)

        # Drop the 'traj_id' and 'user_id' columns
        # df = df.drop(columns=['traj_id', 'user_id'])
        df = df.drop(columns=['traj_id'])

        return df
    
    def df_to_npy(self):
        df = self.staypoints_processing()
        
        # Rename columns
        df = df.rename(columns={
            'user_id': 'agent',
            'lat': 'latitude',
            'long': 'longitude',
            'arrival_time': 'start_timestamp',
            'leave_time': 'stop_timestamp',
            'stay_time': 'duration'
        })
        
        # Calculate duration in integer minutes
        df['duration'] = df['duration'].astype(float)
        df['duration'] = np.round(df['duration'] / 60).astype(int)
        
        # Select the columns we need
        data = df[['agent', 'latitude', 'longitude', 'start_timestamp', 'duration']].copy()

        # Convert to numpy array
        npy_array = data.to_numpy()
        
        # Print the total number of unique agents
        unique_agents = df['agent'].nunique()
        print("Number of unique agents:", unique_agents)
        # print(data.head(50))
        print("Number of staypoints:", data.shape[0])
        # Save as npy file
        np.save(self.npy_file_path, npy_array)
        
        return npy_array
        
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
        # Convert timestamp to a 15-minute interval index
        return timestamp.hour * 4 + timestamp.minute // 15

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
            for event in events:
                _, lat, lon, timestamp, duration = event
                
                if self.is_interval:
                    time_interval = self.time_to_interval(timestamp)
                    duration_interval = duration // 15  # Convert duration to 15-minute intervals
                    departure_time = max(time_interval + duration_interval, 96) # departure time not exceed today
                else:
                    time_interval = timestamp.strftime('%H:%M')
                    duration_interval = f"{duration} minutes"
                    departure_time = self.calculate_departure_time(time_interval, duration)
                    
                location = f"location is {self.coord2grid((lat, lon)) if self.use_grid_id else f'({lat},{lon})'}"
                
                if self.is_departure:
                    departure_time = f"departure time is {departure_time}"
                    event_description = f"arrival time is {time_interval}, {location}, duration is {duration_interval}, {departure_time}"
                else:
                    event_description = f"arrival time is {time_interval}, {location}, duration is {duration_interval}"
                    
                event_descriptions.append(event_description)
                
            single_event = [f"{';'.join(event_descriptions)};"]
            single_event = self.shuffle_trajectory(single_event)
            output_data.append(single_event)
        
        # Save data to a file
        with open(self.output_filename, 'w') as file:
            for data in output_data:
                file.write(f"{data}\n")
        
        print('The number of trajectories in Geolife is:', len(output_data))
        return output_data
    
    def calculate_departure_time(self, time_interval, duration):
        # start time representations
        start_time = datetime.strptime(time_interval, '%H:%M')
        
        # duration (minutes) to timedelta
        duration_td = timedelta(minutes=int(duration))
        
        # departure time calculation: hour: minute
        departure_time = start_time + duration_td
        
        # Ensure departure time does not exceed the last minute of the day
        end_of_day = start_time.replace(hour=23, minute=59)  # Set to 23:59 on the same day
        
        if departure_time > end_of_day:
            departure_time = end_of_day
            
        departure_time_formatted = departure_time.strftime('%H:%M')
        
        return departure_time_formatted
    
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
            ### For handling convenience, no space before first feature is easier.
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
    ##### drop_single means drop trajectories with only xxx staypoint/events
    geo_staypoints_path = ''
    npy_file_path = ''
    traj_prompt_filename = ''
    horizontal_n = 400
    geolife_process = GeoLifePreprocessingDF(geo_staypoints_path, npy_file_path, traj_prompt_filename, horizontal_n, 
                                             include_agent_id=False, use_grid_id=True, is_interval=True, event_permute=True, 
                                             feature_permute=False, is_departure=False, drop_single=True)
    geolife_staypoints = geolife_process.process_events()
    print(f'The Vocab of Geolife is:{geolife_process.horizontal_n * geolife_process.vertical_n}')
    
