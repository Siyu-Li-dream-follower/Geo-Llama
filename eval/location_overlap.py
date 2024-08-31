import re

class LocationOverlap:    
    def __init__(self, traj_input1, traj_input2):
        self.traj_input1 = traj_input1
        self.traj_input2 = traj_input2

    def read_and_process_formatted(self):
        trajectories1 = self.read_trajectories(self.traj_input1)
        trajectories2 = self.read_trajectories(self.traj_input2)
        
        unique_locations1 = self.get_unique_locations(trajectories1)
        unique_locations2 = self.get_unique_locations(trajectories2)
        
        overlap_ratio = self.calculate_overlap_ratio(unique_locations1, unique_locations2)
        return overlap_ratio
    
    def read_trajectories(self, traj_input):
        trajectories = []
        with open(traj_input, 'r') as traj_data:
            for line in traj_data:
                trajectory = line.strip() 
                events = trajectory.strip("[]'").rstrip(';').rstrip('; ').split(';')
                trajectories.append(events)
        return trajectories
    
    def get_unique_locations(self, trajectories):
        unique_locations = set()
        for events in trajectories:
            for event in events:
                location = self.extract_location(event)
                unique_locations.add(location)
        return unique_locations
    
    def extract_location(self, event):
        location_match = re.search(r'location is (\d+)', event)
        if location_match:
            return location_match.group(1)
        return None
    
    def calculate_overlap_ratio(self, locations1, locations2):
        intersection = locations1.intersection(locations2)
        min_unique = min(len(locations1), len(locations2))
        max_unique = max(len(locations1), len(locations2))
        overlap_ratio = len(intersection) / max_unique if max_unique != 0 else 0
        return overlap_ratio

if __name__ == '__main__':
    #### Gen data or constraints
    traj_input1 = ''
    #### Ground Truth
    traj_input2 = ''

    location_overlap = LocationOverlap(traj_input1, traj_input2)
    overlap_ratio = location_overlap.read_and_process_formatted()

    print(f"Overlap Ratio: {overlap_ratio}")
                        
    
