import os
## Only works with grid id, discretized time, not handling event permute, feature permute ##
class TrajectoryString2Prompt:
    def __init__(self, input_file, output_file, plus_one=False):
        self.input_file = input_file
        self.output_file = output_file
        self.plus_one = plus_one

    def read_gen_file(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        trajectories = []
        for line in lines:
            grid_ids = line.strip().split(' ')
            grid_ids = [int(s) + 1 if self.plus_one else int(s) for s in grid_ids]
            trajectories.append(grid_ids)
        return trajectories

    def convert_to_events(self, trajectories):
        all_events = []
        for trajectory in trajectories:
            events = []
            current_location = trajectory[0]
            arrival_time = 0
            duration = 1
            
            for timestep in range(1, len(trajectory)):
                if trajectory[timestep] == current_location:
                    duration += 1
                else:
                    event_description = f" arrival time is {arrival_time}, location is {current_location}, duration is {duration}"
                    events.append(event_description)
                    current_location = trajectory[timestep]
                    arrival_time = timestep
                    duration = 1
            
            # Add the last event
            event_description = f" arrival time is {arrival_time}, location is {current_location}, duration is {duration}"
            events.append(event_description)

            single_event = [f"{';'.join(events)};"]
            all_events.append(single_event)

        return all_events

    def write_events_to_file(self, all_events):
        with open(self.output_file, 'w') as f:
            for event in all_events:
                f.write(f"{event}\n")

    def process(self):
        trajectories = self.read_gen_file()
        events = self.convert_to_events(trajectories)
        self.write_events_to_file(events)

if __name__ == '__main__':
    ############################################################

    input_file = ''  
    output_file = ''     

    processor = TrajectoryString2Prompt(input_file, output_file, plus_one=True) # If Min Gen Grid ID is 0, plus one
    processor.process()
