import random

def create_random_subsets(traj_path, hos_path, subset_ratio, output_traj_path, output_hos_path):
    
    with open(traj_path, 'r') as file:
        traj_lines = file.readlines()

    with open(hos_path, 'r') as file:
        hos_lines = file.readlines()

    # use the minimum length of the two datasets
    total_lines = min(len(traj_lines), len(hos_lines))

    # calculate the subset size
    subset_size = int(total_lines * subset_ratio)

    random_indices = random.sample(range(total_lines), subset_size)

    traj_subset = [traj_lines[index] for index in sorted(random_indices)]
    hos_subset = [hos_lines[index] for index in sorted(random_indices)]

    # write traj subset
    with open(output_traj_path, 'w') as file:
        file.writelines(traj_subset)

    # write corresponding hos subset
    with open(output_hos_path, 'w') as file:
        file.writelines(hos_subset)
        
def create_random_subsets_HOS(hos_path, subset_ratio, output_hos_path):

    with open(hos_path, 'r') as file:
        hos_lines = file.readlines()

    # use the minimum length of the two datasets
    total_lines = len(hos_lines)

    # calculate the subset size
    subset_size = int(total_lines * subset_ratio)

    random_indices = random.sample(range(total_lines), subset_size)

    hos_subset = [hos_lines[index] for index in sorted(random_indices)]

    # write hos subset
    with open(output_hos_path, 'w') as file:
        file.writelines(hos_subset)
        
if __name__ == '__main__':
    ########### Create a subset of trajectories and constraints with same indexing ###########
    traj_path = ''
    hos_path = ''
    subset_ratio = 0.05
    output_traj_path = ''
    output_hos_path = ''
    create_random_subsets(traj_path, hos_path, subset_ratio, output_traj_path, output_hos_path)
    
    ########### Create a subset of constraints ###########
    hos_path = ''
    subset_ratio = 0.05
    output_hos_path = ''
    create_random_subsets_HOS(hos_path, subset_ratio, output_hos_path)
