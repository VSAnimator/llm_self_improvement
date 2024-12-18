import os
import glob

# Get all nested subfolders containing txt files
root_dir = "logs/episodes"
# Walk through directory tree and find all folders containing txt files
txt_folders = set()
for dirpath, dirnames, filenames in os.walk(root_dir):
    if any(f.endswith('.txt') for f in filenames):
        txt_folders.add(dirpath)

print("Folders containing txt files:")
for folder in sorted(txt_folders):
    print(folder)
    # Get all episode files
    episode_files = glob.glob(f'{folder}/*.txt')

    total_episodes = len(episode_files)
    successful_episodes = 0

    success_count_dict = {}
    failure_count_dict = {}

    # Process each episode file
    for episode_file in episode_files:
        with open(episode_file, 'r') as f:
            content = f.read()
            # Look for final reward of 1 
            if 'Reward: 1' in content:
                successful_episodes += 1
            # Figure out the try in which it succeeded
            # Get count of how many times Final reward: 0 shows up
            #zero_count = content.count('Final reward: 0')
            zero_count = content.count('Reward: 0\nStep 20 of 20')
            one_count = content.count('Reward: 1')
            success_count_dict[one_count] = success_count_dict.get(one_count, 0) + 1
            failure_count_dict[zero_count] = failure_count_dict.get(zero_count, 0) + 1
    # Calculate success rate
    success_rate = (successful_episodes / total_episodes) * 100

    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {success_rate:.2f}%")

    print(success_count_dict)
    print(failure_count_dict)


