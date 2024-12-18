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
    attempt_dict = {}
    success_attempts_dict = {}

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

            attempts = []
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'Step 20 of 20' in line or 'Reward: 1' in line:
                    # Check if previous line had Reward: 1 or 0
                    prev_line = lines[i-1]
                    if 'Reward: 1' in line:
                        attempts.append(1)
                    elif 'Reward: 0' in prev_line:
                        attempts.append(0)

            success_attempts = attempts.index(1) + 1 if 1 in attempts else -1
            success_attempts_dict[success_attempts] = success_attempts_dict.get(success_attempts, 0) + 1
            
            for i in range(len(attempts)):
                if i+1 not in attempt_dict:
                    attempt_dict[i+1] = []
                attempt_dict[i+1].append(attempts[i])

    # Continue if attempt_dict is empty
    if not attempt_dict:
        continue

    # Calculate success rate
    success_rate = (successful_episodes / total_episodes) * 100

    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {success_rate:.2f}%")

    print("Success dict", dict(sorted(success_count_dict.items(), key=lambda item: item[0]))) 
    print("Failure dict", dict(sorted(failure_count_dict.items(), key=lambda item: -1*item[0])))

    print("Average success per attempt", {k: sum(v)/len(v) for k,v in attempt_dict.items()})

    print("Overall average success", sum(sum(v)/len(v) for v in attempt_dict.values())/len(attempt_dict.values()))

    # Get weighted average of inverse of keys of success_attempts_dict, weighted by values.
    # Where the key is -1, set the inverse to 0.
    weighted_sum = sum((1/k if k > 0 else 0) * v for k,v in success_attempts_dict.items())
    total_weight = sum(success_attempts_dict.values())
    weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
    print("Weighted average of inverse attempts:", weighted_avg)

    # Print the success rate on each attempt of the ones that haven't succeeded yet
    cumulative_success_rate = 0
    cumulative_success_rate_dict = {}
    for k,v in sorted(success_attempts_dict.items(), key=lambda item: item[0]):
        if k > 0:
            cumulative_success_rate += v/sum(success_attempts_dict.values())
            cumulative_success_rate_dict[k] = cumulative_success_rate
    print("Cumulative success rate dict:", cumulative_success_rate_dict)

