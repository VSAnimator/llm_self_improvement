import os
import glob
import numpy as np

# Get all nested subfolders containing txt files
root_dir = "logs/episodes"
# Walk through directory tree and find all folders containing txt files
txt_folders = set()
for dirpath, dirnames, filenames in os.walk(root_dir):
    if any(f.endswith('.txt') for f in filenames):
        txt_folders.add(dirpath)

print("Folders containing txt files:")
average_success_rates = {}
cumulative_success_rates = {}
for folder in sorted(txt_folders):
    #if "bird_gold" not in folder or "backups" in folder or "4o-mini" in folder:
    # Table 1: if "bird_gold" not in folder or "cont" in folder or "4o-mini" in folder:
    # Table 2: if "alfworld" not in folder or "agentbank" not in folder or "rap_flex" not in folder:
    # Table 3: if "intercode_sql" not in folder or "bird_intercode" not in folder or "retry" not in folder:
    # Table 4: if "intercode_sql" not in folder or "bird_gold" not in folder or "cont" not in folder:
    #if "intercode_sql" not in folder or "bird_gold" not in folder or "gpt-4o-mini" not in folder:# or "4o-mini" not in folder:
    #if 'intercode_sql' not in folder or "gold" not in folder or "chroma" in folder or "4o-mini" not in folder or 'copy' in folder:
    #if 'wordcraft' not in folder or "4o-mini" not in folder or "backups" in folder or "copy" in folder or "4tries" not in folder:
    #if 'spider' not in folder or "4o-mini" not in folder or "backups" in folder or "copy" in folder or "trial" not in folder or "test" not in folder or "30ic" not in folder:
    if "alfworld" not in folder or "test" not in folder or "4o-mini" not in folder or "trial" not in folder or "3ic" not in folder:
        continue
    print(folder)
    # Get all episode files
    episode_files = glob.glob(f'{folder}/*.txt')

    # Filter files which can be cast to int
    episode_files = [f for f in episode_files if f.split('/')[-1].split('.')[0].isdigit()]
    # Filter out files with names >= 4800
    episode_files = [f for f in episode_files if int(f.split('/')[-1].split('.')[0]) < 4800]

    total_episodes = len(episode_files)
    successful_episodes = 0

    success_count_dict = {}
    failure_count_dict = {}
    attempt_dict = {}
    success_attempts_dict = {}
    final_rewards = []

    # Process each episode file
    # Sort files numerically
    # Try to cast file numbers to int, skip files where it fails
    episode_files = [f for f in episode_files if f.split('/')[-1].split('.')[0].isdigit()]
    episode_files = sorted(episode_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for episode_file in episode_files:
        with open(episode_file, 'r') as f:
            content = f.read()
            # Get final reward in file
            final_reward = content.split('Reward: ')[-1].split('\n')[0]
            # Cast to float if valid, otherwise 0
            final_reward = float(final_reward) if final_reward.replace('.', '', 1).isdigit() else 0
            final_rewards.append(final_reward)
            # Look for final reward of 1 
            if 'Reward: 1' in content:
                successful_episodes += 1
            # If Reward: 1 is not in the file, and Step 4 is not in the file, then skip and decrement the total count
            if 'Reward: 1' not in content and 'Step 4' not in content:
                total_episodes -= 1
                continue
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
                if 'Step 20 of 20' in line or 'Step 30 of 30' in line or 'Reward: 1' in line:
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

    for i in range(0, len(final_rewards), 200):
        print(f"Average success rate over {i+200} episodes:", np.mean(np.array(final_rewards[i:i+200]) > 0.99))


    continue
    print("Success dict", dict(sorted(success_count_dict.items(), key=lambda item: item[0]))) 
    print("Failure dict", dict(sorted(failure_count_dict.items(), key=lambda item: -1*item[0])))

    print("Average success per attempt", {k: sum(v)/len(v) for k,v in attempt_dict.items()})
    average_success_rates[folder] = {k: sum(v)/len(v) for k,v in attempt_dict.items()}

    print("Overall average success", sum(sum(v)/len(v) for v in attempt_dict.values())/len(attempt_dict.values()))

    print("Success attempts dict", success_attempts_dict)

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
    cumulative_success_rates[folder] = cumulative_success_rate_dict

    print("Avg final reward:", sum(final_rewards)/len(final_rewards))

    # Also print the average final reward over each segment of 100 episodes
    
    '''
    for i in range(0, len(final_rewards), 10):
        print(f"Average final reward over {i+10} episodes:", np.mean(final_rewards[:i+10]))

    for i in range(0, len(final_rewards), 10):
        print(f"Average success rate over {i+10} episodes:", np.mean((np.array(final_rewards[:i+10]) == 1)))
    '''
    print(f"Last 100 episode average success rate:", np.mean(np.array(final_rewards[-100:]) == 1))
    
    '''
    # Try a different version which is tracking squared hundreds
    for i in range(10):
        print(f"Average final reward over {i}th interval:", np.mean(np.array(final_rewards[(i**2)*50:((i+1)**2)*50])))

    for i in range(0, len(final_rewards), 100):
        print(f"Average success rate over {i+100} episodes:", np.mean(np.array(final_rewards[i:i+100]) > 0.99))

    for i in range(5):
        print(f"Average success rate over {i}th interval:", np.mean(np.array(final_rewards[(i**2)*100:((i+1)**2)*100]) > 0.99))
    '''

    # if there are 2000 episodes, plot a running average of the final rewards
    if len(final_rewards) >= 28000:
        final_rewards_array = np.array(final_rewards)
        running_avg = np.array(final_rewards).copy()
        start = 30
        for i in range(start, len(running_avg)):
            #running_avg[i] = alpha * final_rewards[i] + (1 - alpha) * running_avg[i-1]
            running_avg[i] = np.mean(final_rewards_array[:i])
        running_avg = running_avg[start:]
        
        import matplotlib.pyplot as plt
        # Create plot
        plt.figure(figsize=(10,5))
        plt.plot(range(len(running_avg)), running_avg)
        plt.xlabel('Episode')
        plt.ylabel('Average Final Reward')
        plt.title('Running Average of Final Rewards (Window Size 100)')
        plt.grid(True)
        plt.show()

        # Also make plot for success rate
        start=100
        success_rate_array = (np.array(final_rewards) > 0.99).astype(int).astype(float)
        print(success_rate_array)
        print(np.mean(success_rate_array))
        running_avg_success_rate = success_rate_array.copy()
        for i in range(start, len(running_avg_success_rate)):
            running_avg_success_rate[i] = np.mean(success_rate_array[:i])
        running_avg_success_rate = running_avg_success_rate[start:]

        # Create plot
        plt.figure(figsize=(10,5))
        plt.plot(range(len(running_avg_success_rate)), running_avg_success_rate)
        plt.xlabel('Episode')
        plt.ylabel('Average Success Rate')
        plt.title('Running Average of Success Rate (Window Size 100)')
        plt.grid(True)
        plt.show()

# Plot average and cumulative success rates in two separate plots
import matplotlib.pyplot as plt

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

for folder in average_success_rates:
    for i in range(1,6):
        if i not in average_success_rates[folder]:
            average_success_rates[folder][i] = average_success_rates[folder][i-1]
        if i not in cumulative_success_rates[folder]:
            cumulative_success_rates[folder][i] = cumulative_success_rates[folder][i-1]

# Sort average and cumulative success rates by attempt
for folder in average_success_rates:
    average_success_rates[folder] = dict(sorted(average_success_rates[folder].items()))
    cumulative_success_rates[folder] = dict(sorted(cumulative_success_rates[folder].items()))

# Plot average success rates
for folder in average_success_rates:
    label = folder.split('/')[-3]
    if label == "alfworld":
        continue
    elif label == "expel_train":
        label = "react + example + reflection"
    elif label == "vanilla_train":
        label = "react + example"
    elif label == "zero_shot_train":
        label = "react"
    ax1.plot(list(average_success_rates[folder].keys()), 
             list(average_success_rates[folder].values()),
             label=label)
ax1.set_xlabel('Attempt')
ax1.set_ylabel('Success Rate') 
ax1.set_title('Average Success Rate by Attempt (gpt-4o-mini)')
ax1.legend()

# Plot cumulative success rates
for folder in cumulative_success_rates:
    label = folder.split('/')[-3]
    if label == "alfworld":
        continue
    elif label == "expel_train":
        label = "react + example + reflection"
    elif label == "vanilla_train":
        label = "react + example"
    elif label == "zero_shot_train":
        label = "react"
    ax2.plot(list(cumulative_success_rates[folder].keys()),
             list(cumulative_success_rates[folder].values()),
             label=label)
ax2.set_xlabel('Attempt')
ax2.set_ylabel('Success Rate')
ax2.set_title('Cumulative Success Rate (gpt-4o-mini)')
ax2.legend()
plt.tight_layout()
plt.savefig('success_rates.png')
plt.close()


