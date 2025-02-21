from llm_agent.database.learning_db import LearningDB
from llm_agent.env.base_env import Observation, Action
from datasets import load_dataset

# Import get_task_type from alfworld_fewshots
from llm_agent.in_context.alfworld_fewshots import get_task_type

# Load the dataset
dataset_name = "Solaris99/AgentBank"
for subset_name in ['intercode_sql']:
    dataset = load_dataset(dataset_name, subset_name)

    # Now ingest the dataset
    db = LearningDB(f"./data/agentbank_{subset_name}/learning.db")
    
    # Process each split
    for split in dataset.keys():
        # Process each trajectory in the split
        for i, trajectory in enumerate(dataset[split]):
            conversations = trajectory['conversations']
            
            # Extract observations, thoughts/reasoning, and actions
            obs_list = []
            thought_list = []  
            act_list = []
            goal = None
            
            for conv in conversations:
                if conv['from'] == 'human':
                    # Extract observation from human messages
                    if conv['value'].startswith('Observation:'):
                        obs = conv['value'].replace('Observation: ', '')
                        obs_list.append(obs)
                    # Otherwise its the first message from the agent
                    else:
                        # Split on newline to separate initial observation from goal
                        if subset_name == 'alfworld':
                            parts = conv['value'].split('\n')
                            obs_list.append(parts[0])
                            goal = parts[2].replace('Your task is to:', '').strip()
                        else:
                            obs_list.append(conv['value'])
                            goal = conv['value']

                elif conv['from'] == 'gpt':
                    # Extract thought and action from GPT messages
                    message = conv['value']
                    thought = None
                    action = None
                    
                    # Remove all newline characters except for the one right before Action:
                    message = message.replace('\n', '')
                    message = message.replace('Action:', '\nAction:')

                    if subset_name == 'intercode_sql':
                        message = message.replace('```sql', '')
                        message = message.replace('```', '')

                    # Split on newline to separate thought from action
                    parts = message.split('\n')
                    for part in parts:
                        if part.startswith('Thought:'):
                            thought = part.replace('Thought: ', '')
                        elif part.startswith('Action:'):
                            action = part.replace('Action: ', '')
                    
                    if thought:
                        thought_list.append(thought)
                    else:
                        thought_list.append("")
                    if action:
                        act_list.append(action)
                    else:
                        act_list.append("")

            # The plan is the first sentence of the first thought
            if subset_name == 'alfworld':
                plan = thought_list[0].split('.')[0] + "."
                thought_list[0] = thought_list[0].replace(plan, '')[1:].strip()
            else:
                plan = ""

            # Put in one last observation indicating task success
            obs_list.append("Task completed successfully.")

            # Convert to Observation and Action objects
            obs_list = [Observation(o) for o in obs_list]
            act_list = [Action(a) for a in act_list]
            
            # Assume success if trajectory completed (give reward of 1 at end)
            rewards = [0] * (len(act_list)-1) + [1]

            if subset_name == 'alfworld':
                category = get_task_type(goal)
            else:
                category = "sql"
            
            # Store trajectory
            try:
                db.store_episode(
                    environment_id=f"agentbank_{split}_{i}",
                    goal=goal,  # Use goal if available, else Unknown
                    category=category,  # Use task_type if available
                    observations=obs_list,
                    reasoning=thought_list,
                    actions=act_list,
                    rewards=rewards,
                    plan=plan,
                    reflection=None,
                    summary=None
                )
            except Exception as e:
                print(f"Error storing episode {i} in split {split}: {e}")
                continue