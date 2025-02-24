from llm_agent.database.learning_db import LearningDB
from llm_agent.in_context.intercode_fewshots import parse_all_fewshots
from llm_agent.env.base_env import Observation, Action
import os

def ingest_intercode_fewshots():
    db = LearningDB("./data/intercode_sql/learning.db")
    # Create the folder if it doesn't exist
    os.makedirs("./data/intercode_sql", exist_ok=True)
    
    # Process SQL fewshots
    i = 0
    react_examples = parse_all_fewshots("SQL")
    for example in react_examples:
        goal, plan, obs_list, thought_list, act_list = example
        
        # Extract trajectory components
        environment_id = f"sql_exemplar_{i}"
        rewards = [0] * (len(act_list)-1) + [1]

        # Convert obs_list and act_list to Observation and Action objects
        obs_list = [Observation(o) for o in obs_list]
        act_list = [Action(a) for a in act_list]
        
        # Store trajectory
        db.store_episode(
            environment_id=environment_id,
            goal=goal,
            category="SQL",
            observations=obs_list,
            reasoning=thought_list,
            actions=act_list,
            rewards=rewards,
            plan=None,
            reflection=None,
            summary=None
        )
        i += 1

        print("Goal: ", goal)

    # Separate database for BASH
    db = LearningDB("./data/intercode_bash/learning.db")
    # Create the folder if it doesn't exist
    os.makedirs("./data/intercode_bash", exist_ok=True)
    
    # Process BASH fewshots 
    i = 0
    react_examples = parse_all_fewshots("BASH")
    for example in react_examples:
        goal, plan, obs_list, thought_list, act_list = example
        
        # Extract trajectory components
        environment_id = f"bash_exemplar_{i}"
        rewards = [0] * (len(act_list)-1) + [1]

        # Convert obs_list and act_list to Observation and Action objects
        obs_list = [Observation(o) for o in obs_list]
        act_list = [Action(a) for a in act_list]
        
        # Store trajectory
        db.store_episode(
            environment_id=environment_id,
            goal=goal,
            category="BASH", 
            observations=obs_list,
            reasoning=thought_list,
            actions=act_list,
            rewards=rewards,
            plan=None,
            reflection=None,
            summary=None
        )
        i += 1

        print("Goal: ", goal)

if __name__ == "__main__":
    ingest_intercode_fewshots()
