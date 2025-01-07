from llm_agent.database.learning_db import LearningDB
from llm_agent.in_context.webshop_fewshots import WEBSHOP_FEWSHOTS
# Import wrapper classes for obs and act
from llm_agent.env.base_env import Observation, Action

def ingest_webshop_fewshots():
    db = LearningDB("./data/webshop/learning.db")
     # Process each trajectory
    i = 0
    for elem in WEBSHOP_FEWSHOTS:
        print(elem)
        goal, obs_list, thought_list, act_list = elem
        # Extract trajectory components
        environment_id = f"exemplar_{i}"
        rewards = [0] * (len(act_list)-1) + [1]

        # Convert obs_list and act_list to Observation and Action objects
        obs_list = [Observation(o) for o in obs_list]
        act_list = [Action(a) for a in act_list]
        
        # Store trajectory using store_episode
        db.store_episode(
            environment_id=environment_id,
            goal=goal,
            category="None",
            observations=obs_list,
            reasoning=thought_list,
            actions=act_list,
            rewards=rewards,
            plan=None,
            reflection=None,
            summary=None
        )

        i += 1

if __name__ == "__main__":
    ingest_webshop_fewshots()
