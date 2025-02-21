from llm_agent.database.learning_db import LearningDB
from llm_agent.in_context.animation_fewshots import ANIMATION_FEWSHOTS
from llm_agent.env.base_env import Observation, Action

def ingest_animation_fewshots():
    db = LearningDB("./data/animation/learning.db")
    
    # Process each trajectory
    for i, elem in enumerate(ANIMATION_FEWSHOTS):
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
            actions=act_list,
            rewards=rewards,
            thoughts=thought_list
        )
