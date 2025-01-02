from llm_agent.database.learning_db import LearningDB
from llm_agent.in_context.alfworld_fewshots import PARSED_FEWSHOTS_LISTS
from llm_agent.in_context.alfworld_fewshots_trad import FEWSHOTS_TRAD
# Import wrapper classes for obs and act
from llm_agent.env.base_env import Observation, Action

def ingest_alfworld_fewshots_expel():
    # Initialize database
    db = LearningDB("./data/alfworld_expel/learning.db")
    
    # Process each trajectory
    i = 0
    for task, trajectories in PARSED_FEWSHOTS_LISTS.items():
        print(f"Processing task: {task}")
        for goal, plan, obs_list, thought_list, act_list in trajectories:
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
                category=task,
                observations=obs_list,
                reasoning=thought_list,
                actions=act_list,
                rewards=rewards,
                plan=plan,
                reflection=None,
                summary=None
            )

            i += 1

def ingest_alfworld_fewshots_trad():
    db = LearningDB("./data/alfworld_trad/learning.db")
    i = 0
    for goal, obs_list, thought_list, act_list in FEWSHOTS_TRAD:
        obs_list = [Observation(o) for o in obs_list]
        act_list = [Action(a) for a in act_list]
        db.store_episode(
            environment_id=f"trad_{i}",
            goal=goal,
            observations=obs_list,
            reasoning=thought_list,
            actions=act_list,
            rewards=[0] * (len(act_list)-1) + [1],
            plan=None,
            reflection=None,
            summary=None
        )

        i += 1

if __name__ == "__main__":
    ingest_alfworld_fewshots_expel()
    ingest_alfworld_fewshots_trad()
