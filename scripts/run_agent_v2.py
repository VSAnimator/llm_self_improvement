import asyncio

import yaml
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from llm_agent.agent.base_agent_v2 import BaseAgent
from llm_agent.agent.react import React
from llm_agent.agent.reflexion import Reflexion
from llm_agent.agent.reflexion_collect import ReflexionCollect
from llm_agent.agent.rap import RAP
from llm_agent.agent.synapse import Synapse
from llm_agent.agent.autoguide import AutoGuide
from llm_agent.env.base_env import Observation, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.env.alfworld_env import AlfWorldEnv
from llm_agent.env.gym_env import GymEnv
from llm_agent.database.learning_db import LearningDB
import random

def dict_to_namespace(d):
    """Convert dictionary to namespace recursively"""
    if not isinstance(d, dict):
        return d
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

def config():
    # Load default config first
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with alfworld-specific config
    with open('config/benchmark/alfworld.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
        
    # Ensure required alfworld configuration exists
    env_config.update({
        'type': 'AlfredTWEnv',  # Required env type for alfworld
        'split': 'eval_out_of_distribution'  # Required split parameter # eval_out_of_distribution
    })
    
    # Update with benchmark config
    config['benchmark'] = env_config
    
    return config

use_gym = False

def env(config):
    if use_gym:
        env_config = {"env_name": "CartPole-v1"}
        return GymEnv(env_config)
    else:
        return AlfWorldEnv(config['benchmark'])

def real_llm(config):
    return LiteLLMWrapper(config)

def test_config():
    return {
        "max_retries": 3,
        "memory_size": 50,
        "temperature": 0.7,
        "agent_type": "reflexion_collect"
    }

def test_agent(real_llm, db, env, test_config):
    if test_config.get('agent_type', 'react') == 'react':
        return React(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'reflexion':
        return Reflexion(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'reflexion_collect':
        return ReflexionCollect(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'rap':
        return RAP(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'synapse':
        return Synapse(real_llm, db, env, test_config)
    elif test_config.get('agent_type', 'react') == 'autoguide':
        return AutoGuide(real_llm, db, env, test_config)
    else:
        raise ValueError(f"Invalid agent type: {test_config.get('agent_type', 'react')}")

def db():
    return LearningDB("data/learning.db")

# Placeholder for configs
learning_config = {}
#num_attempts = learning_config.get('num_attempts', 3)
use_plan = learning_config.get('use_plan', True)
use_reflexion = learning_config.get('use_reflexion', True)

# llm should be a property of the agent

async def run_env(agent, env, log_file):
    # Goal: run the agent on the environment and log the results
    attempt_count = 0
    num_attempts = env.num_attempts
    print("Num attempts", num_attempts)
    with open(log_file, "w") as f:
        for attempt in range(num_attempts):
            # Initial reset
            obs, info = env.reset()
            f.write(f"Initial observation: {obs}\n")
            done = False
            steps = 0

            while not done and steps < env.max_steps:
                # Get valid actions
                valid_actions = env.get_available_actions(info)
                f.write(f"Valid actions: {valid_actions}\n")
                # Choose action
                selected_action = await agent.choose_action(obs, valid_actions, log_file)
                f.write(f"Selected action: {selected_action}\n")
                # Take step in env
                obs, reward, done, info = env.step(selected_action.text) # This feedback needs to be looped anyways
                f.write(f"Obs: {obs}, Reward: {reward}\n")
                # Pass feedback to agent
                await agent.process_feedback(obs, reward, done, log_file)
                # Increment step count
                steps += 1
                f.write(f"Step {steps} of {env.max_steps}\n")
                # Flush the file to ensure the log is written
                f.flush()
            
            if not done:
                f.write("\nEpisode timed out after reaching max steps\n")
                f.flush()

            if reward < 1:
                attempt_count += 1 
                agent.clear_history()
            else:
                break

# Run the environment
async def main():
    for i in range(9, 134):
        cfg = config()
        cfg['benchmark']['problem_id'] = i
        environment = env(cfg)
        llm = real_llm(cfg)
        agent_config = test_config()
        learning_db = db()
        agent = test_agent(llm, learning_db, environment, agent_config)
        # Create log file folder matching config
        # Make sure to name sub-folder to include environment and agent type
        log_dir = Path("logs/episodes") / f"{cfg['benchmark']['name']}/{cfg['benchmark']['split']}/{agent_config.get('agent_type', 'react')}/{cfg['llm']['model']}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{i}.txt"
        await run_env(agent, environment, log_file)

if __name__ == "__main__":
    asyncio.run(main())