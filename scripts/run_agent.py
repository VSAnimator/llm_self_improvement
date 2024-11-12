import pytest

import yaml
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from llm_agent.agent.base_agent import BaseAgent
from llm_agent.env.base_env import Observation, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.env.alfworld_env import AlfWorldEnv
from llm_agent.env.gym_env import GymEnv
from llm_agent.logging.setup_db import LoggingDatabases

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
        'split': 'eval_out_of_distribution'  # Required split parameter
    })
    
    # Update with benchmark config
    config['benchmark'] = env_config
    
    return config

use_gym = True

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
        "temperature": 0.7
    }

def test_agent(real_llm, test_config):
    return BaseAgent(real_llm, test_config)

def clean_db():
    print("Did this run")
    # Clean up before test
    db_path = Path("logs/test_env.db")
    if db_path.exists():
        os.remove(db_path)
    print(f"Database path: {db_path}")
    
    # Create logs directory if it doesn't exist
    db_path.parent.mkdir(exist_ok=True)
    
    yield
    
    # Clean up after test
    if db_path.exists():
        os.remove(db_path)

def db():
    db = LoggingDatabases(
        env_name="test_env",
        state_dim=4,  # Match test vector dimensions
        trajectory_dim=4
    )
    yield db
    db.close()
    # Cleanup after tests
    if os.path.exists("test_env"):
        shutil.rmtree("test_env")

# Placeholder for configs
use_plan = True
use_reflexion = True
reflexion_steps = 3

async def main():
    # Create logs directory if it doesn't exist
    log_dir = Path("logs/episodes")
    log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        # Create log file for this episode
        log_file = log_dir / f"episode_{i}.txt"
        reflexion_count = 0
        with open(log_file, "w") as f:
            # Initialize config and environment
            cfg = config()
            cfg['benchmark']['problem_id'] = i
            environment = env(cfg)
            llm = real_llm(cfg)
            agent_config = test_config()
            agent = test_agent(llm, agent_config)

            while True:
                # Initial reset
                obs, info = environment.reset()
                done = False
                steps = 0
                max_steps = 50  # Increased from test value
                
                f.write(f"Episode {i}\n")
                f.write(f"Goal: {environment.goal}\n\n")
                
                while not done and steps < max_steps:
                    # Convert observation to Observation object
                    observation = Observation(obs)
                    f.write(f"\nStep {steps + 1}\n")
                    f.write(f"Observation: {obs}\n")
                    
                    # Get valid actions
                    valid_actions = environment.get_available_actions(info)
                    actions = [Action(cmd) for cmd in valid_actions]
                    f.write(f"Valid actions: {[a.text for a in actions]}\n")

                    # Restructure the code to reflect the pseudocode
                    # Generate plan if it doesn't exist
                    if not agent.plan and use_plan:
                        plan = await agent.create_plan(environment.goal, observation, actions)
                        f.write(f"Plan: {plan}\n")

                    # Reason about the available actions
                    reasoning = await agent.reason(environment.goal, observation, actions)
                    f.write(f"Reasoning: {reasoning}\n")
                    
                    # Get agent's action
                    selected_action = await agent.act(environment.goal, observation, actions, reasoning)
                    f.write(f"Selected action: {selected_action.text}\n")
                    
                    # Take step in environment
                    obs, reward, done, info = environment.step(selected_action.text)
                    f.write(f"Reward: {reward}\n")
                    
                    steps += 1
                    
                    if done:
                        f.write("\nEpisode finished!\n")
                        f.write(f"Steps taken: {steps}\n")
                        f.write(f"Final reward: {reward}\n")

                    # Flush the file buffer
                    f.flush()
                
                if not done:
                    f.write("\nEpisode timed out after reaching max steps\n")

                # If reflexion is enabled, perform reflexion
                if reward < 200 and use_reflexion and reflexion_count < reflexion_steps:
                    reflexion = await agent.reflect(environment.goal, observation)
                    f.write(f"Reflexion: {reflexion}\n")
                    f.flush()
                    reflexion_count += 1
                    # Clear history after reflexion
                    agent.clear_history()
                else:
                    break

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
