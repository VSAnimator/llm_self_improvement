import pytest

import yaml
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from llm_agent.agent.base_agent import BaseAgent
from llm_agent.env.base_env import State, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.env.alfworld_env import AlfWorldEnv
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

def env(config):
    return AlfWorldEnv(config['benchmark'])

def real_llm(config):
    return LiteLLMWrapper(config)

def test_config():
    return {
        "max_retries": 3,
        "memory_size": 5,
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

async def main():
    # Initialize config and environment
    cfg = config()
    environment = env(cfg)
    llm = real_llm(cfg)
    agent_config = test_config()
    agent = test_agent(llm, agent_config)

    # Initial reset
    obs, info = environment.reset()
    done = False
    steps = 0
    max_steps = 10  # Increased from test value
    
    print(f"Starting new episode with goal: {environment.goal}")
    
    while not done and steps < max_steps:
        # Convert observation to state
        state = State(obs)
        print(f"\nStep {steps + 1}")
        print(f"Current state: {obs}")
        
        # Get valid actions
        valid_actions = environment.get_available_actions(info)
        actions = [Action(cmd) for cmd in valid_actions]
        print(f"Valid actions: {[a.text for a in actions]}")

        # First, reason about the available actions
        reasoning = await agent.reason(environment.goal, state, actions)
        print(f"Reasoning: {reasoning}")
        
        # Get agent's action
        selected_action = await agent.act(environment.goal, state, actions)
        print(f"Agent selected: {selected_action.text}")
        
        # Take step in environment
        obs, reward, done, info = environment.step(selected_action.text)
        print(f"Reward: {reward}")
        
        steps += 1
        
        if done:
            print("\nEpisode finished!")
            print(f"Steps taken: {steps}")
            print(f"Final reward: {reward}")
    
    if not done:
        print("\nEpisode timed out after reaching max steps")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
