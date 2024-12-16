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
from llm_agent.logging.setup_db import LoggingDatabases

def dict_to_namespace(d):
    """Convert dictionary to namespace recursively"""
    if not isinstance(d, dict):
        return d
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)

@pytest.fixture
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

@pytest.fixture
def env(config):
    return AlfWorldEnv(config['benchmark'])

@pytest.fixture
def real_llm(config):
    return LiteLLMWrapper(config)

@pytest.fixture
def test_config():
    return {
        "max_retries": 3,
        "memory_size": 5,
        "temperature": 0.7
    }

@pytest.fixture
def test_agent(real_llm, test_config):
    return BaseAgent(real_llm, test_config)

@pytest.fixture(autouse=True)
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

@pytest.fixture
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

@pytest.mark.asyncio
async def test_alfworld_interaction(test_agent, env):
    # Get initial state from AlfWorld
    obs, info = env.reset()
    observation = Observation(obs)
    
    # Get valid actions from environment
    valid_actions = env.get_available_actions(info)
    actions = [Action(cmd) for cmd in valid_actions]
    
    # Test agent's action selection
    selected_action = await test_agent.act(env.goal, observation, actions)
    assert selected_action in actions
    
    # Test environment step
    print(selected_action)
    obs, reward, done, info = env.step(selected_action.text)
    assert isinstance(obs, str)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)

@pytest.mark.asyncio
async def test_alfworld_multi_step(test_agent, env):
    obs, info = env.reset()
    done = False
    steps = 0
    max_steps = 5
    
    while not done and steps < max_steps:
        observation = Observation(obs)
        valid_actions = env.get_available_actions(info)
        actions = [Action(cmd) for cmd in valid_actions]
        
        selected_action = await test_agent.act(env.goal, observation, actions)
        obs, reward, done, info = env.step(selected_action.text)
        
        steps += 1
        
    assert len(test_agent.observation_history) == steps
    assert len(test_agent.action_history) == steps

@pytest.mark.asyncio
async def test_agent_reflection(test_agent, env):
    # Run a few steps to build up history
    obs, info = env.reset()
    observation = Observation(obs)
    valid_actions = env.get_available_actions(info)
    actions = [Action(cmd) for cmd in valid_actions]
    
    # Build conversation history
    conversation = []
    for i in range(3):
        selected_action = await test_agent.act(env.goal, observation, actions)
        conversation.append({"role": "user", "content": str(observation)})
        conversation.append({"role": "assistant", "content": str(selected_action)})
        
        # Step environment
        obs, _, _, info = env.step(selected_action.text)
        observation = Observation(obs)
        valid_actions = env.get_available_actions(info)
        actions = [Action(cmd) for cmd in valid_actions]

    # Test reflection
    reflection = await test_agent.reflect(env.goal, conversation, observation)
    
    # Verify reflection output
    assert isinstance(reflection, str)
    assert len(reflection) > 0
    assert test_agent.reflections is not None
    assert len(test_agent.reflections) == 1
