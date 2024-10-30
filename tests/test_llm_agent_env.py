import pytest
import yaml
from llm_agent.agent.base_agent import BaseAgent
from llm_agent.env.base_env import State, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper
from llm_agent.env.alfworld_env import AlfWorldEnv
from types import SimpleNamespace

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

@pytest.mark.asyncio
async def test_alfworld_interaction(test_agent, env):
    # Get initial state from AlfWorld
    obs, info = env.reset()
    state = State(obs)
    
    # Get valid actions from environment
    valid_actions = env.get_available_actions(info)
    actions = [Action(cmd) for cmd in valid_actions]
    
    # Test agent's action selection
    selected_action = await test_agent.select_action(state, actions)
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
        state = State(obs)
        valid_actions = env.get_available_actions(info)
        actions = [Action(cmd) for cmd in valid_actions]
        
        selected_action = await test_agent.select_action(state, actions)
        obs, reward, done, info = env.step(selected_action.text)
        
        steps += 1
        
    assert len(test_agent.state_history) == steps
    assert len(test_agent.action_history) == steps
