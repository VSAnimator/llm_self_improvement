import pytest
import yaml
#from llm_agent.envs import ENVS, INIT_TASKS_FN
from llm_agent.env.alfworld_env import AlfWorldEnv
from types import SimpleNamespace
from llm_agent.env.gym_env import GymEnv
import numpy as np
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
    
    return config #dict_to_namespace(config)

@pytest.fixture
def env(config):
    return AlfWorldEnv(config['benchmark'])

def test_gym_env():
    config = {"env_name": "CartPole-v1"}
    env = GymEnv(config)
    
    # Test initialization
    assert env is not None
    
    # Test reset
    initial_obs = env.reset()
    assert isinstance(initial_obs, (list, tuple, float, int, np.ndarray))
    
    # Test step
    action = 0  # Valid action for CartPole
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, (list, tuple, float, int, np.ndarray))
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # Test action space
    action_space = env.get_action_space()
    assert isinstance(action_space, dict)
    assert action_space["type"] == "integer"
    assert "minimum" in action_space
    assert "maximum" in action_space
    
    # Test available actions
    actions = env.get_available_actions()
    assert isinstance(actions, list)
    assert all(isinstance(a, int) for a in actions)

def test_env_init(env):
    assert env is not None

def test_env_reset(env):
    initial_state, _ = env.reset()
    assert isinstance(initial_state, str)
    assert env.steps == 0
    assert env.current_state == initial_state

def test_env_step(env):
    env.reset()
    action = "look"
    
    next_state, reward, done, info = env.step(action)
    
    assert isinstance(next_state, str)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert env.steps == 1

def test_get_action_space(env):
    action_space = env.get_action_space()
    assert isinstance(action_space, dict)
    assert action_space["type"] == "string"
    assert "description" in action_space

def test_get_available_actions(env):
    _, info = env.reset()
    actions = env.get_available_actions(info)
    assert isinstance(actions, list)
    for action in actions:
        assert isinstance(action, str)

def test_max_steps(env):
    _, info = env.reset()
    
    # Run until max steps
    for _ in range(10):
        action = env.get_available_actions(info)[0]  # Take first available action
        _, _, done, info = env.step(action)
        if done:
            break
