import pytest
import yaml
from llm_agent.agent.base_agent import BaseAgent
from llm_agent.env.base_env import State, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper

@pytest.fixture
def config():
    with open('config/default.yaml', 'r') as f:
        return yaml.safe_load(f)

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
async def test_real_llm_action_selection(test_agent):
    state = State("You are in a kitchen. There is a fridge and a counter.")
    actions = [
        Action("Open the fridge"),
        Action("Look at the counter"),
        Action("Leave the kitchen")
    ]
    selected_action = await test_agent.act(state, actions)
    assert selected_action in actions

@pytest.mark.asyncio
async def test_real_llm_complex_state(test_agent):
    state = State("You are trying to make breakfast. The fridge contains eggs and milk. The counter has bread and a toaster.")
    actions = [
        Action("Make scrambled eggs"),
        Action("Make toast"),
        Action("Pour a glass of milk")
    ]
    selected_action = await test_agent.act(state, actions)
    assert selected_action in actions
    
@pytest.mark.asyncio
async def test_real_llm_memory_usage(test_agent):
    states = [
        State("You enter the kitchen"),
        State("You open the fridge"),
        State("You take out eggs")
    ]
    actions = [
        Action("Look around"),
        Action("Get ingredients"),
        Action("Start cooking")
    ]
    
    for i in range(len(states)):
        selected_action = await test_agent.act(states[i], [actions[i]])
        assert selected_action == actions[i]
    
    assert len(test_agent.state_history) == 3
    assert len(test_agent.action_history) == 3

@pytest.mark.asyncio
async def test_real_llm_with_reset(test_agent):
    state = State("You are in the kitchen")
    action = Action("Look around")
    
    await test_agent.act(state, [action])
    assert len(test_agent.state_history) == 1
    
    test_agent.reset()
    assert len(test_agent.state_history) == 0
    assert len(test_agent.action_history) == 0
