import pytest
from unittest.mock import Mock, patch
from llm_agent.agent.base_agent import BaseAgent
from llm_agent.env.base_env import State, Action
from llm_agent.llm.lite_llm import LiteLLMWrapper, GenerationResponse

@pytest.fixture
def mock_llm():
    llm = Mock(spec=LiteLLMWrapper)
    llm.generate_chat.return_value = "2"  # Mock LLM to always select second action
    return llm

@pytest.fixture
def test_config():
    return {
        "max_retries": 3,
        "memory_size": 5,
        "temperature": 0.7
    }

@pytest.fixture 
def test_agent(mock_llm, test_config):
    return BaseAgent(mock_llm, test_config)

def test_agent_initialization(test_agent, test_config):
    assert test_agent.max_retries == test_config["max_retries"]
    assert test_agent.memory_size == test_config["memory_size"]
    assert test_agent.temperature == test_config["temperature"]
    assert len(test_agent.state_history) == 0
    assert len(test_agent.action_history) == 0

@pytest.mark.asyncio
async def test_act(test_agent):
    state = State("Test state")
    actions = [Action("Action 1"), Action("Action 2")]
    selected_action = await test_agent.act(state, actions)
    assert selected_action in actions

@pytest.mark.asyncio
async def test_memory_size_limit(test_agent):
    states = [State(f"State {i}") for i in range(10)]
    actions = [Action(f"Action {i}") for i in range(10)]
    
    for i in range(10):
        await test_agent.act(states[i], [actions[i]])
    
    assert len(test_agent.state_history) == test_agent.memory_size
    assert len(test_agent.action_history) == test_agent.memory_size

@pytest.mark.asyncio
async def test_agent_reset(test_agent):
    # Add some history
    state = State("Test state")
    action = Action("Test action") 
    await test_agent.act(state, [action])
    
    # Reset agent
    test_agent.reset()
    
    # Verify histories cleared
    assert len(test_agent.state_history) == 0
    assert len(test_agent.action_history) == 0

@pytest.mark.asyncio
async def test_llm_retry_on_failure(test_agent, mock_llm):
    # Make LLM fail twice then succeed
    mock_llm.generate_chat.side_effect = ["invalid", "invalid", "2"]
    
    state = State("Test state")
    actions = [Action("Action 1"), Action("Action 2")]
    
    selected_action = await test_agent.act(state, actions)
    
    # Verify correct action selected after retries
    assert selected_action == actions[1]
    assert mock_llm.generate_chat.call_count == 3

@pytest.mark.asyncio
async def test_llm_max_retries_exceeded(test_agent, mock_llm):
    # Make LLM always fail
    mock_llm.generate_chat.side_effect = Exception("LLM error")
    
    state = State("Test state")
    actions = [Action("Action 1"), Action("Action 2")]
    
    selected_action = await test_agent.act(state, actions)
    
    # Verify first action returned as fallback
    assert selected_action == actions[0]
    assert mock_llm.generate_chat.call_count == test_agent.max_retries
