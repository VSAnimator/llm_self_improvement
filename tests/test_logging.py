import pytest
import yaml
import numpy as np
import os
import shutil
from llm_agent.logging.setup_db import LoggingDatabases
from llm_agent.llm.lite_llm import LiteLLMWrapper
from pathlib import Path

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

@pytest.fixture
def config():
    with open('config/default.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def llm(config):
    return LiteLLMWrapper(config)

@pytest.mark.asyncio
async def test_log_and_retrieve_llm_interaction(db, llm):
    # Print database path
    print(f"Database path: {db.db_path}")
    
    # Verify clean database state
    math_states = db.search_by_task("math", table="states")
    assert len(math_states) == 0, "Database should be empty at start of test"
    
    # 1. Generate an LLM response
    prompt = "What is 2+2?"
    response = await llm.generate_text(prompt)
    
    # 2. Create mock vectors (normally these would be embeddings)
    state_vector = np.array([1.0, 0.0, 0.0, 0.0])
    trajectory_vector = np.array([1.0, 1.0, 0.0, 0.0])
    
    # 3. Log the interaction
    state_data = {
        "prompt": prompt,
        "response": response,
        "timestamp": "2024-01-01"
    }
    state_id = db.add_state(
        state_vector=state_vector,
        state_data=state_data,
        task="math"
    )
    
    # 4. Add a mock action
    action_data = {
        "action_type": "generate_text",
        "model": "gpt-3.5-turbo"
    }
    action_id = db.add_action(state_id, action_data)
    
    # 5. Add trajectory
    trajectory_data = {
        "description": "Simple math question",
        "success": True
    }
    trajectory_id = db.add_trajectory(
        trajectory_vector=trajectory_vector,
        trajectory_data=trajectory_data,
        state_ids=[state_id],
        task="math"
    )
    
    # 6. Test retrieval methods
    similar_states = db.search_similar_states(state_vector)
    assert len(similar_states['distances']) > 0
    
    # Verify we're getting the exact match
    assert similar_states['distances'][0] == 0.0, "First result should be an exact match"
    
    # Now we can directly use the state ID
    retrieved_state = db.get_state(similar_states['state_ids'][0])
    assert retrieved_state['prompt'] == prompt
    assert retrieved_state['response'] == response.text
    
    # Test action retrieval
    action = db.get_action_for_state(state_id)
    assert action['action_type'] == "generate_text"
    assert action['model'] == "gpt-3.5-turbo"
    
    # Test task search
    math_states = db.search_by_task("math", table="states")
    assert len(math_states) > 0
    assert math_states[0]['prompt'] == prompt
    assert math_states[0]['response'] == response.text
    
    # Test similar trajectory search
    similar_trajectories = db.search_similar_trajectories(trajectory_vector)
    assert len(similar_trajectories['distances']) > 0
    assert similar_trajectories['vector_ids'][0] == 0  # Should be first entry

@pytest.mark.asyncio
async def test_error_conditions(db):
    # Test invalid state ID
    with pytest.raises(KeyError):
        db.get_state(-1)  # Non-existent state ID
    
    # Test empty task search
    empty_results = db.search_by_task("nonexistent_task", table="states")
    assert len(empty_results) == 0

@pytest.mark.asyncio
async def test_multiple_states_trajectory(db, llm):
    # Test logging multiple states in a single trajectory
    prompts = ["What is 2+2?", "What is 4+4?"]
    state_ids = []
    
    for prompt in prompts:
        response = await llm.generate_text(prompt)
        state_vector = np.random.rand(4)  # Random test vector
        state_data = {
            "prompt": prompt,
            "response": response,
            "timestamp": "2024-01-01"
        }
        state_id = db.add_state(
            state_vector=state_vector,
            state_data=state_data,
            task="math"
        )
        state_ids.append(state_id)
        
        action_data = {
            "action_type": "generate_text",
            "model": "gpt-3.5-turbo"
        }
        db.add_action(state_id, action_data)
    
    # Add trajectory connecting multiple states
    trajectory_vector = np.random.rand(4)
    trajectory_data = {
        "description": "Multi-step math questions",
        "success": True
    }
    trajectory_id = db.add_trajectory(
        trajectory_vector=trajectory_vector,
        trajectory_data=trajectory_data,
        state_ids=state_ids,
        task="math"
    )
    
    # Verify trajectory contains all states
    trajectory_states = db.get_states_for_trajectory(trajectory_id)
    assert len(trajectory_states) == len(prompts)
    assert all(state["prompt"] in prompts for state in trajectory_states)

@pytest.mark.asyncio
async def test_structured_output_logging(db, llm):
    # Test logging structured output
    prompt = "Return a greeting in JSON format"
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"}
        }
    }
    response = await llm.generate_structured(prompt, schema)
    
    state_vector = np.random.rand(4)
    state_data = {
        "prompt": prompt,
        "response": response.text,
        "parsed_json": response.parsed_json,
        "timestamp": "2024-01-01"
    }
    
    state_id = db.add_state(
        state_vector=state_vector,
        state_data=state_data,
        task="structured_output"
    )
    
    # Verify structured output was logged correctly
    retrieved_state = db.get_state(state_id)
    assert "parsed_json" in retrieved_state
    assert "greeting" in retrieved_state["parsed_json"]
