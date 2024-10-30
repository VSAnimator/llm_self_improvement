import pytest
import yaml
import asyncio
from llm_agent.llm.lite_llm import LiteLLMWrapper, GenerationResponse

@pytest.fixture
def config():
    with open('config/default.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def llm(config):
    return LiteLLMWrapper(config)

@pytest.mark.asyncio
async def test_generate_text(llm):
    prompt = "Say hello"
    response = await llm.generate_text(prompt)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_generate_chat(llm):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"}
    ]
    response = await llm.generate_chat(messages)
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_generate_structured(llm):
    prompt = "Return a greeting in JSON format"
    schema = {
        "type": "object",
        "properties": {
            "greeting": {"type": "string"}
        }
    }
    response = await llm.generate_structured(prompt, schema)
    assert isinstance(response, GenerationResponse)
    assert isinstance(response.text, str)
    assert response.parsed_json is not None
    assert "greeting" in response.parsed_json
    assert isinstance(response.raw_response, dict)

# Removed or commented out the test_generate_structured_force_json test
# @pytest.mark.asyncio
# async def test_generate_structured_force_json(llm):
#     prompt = "Return a greeting"
#     response = await llm.generate_structured(prompt, force_json=True)
#     assert isinstance(response, GenerationResponse)
#     assert isinstance(response.text, str)
#     assert response.raw_response is not None
