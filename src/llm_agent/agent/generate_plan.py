import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import State, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

# We can call this a plan/strategy/thought process/etc.
# What is the best way to generate a plan? Get options from the literature
# So we can generate the plan every time from scratch, or we can choose from a list of pre-generated plans/strategies

async def generate_plan(state: State, llm: LiteLLMWrapper, output_type: str = "list") -> Union[str, List[str]]:
    """Generate a plan for the agent to follow"""
    # Format state for LLM
    prompt = f"Given the current state:\n{state.text}\n\nGenerate a high-level plan of actions to achieve the goal. Format the plan as a numbered list." # TODO: make this match the literature on plan generation

    # Get LLM response
    try:
        # If output_type is "list", return a list of actions
        if output_type == "list":
            response = await llm.generate_chat([{"role": "user", "content": prompt}])
            return response.strip().split('\n')
        # If output_type is "string", return a string
        elif output_type == "string":
            response = await llm.generate_chat([{"role": "user", "content": prompt}])
            return response.strip()
        else:
            raise ValueError(f"Invalid output_type: {output_type}")
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        return []