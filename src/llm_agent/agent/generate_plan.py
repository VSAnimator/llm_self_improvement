import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

# We can call this a plan/strategy/thought process/etc.
# What is the best way to generate a plan? Get options from the literature
# So we can generate the plan every time from scratch, or we can choose from a list of pre-generated plans/strategies

async def generate_plan(conversation: List[Dict], goal: str, observation: Observation, llm: LiteLLMWrapper, curr_plan: Optional[str] = None, curr_reflexion: Optional[str] = None, output_type: str = "string") -> Union[str, List[str]]:
    """Generate a plan for the agent to follow"""
    curr_prompt = f"Now, the goal you would like to achieve is:\n{goal}\n\nGiven the current observation:\n{observation.structured}\n\nGenerate a high-level plan of actions to achieve the goal. This plan should be short and concise." # TODO: make this match the literature on plan generation

    # If we have a current plan and reflexion, we can use that to guide the plan generation
    if curr_plan is not None and curr_reflexion is not None:
        curr_prompt = f"The goal you would like to achieve is:\n{goal}. You have previously generated the following plan:\n{curr_plan}\n\nYou have also previously reflected on your plan as follows:\n{curr_reflexion}\n\nGiven this, as well as the current observation:\n{observation.structured}\n\n Generate a new plan of actions to achieve the goal."

    conversation.append({"role": "user", "content": curr_prompt})

    # Get LLM response
    try:
        # If output_type is "list", return a list of actions
        if output_type == "list":
            response = await llm.generate_chat(conversation)
            return response.strip().split('\n')
        # If output_type is "string", return a string
        elif output_type == "string":
            response = await llm.generate_chat(conversation)
            return response.strip()
        else:
            raise ValueError(f"Invalid output_type: {output_type}")
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        return []