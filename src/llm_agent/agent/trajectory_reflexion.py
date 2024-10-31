import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import State, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

async def trajectory_reflexion(trajectory: List[Tuple[State, Action]], llm: LiteLLMWrapper, reflection_type: str = "whole") -> Union[str, List[str]]: # Output either text reflecting on the whole trajectory, or a list of text reflecting on each step
    """Reflect on a trajectory using an LLM"""
    # Format trajectory for LLM
    formatted_trajectory = format_trajectory(trajectory)
    
    # Get LLM response: if reflection_type is "whole", return a single string reflecting on the whole trajectory; if "step", return a list of strings reflecting on each step
    if reflection_type == "whole":
        prompt = f"Reflect on the following trajectory:\n{formatted_trajectory}"
    elif reflection_type == "step":
        prompt = f"Reflect on each step of the following trajectory:\n{formatted_trajectory}" # TODO: Add reflection instructions from the reflexion paper: https://arxiv.org/abs/2303.11366

    # Get LLM response
    response = await llm.generate_chat([{"role": "user", "content": prompt}]) 

    return response

def format_trajectory(trajectory: List[Tuple[State, Action]]) -> str:
    """Format a trajectory for LLM reflection"""
    formatted_trajectory = ""
    for i, (state, action) in enumerate(trajectory):
        formatted_trajectory += f"Step {i+1}:\nState: {state.text}\nAction: {action.text}\n\n"
    return formatted_trajectory