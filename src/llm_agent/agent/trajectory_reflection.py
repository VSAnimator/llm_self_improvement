import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

async def trajectory_reflection_old(goal: str, trajectory: List[Tuple[Observation, Action]], llm: LiteLLMWrapper, reflection_type: str = "whole") -> Union[str, List[str]]: # Output either text reflecting on the whole trajectory, or a list of text reflecting on each step
    """Reflect on a trajectory using an LLM"""
    # Format trajectory for LLM
    formatted_trajectory = format_trajectory(trajectory)
    
    # Get LLM response: if reflection_type is "whole", return a single string reflecting on the whole trajectory; if "step", return a list of strings reflecting on each step
    if reflection_type == "whole":
        prompt = f"Reflect on the following trajectory:\n{formatted_trajectory}"
    elif reflection_type == "step":
        prompt = f"Reflect on each step of the following trajectory:\n{formatted_trajectory}" # TODO: Add reflection instructions from the reflection paper: https://arxiv.org/abs/2303.11366

    # Get LLM response
    response = await llm.generate_chat([{"role": "system", "content": f"You are an agent in an environment. Given the goal: {goal}, your task is to reflect on the trajectory of observations and actions taken. Identify any mistakes or areas for improvement in the plan or execution."}, {"role": "user", "content": prompt}]) 

    return response

async def conversation_reflection(goal: str, conversation: List[Dict], llm: LiteLLMWrapper, reward: float) -> str:
    """Reflect on a conversation using an LLM"""
    prompt = f"Reflect on the following conversation:\n{conversation}"
    success_str = "successful" if reward == 1 else "unsuccessful"
    response = await llm.generate_chat([{"role": "system", "content": f"You are an agent in an environment. Given the goal: {goal}, your task is to reflect on an attempt to achieve the goal. This attempt was {success_str}. Identify any mistakes or areas for improvement in the plan, as well as the choice of actions taken. Respond with a single paragraph."}, {"role": "user", "content": prompt}]) 
    return response

async def trajectory_summary(goal: str, conversation: List[Dict], llm: LiteLLMWrapper) -> str:
    """Generate a concise summary of a trajectory from conversation history using an LLM"""
    prompt = f"Based on the conversation history, summarize what has happened so far in 1-2 sentences:\n{conversation}"
    
    response = await llm.generate_chat([
        {"role": "system", "content": f"You are an agent in an environment. Given the goal: {goal}, your task is to provide a brief summary of what has happened in the conversation history."},
        {"role": "user", "content": prompt}
    ])
    return response

async def observation_summary(goal: str, observation: Observation, prev_summary: List[Dict], llm: LiteLLMWrapper) -> str:
    """Generate a concise summary of the current observation in context of conversation history using an LLM"""
    prompt = f"Given the prior state, action taken, and current observation, summarize the current state of the environment:\n\nCPrior summary:\n{prev_summary.structured}\n\nCurrent Observation:\n{observation.structured}"
    
    response = await llm.generate_chat([
        {"role": "system", "content": f"You are an agent in an environment that is solving a partially-observable markov decision process. Given the previous state and the action taken, your task is to provide a brief, clear summary of the current state of the environment. Make sure to include all known information and any relevant context."},
        {"role": "user", "content": prompt}
    ])
    return response

def format_trajectory(trajectory: List[Tuple[Observation, Action]]) -> str:
    """Format a trajectory for LLM reflection"""
    formatted_trajectory = ""
    for i, (observation, action) in enumerate(trajectory):
        formatted_trajectory += f"Step {i+1}:\nObservation: {repr(observation)}\nAction: {action.text}\n\n"
    return formatted_trajectory
