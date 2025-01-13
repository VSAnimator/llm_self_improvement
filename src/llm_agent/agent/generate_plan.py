import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

async def generate_plan(conversation: List[Dict], goal: str, llm: LiteLLMWrapper) -> str:
    """Generate a plan for the agent to follow"""
    curr_prompt = f"goal: {goal}\n plan: "

    conversation.append({"role": "user", "content": curr_prompt})

    try:
        response = await llm.generate_chat(conversation)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating plan: {str(e)}")
        return []