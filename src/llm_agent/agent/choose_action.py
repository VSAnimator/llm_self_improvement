import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger
from pydantic import BaseModel
import json
import enum

from ..env.base_env import Observation, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

async def reason(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> List[Dict]:
    response = await llm.generate_chat(conversation)
    return response

async def select_action(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> Action:
    """Select an action from available actions given the current observation"""
    response = await llm.generate_chat(conversation)
    return Action(text=response)

async def select_action_structured(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> Action:
    """Select an action from available actions given the current observation"""
    # Get config values
    max_retries = config.get('max_retries', 3)
    # Get LLM response
    for _ in range(max_retries):
        try:
            # Use structured output
            class ActionNumber(BaseModel):
                action_number: int
            response = await llm.generate_structured(conversation, output_schema=ActionNumber)
            action_number = json.loads(response.choices[0].message.content)['action_number']
            action = available_actions[action_number - 1]
            if action is not None:
                return action
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            
    # If all retries failed, return first available action
    logger.warning("Failed to select action, defaulting to first available")
    # Throw error
    #raise ValueError(f"Failed to select action. Conversation: {conversation}, Response: {response}")
    action = available_actions[0]

    return action