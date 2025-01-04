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

async def reason_old(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> List[Dict]:
    # Take the last conversation message and add a string saying to reason
    conversation[-1]['content'] += "\n Think about the most appropriate action to take from the available actions. The task is not yet complete."
    response = await llm.generate_chat(conversation)
    return response

async def reason(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> List[Dict]:
    # Take the last conversation message and add a string saying to reason
    response = await llm.generate_chat(conversation)
    return response

async def select_action(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> Action:
    """Select an action from available actions given the current observation"""
    response = await llm.generate_chat(conversation)
    return Action(text=response)

async def select_action_old(conversation: List[Dict], observation: Observation, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> Action:
    """Select an action from available actions given the current observation"""
    # Get config values
    max_retries = config.get('max_retries', 3)
    
    # Format prompt with observation and action info
    #prompt = _format_prompt(observation, available_actions)
    
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
    
def _format_prompt(observation: Observation, available_actions: List[Action]) -> str:
    """Format prompt for LLM with observation and action information
    
    Args:
        observation: Current environment observation
        available_actions: List of available actions
        
    Returns:
        Formatted prompt string
    """
    # Basic prompt template - override in subclasses for custom prompting
    prompt = f"Current observation: {repr(observation)}\n\n"
    prompt += "Available actions:\n"
    for i, action in enumerate(available_actions):
        prompt += f"{i+1}. {action.text}\n"
    prompt += "\nSelect the most appropriate action number:"
    return prompt
    
def _parse_action(response: str, available_actions: List[Action]) -> Optional[Action]:
    """Parse LLM response to get selected action
    
    Args:
        response: Raw response from LLM
        available_actions: List of available actions
        
    Returns:
        Selected action or None if parsing failed
    """
    try:
        # Try to parse action number from response
        action_num = int(response.strip())
        if 1 <= action_num <= len(available_actions):
            return available_actions[action_num - 1]
    except:
        pass
    return None