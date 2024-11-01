import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger
from pydantic import BaseModel
import json

from ..env.base_env import State, Action
from ..llm.lite_llm import LiteLLMWrapper

logger = getLogger(__name__)

async def reason(conversation: List[Dict], state: State, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> List[Dict]:
    # Reason through the action options available
    prompt = "Given the following conversation and state, reason about the most appropriate action to take from the available actions."
    prompt += "\nState: " + repr(state)
    prompt += "\nAvailable actions:\n"
    for i, action in enumerate(available_actions):
        prompt += f"{i+1}. {action.text}\n"
    conversation.append({"role": "user", "content": prompt})
    response = await llm.generate_chat(conversation)
    return response

async def select_action(conversation: List[Dict], state: State, available_actions: List[Action], llm: LiteLLMWrapper, config: Dict) -> Action:
    """Select an action from available actions given the current state"""
    # Get config values
    max_retries = config.get('max_retries', 3)
    
    # Format prompt with state and action info
    prompt = _format_prompt(state, available_actions)
    
    # Get LLM response
    for _ in range(max_retries):
        try:
            conversation.append({"role": "user", "content": prompt})
            #print(f"Conversation: {conversation}")
            # Use structured output
            class ActionNumber(BaseModel):
                action_number: int
            response = await llm.generate_structured(conversation, output_schema=ActionNumber)
            # Get action number from response
            action_number = json.loads(response.choices[0].message.content)['action_number']
            action = available_actions[action_number - 1]
            if action is not None:
                return action
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            
    # If all retries failed, return first available action
    logger.warning("Failed to select action, defaulting to first available")
    # Throw error
    raise ValueError(f"Failed to select action. Conversation: {conversation}, Response: {response}")
    action = available_actions[0]

    return action
    
def _format_prompt(state: State, available_actions: List[Action]) -> str:
    """Format prompt for LLM with state and action information
    
    Args:
        state: Current environment state
        available_actions: List of available actions
        
    Returns:
        Formatted prompt string
    """
    # Basic prompt template - override in subclasses for custom prompting
    prompt = f"Current state: {repr(state)}\n\n"
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