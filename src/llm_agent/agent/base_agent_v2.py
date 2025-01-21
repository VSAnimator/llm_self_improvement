import os
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from logging import getLogger
from ..env.base_env import Observation, Action
from ..agent.generate_plan import generate_plan
from ..agent.trajectory_reflection import trajectory_summary, observation_summary

logger = getLogger(__name__)

class BaseAgent:
    """Base agent class that uses an LLM to select actions"""
    
    def __init__(self, llm, db, env, config):
        """Initialize the agent
        
        Args:
            llm: LLM instance to use for decision making
            config: Configuration dictionary containing agent parameters
        """
        self.config = config

        # LLM
        self.llm = llm
        self.max_retries = config.get('max_retries', 3)
        self.memory_size = config.get('memory_size', 20)
        self.temperature = config.get('temperature', 0.7)
        
        # Trajectory history
        self.observation_history: List[Observation] = []
        self.reasoning_history: List[str] = []
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.plan: Optional[str] = None
        self.in_context_data: Optional[Dict] = None
        
        # Environment info
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal
        self.category: Optional[str] = env.category if hasattr(env, 'category') else None
        self.action_space: Optional[str] = env.get_action_space() if hasattr(env, 'get_action_space') else None
        # Database
        self.db = db

    """ Helper functions """

    def _create_conversation(self, keys: List[str], available_actions: List[Action]) -> List[Dict]:
        """Create a conversation with the observation and action history"""
        # First create a dictionary of the current trajectory. Valid keys are "plan", "observation", "reasoning", "action"
        trajectory_dict = {}
        # Sort keys according to the following order: goal, plan, observation, reasoning, action, available_actions
        key_order = ["goal", "plan", "observation", "reasoning", "action", "available_actions"]
        keys = [key for key in key_order if key in keys]
        for key in keys:
            if key == "goal":
                trajectory_dict[key] = self.goal
            elif key == "plan":
                trajectory_dict[key] = self.plan
            elif key == "observation":
                trajectory_dict[key] = [self.observation_history[i].structured for i in range(len(self.observation_history))]
            elif key == "reasoning":
                trajectory_dict[key] = [self.reasoning_history[i] for i in range(len(self.reasoning_history))]
            elif key == "action":
                trajectory_dict[key] = [self.action_history[i].text for i in range(len(self.action_history))]
            elif key == "available_actions":
                trajectory_dict[key] = repr([f"{i+1}. {a.text}" for i, a in enumerate(available_actions)])
        # Create a string of the current trajectory by rolling up the trajectory_dict
        trajectory_dict = self._roll_up_trajectory([trajectory_dict])[0]
        # Create a string of the current trajectory
        trajectory_string = "Current trajectory:\n"
        for key in trajectory_dict:
            trajectory_string += f"{key}: {trajectory_dict[key]}\n"
        return trajectory_string
    
    def _roll_up_trajectory(self, entries):
        interleaved_keys = ["observation", "reasoning", "action"]
        """Roll up the trajectory into a single string"""
        value_type = list(entries[0].keys())
        if any(k in value_type for k in interleaved_keys):
            for i in range(len(entries)):
                trajectory_string = "\n"
                for j in range(len(entries[i][interleaved_keys[0]])):
                    for k in interleaved_keys:
                        if len(entries[i][k]) > j and len(entries[i][k][j]) > 0:
                            trajectory_string += f"{k}: {entries[i][k][j]}\n"
                entries[i]["trajectory"] = trajectory_string
                # Delete the interleaved keys from the dictionary
                for k in interleaved_keys:
                    del entries[i][k]
        return entries
    
    # Level of abstraction of the in-context data should come in here
    def _in_context_prompt(self, in_context_data):
        """Create a system prompt containing in-context examples from similar episodes"""
        # Determine which set of entries to use
        success_entries, failure_entries = in_context_data[0], in_context_data[1]
        entries = success_entries if len(success_entries) > 0 else failure_entries
        outcome = "successfully achieved" if len(success_entries) > 0 else "failed to achieve"

        if len(entries) == 0:
            return ""
        
        entries = self._roll_up_trajectory(entries)
        value_type = list(entries[0].keys()) # Keys from entries

        system_prompt = f"\nHere are some examples of {', '.join(value_type)} from episodes that {outcome} similar goals:\n"

        # Add each example
        for i, entry in enumerate(entries):
            system_prompt += f"\nExample {i+1}:\n"
            # Add keys
            for k_type in value_type:
                system_prompt += f"{k_type}: {entry[k_type]}\n"

        system_prompt += f"\nFollow the provided examples closely."

        return system_prompt
    
    def _get_in_context_data(self, key_type, key, value_type, outcome="winning", k=5, window=20) -> List[Dict]:
        """Retrieve in context examples from the database"""
        success_entries, failure_entries = self.db.get_similar_entries(key_type, key, outcome=outcome, k=k, window=window)
        # Now figure out which part of the examples to return in-context
        if not isinstance(value_type, list): # Check that this is a list, not a string
            value_type = [value_type]
        # Filter out the keys that are not in the value_type
        success_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in success_entries]
        failure_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in failure_entries]
        return success_entries, failure_entries
    
    """ Main components used by agent's choose_action function """

    # A wrapper function for the get_in_context_data function when getting state-level data with a window
    def get_state_data(self, trajectory_key_types, trajectory_keys, state_key_types, state_keys, value_types, outcome, k, window) -> List[Dict]:
        # Combine the trajectory and state keys for now since the wrapped function will split them back out
        key = trajectory_keys + state_keys
        key_type = trajectory_key_types + state_key_types
        """Retrieve state-level in-context examples from the database"""
        return self._get_in_context_data(key_type, key, value_types, outcome, k, window)
    
    # A wrapper function for the get_in_context_data function when getting trajectory-level data
    def get_trajectory_data(self, key_types, keys, value_types, outcome, k) -> List[Dict]:
        """Retrieve trajectory-level in-context examples from the database"""
        return self._get_in_context_data(key_types, keys, value_types, outcome, k)
    
    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """Generate a plan for the agent to follow"""
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal. "
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        plan = await generate_plan(conversation, self.goal, self.llm)
        self.plan = plan
        return plan

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """Reason about the conversation and observation"""
        self.observation_history.append(observation)
        conversation = []
        # Add system prompt
        system_prompt = f"You are an expert at reasoning about the most appropriate action to take towards achieving a goal. "
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        # Add conversation history
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "reasoning: "})
        reasoning = await self.llm.generate_chat(conversation)
        self.reasoning_history.append(reasoning)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """Select an action from available actions given the current observation"""
        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        system_prompt = f"""You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}."""
        # If this is a TRAD agent, we want to add the action space to the system prompt
        if "trad" in self.config.get("agent_type", "").lower():
            system_prompt += "\nHere is your action space:\n" + self.action_space['description']
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "action: "})
        
        # Select action
        action = await self.llm.generate_chat(conversation)
        action = Action(text=action)
        self.action_history.append(action)

        # Enforce memory size limit
        self.observation_history = self.observation_history[-self.memory_size:]
        self.action_history = self.action_history[-self.memory_size:]

        return action
    
    """ Main components used by agent's process_feedback function """
    
    async def reflect(self, observation: Observation, reward: float) -> List[Dict]:
        """Reflect on the conversation and observation"""
        user_prompt = self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], [])
        response = await self.llm.generate_chat([{"role": "system", "content": f"You are an agent in an environment. Given the goal: {self.goal}, your task is to reflect on the trajectory of observations and actions taken. Identify any mistakes or areas for improvement in the plan or execution."}, {"role": "user", "content": user_prompt}]) 
        return response
    
    async def summarize(self, obs=None) -> str:
        """Summarize the conversation and observation"""
        conversation = []
        conversation = self._create_conversation(conversation, None, None, None)
        if obs is None:
            summary = await trajectory_summary(self.goal, conversation, self.llm)
        else:
            prev_obs = self.observation_history[-1] if len(self.observation_history) > 0 else Observation(None)
            summary = await observation_summary(self.goal, obs, prev_obs, self.llm)
        # Create an observation object again
        summary = Observation(summary)
        return summary
    
    def store_episode(self, reflection, summary):
        """Store an episode in the database"""
        self.db.store_episode(self.environment_id, self.goal, self.category,self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, reflection, summary)

    """ For updating rules offline """

    def get_contrastive_pairs(self):
        """Get contrastive pairs of episodes"""
        # Fetch contrastive pairs from database and return
        contrastive_pairs = self.db.get_contrastive_pairs()
        return contrastive_pairs

    """ Placeholder functions for agent's choose_action and process_feedback functions """
    
    async def choose_action(self, obs, valid_actions, log_file):
        """Choose an action from available actions given the current observation"""
        pass
    
    async def process_feedback(self, new_obs, reward, done, log_file):
        """Process feedback from the environment"""
        pass

    """ Rule generation functions """

    async def update_rules_offline(self):
        """Update rules offline"""
        pass
    
    async def update_rules_online(self, env_id):
        """Update rules online"""
        pass

    """ Between-episode functions for the outer loop """
        
    def clear_history(self):
        """Clear the agent's history"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.in_context_data = None

    def reset(self):
        """Reset agent state between episodes"""
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.reflections = None
        self.in_context_data = None