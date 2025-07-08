from typing import Dict, List, Optional, Tuple, Union
from logging import getLogger
from ..env.base_env import Observation, Action

logger = getLogger(__name__)

class BaseAgent:
    """
    Base agent class that uses an LLM to select actions in sequential environments.
    
    This class provides core primitives for building different agent algorithms:
    1. Decision-making components (create_plan, reason, act)
    2. In-context learning utilities (retrieve_trajectory_data, retrieve_state_data)
    3. Episode analysis tools (reflect, store_episode, clean_history)
    
    Subclasses should implement the choose_action method to define their specific
    decision-making process using these primitives.
    """
    
    def __init__(self, llm, db, env, config):
        """
        Initialize the agent with LLM, database, environment, and configuration.
        
        Args:
            llm: LLM instance to use for decision making
            db: Database for storing and retrieving episodes
            env: Environment the agent will interact with
            config: Configuration dictionary containing agent parameters
        """
        self.config = config

        # LLM configuration
        self.llm = llm
        self.max_retries = config.get('max_retries', 3)
        self.memory_size = config.get('memory_size', 50)
        self.temperature = config.get('temperature', 0.7)
        self.num_ic = config.get('num_ic', None)
        
        # Trajectory history storage
        self.observation_history: List[Observation] = []
        self.reasoning_history: List[str] = []
        self.action_history: List[Action] = []
        self.reward_history: List[float] = []
        self.plan: Optional[str] = None
        self.in_context_data: Optional[Dict] = None
        self.reflection: Optional[str] = None
        self.summary: Optional[str] = None
        
        # Environment information
        self.environment_id: Optional[str] = env.id
        self.goal: Optional[str] = env.goal if hasattr(env, 'goal') else None
        self.category: Optional[str] = env.category if hasattr(env, 'category') else None
        self.action_space: Optional[str] = env.get_action_space() if hasattr(env, 'get_action_space') else None
        
        # Database for in-context learning
        self.db = db

        # File to log to
        self.f: None

    """ Helper functions """

    def _trajectory_to_string(self, trajectory):
        """
        Convert a trajectory dictionary to a formatted string representation.
        
        Args:
            trajectory: Dictionary containing trajectory components
            
        Returns:
            String representation of the trajectory
        """
        # First roll up the trajectory
        trajectory = self._roll_up_trajectory([trajectory])[0]
        key_order = ["goal", "plan", "trajectory", "available_actions"]
        # Now convert to a string following key order
        trajectory_string = "\n"
        for key in key_order:
            if key in trajectory:
                trajectory_string += f"{key}: {trajectory[key]}\n"
        return trajectory_string

    def _create_conversation(self, keys: List[str], available_actions: List[Action]) -> List[Dict]:
        """
        Create a conversation string with the observation and action history.
        
        This formats the agent's history into a structured conversation that can be
        used as input to the LLM for decision making.
        
        Args:
            keys: List of keys to include in the conversation
            available_actions: List of available actions
            
        Returns:
            Formatted conversation string
        """
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
        return self._trajectory_to_string(trajectory_dict)
    
    def _create_conversation_for_finetune(self) -> List[Dict]:
        """
        Create a conversation in the format used for fine-tuning.
        
        This formats the agent's history into the specific format expected by
        OpenAI's fine-tuning API, with alternating user and assistant messages.
        
        Returns:
            List[Dict]: List of messages in the format used by OpenAI's fine-tuning API
        """
        messages = []
        
        # Add system message
        system_prompt = """You are a ReAct agent that helps users accomplish tasks. 
Given a goal, you will receive observations about the environment and respond with your reasoning and actions.
For each observation, first think through the problem step by step (Thought), then decide on an action (Action).
Your actions should be clear, concise, and directly executable in the environment."""
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add goal and initial observation
        if self.goal:
            initial_obs = self.observation_history[0].structured if self.observation_history else ""
            messages.append({
                "role": "user",
                "content": f"Goal: {self.goal}\nInitial observation: {initial_obs}"
            })
        
        # Process each step in the trajectory
        for i in range(len(self.reasoning_history)):
            # Add assistant's response with reasoning and action
            if i < len(self.reasoning_history) and i < len(self.action_history):
                assistant_content = f"Thought: {self.reasoning_history[i]}\nAction: {self.action_history[i].text}"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # Add next observation if available
            if i + 1 < len(self.observation_history):
                messages.append({
                    "role": "user",
                    "content": f"Observation: {self.observation_history[i+1].structured}"
                })
        return messages
    
    def _roll_up_trajectory(self, entries):
        """
        Roll up the trajectory into a single string representation.
        
        This interleaves observations, reasoning, and actions to create a
        coherent narrative of the agent's interaction with the environment.
        
        Args:
            entries: List of dictionaries containing trajectory components
            
        Returns:
            List of dictionaries with rolled-up trajectories
        """
        interleaved_keys = ["observation", "reasoning", "action"]
        value_type = list(entries[0].keys())
        if any(k in value_type for k in interleaved_keys):
            for i in range(len(entries)):
                trajectory_string = "\n"
                for j in range(len(entries[i][interleaved_keys[0]])):
                    for k in interleaved_keys:
                        if len(entries[i][k]) > j and len(entries[i][k][j]) > 0:
                            trajectory_string += f"{k}: {entries[i][k][j]}\n"
                entries[i]["trajectory"] = trajectory_string
                # If the category is "intercode_sql", then add the second-to-last action text to the goal
                if self.category == "sql" and "action" in entries[i] and len(entries[i]["action"]) >= 2:
                    if "SELECT" in entries[i]["action"][-2]:
                        entries[i]["goal"] += f"\nSolution query: {entries[i]['action'][-2]}"
                # Delete the interleaved keys from the dictionary
                for k in interleaved_keys:
                    del entries[i][k]
        return entries
    
    def _in_context_prompt(self, in_context_data):
        """
        Create a system prompt containing in-context examples from similar episodes.
        
        This formats retrieved examples into a system prompt that can guide the LLM's
        decision making by providing relevant examples of successful or failed episodes.
        
        Args:
            in_context_data: Dictionary or list of dictionaries containing in-context examples
            
        Returns:
            Formatted system prompt with in-context examples
        """
        in_context_list = []
        # If this is a dictionary, we need to unpack it
        if isinstance(in_context_data, dict):
            for key, value in in_context_data.items():
                in_context_list.append(value)
        else:
            in_context_list = [in_context_data]
        
        for in_context in in_context_list:
            # Determine which set of entries to use
            success_entries, failure_entries = in_context[0], in_context[1]
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
    
    def _retrieve_in_context_data(self, key_type, key, value_type, outcome="winning", k=5, window=20, filtered_environment_id=None) -> List[Dict]:
        """
        Retrieve in-context examples from the database.
        
        This is the core method for retrieving relevant examples based on specified
        key types and values. It supports both trajectory-level and state-level retrieval.
        
        Args:
            key_type: Type(s) of keys to match against in the database
            key: Value(s) of keys to match
            value_type: Type(s) of values to extract from matched examples
            outcome: Whether to retrieve "winning" or "losing" examples
            k: Number of examples to retrieve
            window: Window size for state-level retrieval
            filtered_environment_id: Optional environment ID to filter by
            
        Returns:
            List of dictionaries containing retrieved examples
        """
        success_entries, failure_entries = self.db.retrieve_similar_entries(key_type, key, outcome=outcome, k=k, window=window, filtered_environment_id=filtered_environment_id)
        if self.f:
            success_entry_ids = [entry['id'] for entry in success_entries]
            failure_entry_ids = [entry['id'] for entry in failure_entries]
            self.f.write(f"Success entry ids: {success_entry_ids}\n")
            self.f.write(f"Failure entry ids: {failure_entry_ids}\n")
        # Now figure out which part of the examples to return in-context
        if not isinstance(value_type, list): # Check that this is a list, not a string
            value_type = [value_type]
        # Filter out the keys that are not in the value_type
        success_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in success_entries]
        failure_entries = [{k: v for k, v in entry.items() if k in value_type} for entry in failure_entries]
        return success_entries, failure_entries
    
    """ Main components used by agent's choose_action function """

    # A wrapper function for the retrieve_in_context_data function when getting state-level data with a window
    def retrieve_state_data(self, trajectory_key_types, trajectory_keys, state_key_types, state_keys, value_types, outcome, k, window, filtered_environment_id="self") -> List[Dict]:
        """
        Retrieve state-level in-context examples from the database.
        
        This method performs a two-stage retrieval process:
        1. First identifies relevant trajectories using trajectory-level keys
        2. Then finds the most relevant states within those trajectories using state-level keys
        3. Finally extracts a window of states around the most relevant state
        
        This allows finding not just similar trajectories, but the most relevant parts
        of those trajectories for the current situation.
        
        Args:
            trajectory_key_types: Types of keys for trajectory-level matching
            trajectory_keys: Values of keys for trajectory-level matching
            state_key_types: Types of keys for state-level matching
            state_keys: Values of keys for state-level matching
            value_types: Types of values to extract
            outcome: Whether to retrieve "winning" or "losing" examples
            k: Number of examples to retrieve
            window: Window size around the matched state
            filtered_environment_id: Optional environment ID to filter by
            
        Returns:
            List of dictionaries containing retrieved examples
        """
        # Combine the trajectory and state keys for now since the wrapped function will split them back out
        if filtered_environment_id == "self":
            filtered_environment_id = self.environment_id
        key = trajectory_keys + state_keys
        key_type = trajectory_key_types + state_key_types
        return self._retrieve_in_context_data(key_type, key, value_types, outcome, k, window, filtered_environment_id=filtered_environment_id)
    
    def retrieve_trajectory_data(self, key_types, keys, value_types, outcome, k, filtered_environment_id="self") -> List[Dict]:
        """
        Retrieve trajectory-level in-context examples from the database.
        
        This method retrieves entire trajectories that match the specified keys,
        useful for high-level planning and understanding complete episodes.
        
        Args:
            key_types: Types of keys for trajectory-level matching
            keys: Values of keys for trajectory-level matching
            value_types: Types of values to extract
            outcome: Whether to retrieve "winning" or "losing" examples
            k: Number of examples to retrieve
            filtered_environment_id: Optional environment ID to filter by
            
        Returns:
            List of dictionaries containing retrieved examples
        """
        if filtered_environment_id == "self":
            filtered_environment_id = self.environment_id
        return self._retrieve_in_context_data(key_types, keys, value_types, outcome, k, filtered_environment_id=filtered_environment_id)

    async def create_plan(self, observation: Observation, available_actions: List[Action], in_context_data = None) -> str:
        """
        Generate a high-level plan for the agent to follow.
        
        This is one of the core decision-making primitives that uses the LLM to
        create a plan for achieving the goal based on the initial observation.
        
        Args:
            observation: Initial observation from the environment
            available_actions: List of available actions
            in_context_data: Optional in-context examples to guide planning
            
        Returns:
            Generated plan as a string
        """
        conversation = []
        system_prompt = f"You are an expert at generating high-level plans of actions to achieve a goal. "
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        curr_prompt = f"goal: {self.goal}\n plan: "
        conversation.append({"role": "user", "content": curr_prompt})
        try:
            plan = await self.llm.generate_chat(conversation)
            plan = plan.strip()
        except Exception as e:
            logger.error(f"Error generating plan: {str(e)}")
            plan = ""
        self.plan = plan
        return plan

    async def reason(self, observation: Observation, available_actions, in_context_data = None) -> List[Dict]:
        """
        Reason about the current observation and available actions.
        
        This is one of the core decision-making primitives that uses the LLM to
        generate reasoning about the current state and possible next actions.
        
        Args:
            observation: Current observation from the environment
            available_actions: List of available actions
            in_context_data: Optional in-context examples to guide reasoning
            
        Returns:
            Generated reasoning as a string
        """
        self.observation_history.append(observation)
        conversation = []
        # Add system prompt
        system_prompt = f"You are an expert at reasoning about the most appropriate action to take towards achieving a goal. "
        if self.config.get("give_action_space", False):
            system_prompt += "\nHere is your action space:\n" + self.action_space['description']
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        # Add conversation history
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "reasoning: "})
        stop = None if self.config.get("multiline_reasoning", False) else ["\n"]
        reasoning = await self.llm.generate_chat(conversation, stop=stop)
        self.reasoning_history.append(reasoning)
        return reasoning

    async def act(self, observation: Observation, available_actions: List[Action], reasoning: Union[str, None] = None, in_context_data= None) -> Tuple[Action, List[Dict]]:
        """
        Select an action from available actions given the current observation.
        
        This is one of the core decision-making primitives that uses the LLM to
        select an action based on the current observation and reasoning.
        
        Args:
            observation: Current observation from the environment
            available_actions: List of available actions
            reasoning: Optional reasoning about the current state
            in_context_data: Optional in-context examples to guide action selection
            
        Returns:
            Selected action
        """
        # Create a conversation with observations and actions so far
        conversation = []
        # Want the system prompt to be standardized. Should have environment and goal info, as well as observation and action format. 
        system_prompt = f"""You are an agent in an environment. Given the current observation, you must select an action to take towards achieving the goal: {self.goal}."""
        # If this is a TRAD agent, we want to add the action space to the system prompt
        if self.config.get("give_action_space", False):
            system_prompt += "\nHere is your action space:\n" + self.action_space['description']
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        conversation.append({"role": "system", "content": system_prompt})
        conversation.append({"role": "user", "content": self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], available_actions) + "action: "})
        
        # Select action
        stop = None if self.config.get("multiline_action", False) else ["\n"]
        action = await self.llm.generate_chat(conversation, stop=stop)
        action = Action(text=action)
        self.action_history.append(action)

        return action
    
    async def act_finetune(self, observation: Observation) -> Tuple[Action, List[Dict]]:
        """
        Select an action using a fine-tuned model.
        
        This method is used by the FinetuneAgent to directly map observations to
        actions using a fine-tuned LLM, bypassing the standard reasoning and planning steps.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Selected action
        """
        conversation = self._create_conversation_for_finetune()
        response = await self.llm.generate_chat(conversation, stop=None)
        # Strip the "Action: " prefix
        # Also get the reasoning
        action = response.split("Action:")[1].strip()
        reasoning = response.split("Thought:")[1].split("Action:")[0].strip()
        action = Action(text=action)
        self.action_history.append(action)
        self.reasoning_history.append(reasoning)
        return action
    
    async def reflect(self, in_context_data = None) -> List[Dict]:
        """
        Reflect on the completed episode.
        
        This generates reflections on what worked well and what could be improved,
        which can be stored and used as in-context examples for future episodes.
        
        Args:
            in_context_data: Optional in-context examples to guide reflection
            
        Returns:
            Generated reflection as a string
        """
        user_prompt = self._create_conversation(["goal", "plan", "observation", "reasoning", "action"], [])
        # Identify success or failure
        reward = self.reward_history[-1]
        if reward == 1:
            success = True
        else:
            success = False
        # Create success vs failure reflection prompt
        if success: 
            reflection_prompt = "You are provided with an ultimately successful trajectory of observations and actions taken."
        else:
            reflection_prompt = "You are provided with an ultimately failed trajectory of observations and actions taken."
        reflection_prompt += "You will have to solve this exact goal again in the future. Write down three key facts about this task in this environment that will be provided to you in the future."
        system_prompt = f"You are an agent in an environment. Given the goal: {self.goal}, {reflection_prompt}"
        if in_context_data:
            system_prompt += self._in_context_prompt(in_context_data)
        response = await self.llm.generate_chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], stop=None) # Reflection can be multiline
        self.reflection = response
        return response
    
    def store_episode(self):
        """
        Store an episode in the database for future retrieval.
        
        This saves the complete episode data including observations, reasoning,
        actions, rewards, plan, reflection, and summary to the database.
        """
        # Add one more state to the observation history
        # If success, add a "Task complete" state
        if self.reward_history[-1] == 1:
            self.observation_history.append(Observation("Task complete"))
        else:
            self.observation_history.append(Observation("Task failed"))
        self.db.store_episode(self.environment_id, self.goal, self.category,self.observation_history, self.reasoning_history, self.action_history, self.reward_history, self.plan, self.reflection, self.summary)
    
    def clean_history(self):
        """
        Clean the agent's history by removing unnecessary observations.
        
        This removes "Nothing happens" observations and their corresponding
        reasoning, actions, and rewards to create a cleaner history.
        """
        nothing_indices = [i for i in range(len(self.observation_history)) if "Nothing happens" in self.observation_history[i].structured]
        
        for idx in sorted(nothing_indices, reverse=True):
            del self.observation_history[idx]
            if idx > 0:
                if idx-1 < len(self.reasoning_history):
                    del self.reasoning_history[idx-1]
                if idx-1 < len(self.action_history):
                    del self.action_history[idx-1]
                if idx-1 < len(self.reward_history):
                    del self.reward_history[idx-1]
    
    """ Placeholder functions for agent's choose_action and analyze_episode functions """
    
    async def choose_action(self, obs, valid_actions):
        """
        Choose an action from available actions given the current observation.
        
        This is the main method that subclasses should implement to define their
        specific decision-making process using the primitives provided by BaseAgent.
        
        Args:
            obs: Current observation from the environment
            valid_actions: List of valid actions that can be taken
            
        Returns:
            Selected action to take
        """
        pass

    """ Between-episode functions for the outer loop """
        
    def clear_history(self):
        """
        Clear the agent's history completely.
        
        This resets all history variables to empty lists or None.
        """
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.in_context_data = None
        self.reflection = None
        self.summary = None

    def reset(self):
        """
        Reset agent state between episodes.
        
        This is similar to clear_history but specifically intended for
        use between episodes in a multi-episode run.
        """
        self.observation_history = []
        self.action_history = []
        self.reasoning_history = []
        self.reward_history = []
        self.plan = None
        self.reflection = None
        self.summary = None
        self.in_context_data = None
