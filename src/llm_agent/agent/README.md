# Agent Algorithms

This folder contains implementations of various agent algorithms built on the `BaseAgent` framework. Each agent uses different strategies for decision-making in sequential environments.

## Core Agent Primitives

The `BaseAgent` class provides several key primitives that can be used to build different agent algorithms:

### Decision-Making Components
- `create_plan`: Generates a high-level plan for achieving the goal
- `reason`: Produces reasoning about the current state and possible actions
- `act`: Selects an action based on observation, reasoning, and available actions

### In-Context Learning
- `get_trajectory_data`: Retrieves similar episodes from the database for trajectory-level context
- `get_state_data`: Retrieves similar states from the database for state-level context
- `_in_context_prompt`: Formats retrieved examples into a system prompt

### Episode Analysis
- `reflect`: Generates reflections on completed episodes
- `store_episode`: Saves episode data to the database for future retrieval
- `clean_history`: Cleans the agent's history by removing unnecessary observations

## Agent Implementations

### ZeroShotReact
Basic ReAct agent that uses no in-context examples, relying solely on the LLM's capabilities.

### ReAct
Enhances the ReAct paradigm with in-context examples of successful trajectories for the same goal.

### TrajBS (Trajectory-Based Search)
Extends ReAct by retrieving successful trajectories with similar goals and categories to inform planning and action selection.

### TrajBS_NoPlan
Variant of TrajBS that skips the planning phase and directly retrieves examples based on the current observation, focusing on state-level retrieval for reasoning and action selection.

### TrajBS_Flex
Flexible TrajBS implementation that combines trajectory-level retrieval for planning with state-level retrieval for reasoning and action. Dynamically updates in-context examples based on the current reasoning state.

### Synapse
Focuses on providing in-context examples for reasoning and action selection without explicit planning.

### Reflexion
Uses reflections from previous failed attempts as in-context examples to improve performance.

## In-Context Learning Keys and Values

The behavior of agents is largely governed by how they retrieve and use in-context examples:

- **Key Types**: Determine what to match against in the database
  - Goal-based: `["goal"]` - Match examples with similar goals
  - Category-based: `["goal", "category"]` - Match examples with similar goals and categories
  - Observation-based: `["observation"]` - Match examples with similar observations
  - Reasoning-based: `["reasoning"]` - Match examples with similar reasoning patterns
  - Environment-based: `["environment_id"]` - Match examples from the same environment

- **Value Types**: Determine what information to extract from matched examples
  - `"goal"`: The objective of the episode
  - `"plan"`: High-level plan for achieving the goal
  - `"observation"`: Environmental state observations
  - `"reasoning"`: Agent's reasoning about the state
  - `"action"`: Actions taken by the agent
  - `"reflection"`: Post-episode analysis

Different agents combine these keys and values in various ways to achieve different behaviors:
- TrajBS uses `["goal", "category"]` keys to retrieve `["goal", "plan"]` values for planning
- TrajBS_NoPlan uses `["goal", "observation"]` keys to retrieve `["goal", "observation", "reasoning", "action"]` values
- TrajBS_Flex uses state-specific retrieval with `["reasoning"]` keys to dynamically update context

### Combined Trajectory and State-Level Retrieval

When both trajectory-level and state-level keys are provided (as in `get_state_data`), the system performs a two-stage retrieval process:

1. First, it identifies relevant trajectories using the trajectory-level keys
2. Then, within those trajectories, it finds the most relevant states using the state-level keys
3. Finally, it extracts a window of states around the most relevant state

This approach allows the agent to find not just similar trajectories, but the most relevant parts of those trajectories for the current situation. For example, TrajBS_Flex uses this capability to dynamically update its in-context examples based on the current reasoning state, providing more targeted and relevant examples as the episode progresses.

## Creating a New Agent

To implement a new agent algorithm:

1. Subclass `BaseAgent`
2. Implement the `choose_action` method to define your agent's decision-making process
3. Optionally implement `analyze_episode` for post-episode analysis

Example:
```python
from llm_agent.agent.base_agent_v2 import BaseAgent

class MyNewAgent(BaseAgent):
    def __init__(self, *args):
        super().__init__(*args)
    
    async def choose_action(self, obs, valid_actions):
        # Implement your agent's decision-making logic here
        # Use the primitives provided by BaseAgent
        return action
    
    async def analyze_episode(self):
        # Optional: Implement post-episode analysis
        self.clean_history()
        # Additional analysis...
```

## Fine-tuned Agents

The `FinetuneAgent` class provides a different approach to agent implementation:

- Uses a fine-tuned LLM specifically trained for the task
- Bypasses the standard reasoning and planning steps
- Directly maps observations to actions using the fine-tuned model
- Requires no in-context examples, relying instead on knowledge embedded in the model weights

This approach can be more efficient and potentially more effective for specific tasks where sufficient training data is available.

## Configuration

Agents can be configured through the `config` parameter passed to the constructor. Common configuration options include:
- `max_retries`: Maximum number of retries for LLM calls
- `memory_size`: Number of past observations/actions to remember
- `temperature`: Temperature for LLM sampling
- `give_action_space`: Whether to include action space in prompts
- `num_ic`: Number of in-context examples to retrieve
