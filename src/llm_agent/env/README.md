# Environment Framework for LLM Agents

This directory contains the environment framework for LLM agents to interact with various tasks. The framework is designed to be flexible and extensible, allowing for easy integration of new environments.

## Base Environment Interface

The `BaseEnv` abstract class in `base_env.py` defines the interface that all environments must implement. This standardized interface allows agents to interact with different environments in a consistent manner.

### Core Components

- **Observation**: A dataclass representing the environment's state, containing both structured data and text descriptions.
- **Action**: A dataclass representing an agent's action, containing text descriptions.
- **BaseEnv**: Abstract base class that all environments must inherit from.

### Required Methods

All environments must implement these abstract methods:

```python
@abstractmethod
def reset(self) -> Observation:
    """Reset environment to initial state and return initial observation"""
    pass

@abstractmethod
def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
    """Take action in environment and return next observation, reward, done flag, and info dict"""
    pass

@abstractmethod
def get_action_space(self) -> Dict:
    """Return JSON schema describing valid action format"""
    pass
```

### Optional Methods

Environments can optionally override these methods:

```python
def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
    """Return list of available actions (default: empty list)"""
    return []

@property
def current_observation(self) -> Observation:
    """Return current environment observation"""
    return self._observation
```

## Implemented Environments

### AlfWorld Environment

`alfworld_env.py` and `alfworld_train_env.py` implement environments for the [ALFWorld](https://github.com/alfworld/alfworld) text-based household tasks.

- **Key Features**:
  - Text-based household tasks (e.g., "put a clean mug on the coffee table")
  - Observation includes room description and inventory
  - Actions are natural language commands (e.g., "take mug from table")
  - Supports expert demonstrations

- **Setup Requirements**:
  ```bash
  pip install alfworld==0.3.5
  export ALFWORLD_DATA=./data/alfworld
  alfworld-download
  ```

### Intercode SQL Environment

`intercode_sql_env.py` implements an environment for the [Intercode-SQL](https://github.com/princeton-nlp/intercode) database query tasks.

- **Key Features**:
  - SQL query tasks against various databases
  - Observation includes database schema and query results
  - Actions are SQL queries
  - Supports Docker-based execution environment

- **Setup Requirements**:
  - Follow the instructions in the Intercode repository to set up and run the Docker container

### WordCraft Environment

`wordcraft.py` implements an environment for the [WordCraft](https://github.com/minqi/wordcraft) word crafting game.

- **Key Features**:
  - Text-based crafting game where elements combine to create new elements
  - Observation includes current inventory and goal
  - Actions are element combinations
  - Includes recipe book with crafting rules

- **Setup Requirements**:
  - No additional installation needed beyond the base requirements

## Creating a New Environment

To create a new environment, follow these steps:

1. **Create a new Python file** in the `env` directory.

2. **Import the base classes**:
   ```python
   from ..base_env import BaseEnv, Observation, Action
   from typing import Dict, List, Any, Tuple, Optional
   ```

3. **Implement the environment class**:
   ```python
   class MyNewEnvironment(BaseEnv):
       def __init__(self, config: Dict):
           super().__init__(config)
           # Initialize your environment-specific attributes
           self._observation = None

       def reset(self) -> Observation:
           # Reset environment state
           # Return initial observation
           obs = Observation(structured={"text": "Initial state description"})
           self._observation = obs
           return obs

       def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
           # Process the action
           # Update environment state
           # Return (observation, reward, done, info)

       def get_action_space(self) -> Dict:
           # Return JSON schema for valid actions
           return {
               "type": "object",
               "properties": {
                   "action": {"type": "string"}
               }
           }

       def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
           # Optional: Return list of available actions
           return [Action(text="action1"), Action(text="action2")]
   ```

4. **Register your environment** in the appropriate configuration files.

## Example: Minimal Environment Implementation

Here's a minimal example of a simple text-based environment:

```python
from ..base_env import BaseEnv, Observation, Action
from typing import Dict, List, Any, Tuple, Optional

class SimpleTextEnv(BaseEnv):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.state = "initial"
        self.goal = config.get("goal", "reach the final state")
        self._observation = None
        self.steps = 0
        self.max_steps = config.get("max_steps", 10)

    def reset(self) -> Observation:
        self.state = "initial"
        self.steps = 0
        obs = Observation(structured={
            "text": f"You are in the {self.state} state. Your goal is to {self.goal}.",
            "state": self.state,
            "steps": self.steps
        })
        self._observation = obs
        return obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.steps += 1

        # Process action
        if action.text == "move forward":
            if self.state == "initial":
                self.state = "middle"
            elif self.state == "middle":
                self.state = "final"

        # Create observation
        obs = Observation(structured={
            "text": f"You are in the {self.state} state. Your goal is to {self.goal}.",
            "state": self.state,
            "steps": self.steps
        })
        self._observation = obs

        # Calculate reward and done flag
        reward = 1.0 if self.state == "final" else 0.0
        done = self.state == "final" or self.steps >= self.max_steps

        # Additional info
        info = {"state": self.state}

        return obs, reward, done, info

    def get_action_space(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move forward", "move backward", "stay"]
                }
            }
        }

    def get_available_actions(self, info: Optional[Dict] = None) -> List[Action]:
        return [
            Action(text="move forward"),
            Action(text="move backward"),
            Action(text="stay")
        ]
```

## Environment Configuration

When initializing environments, a configuration dictionary is passed to the constructor. Common configuration parameters include:

- `max_steps`: Maximum number of steps per episode
- `num_attempts`: Number of attempts allowed for each action
- Environment-specific parameters (e.g., problem ID, difficulty level)

Example configuration:

```python
config = {
    "max_steps": 50,
    "num_attempts": 3,
    "problem_id": 42,
    "difficulty": "hard"
}
env = MyEnvironment(config)
```

## Using Environments with Agents

The standard interaction loop between an agent and environment follows this pattern:

```python
# Initialize environment
env = MyEnvironment(config)

# Reset environment to get initial observation
obs = env.reset()

done = False
while not done:
    # Agent selects action based on observation
    action = agent.choose_action(obs)

    # Environment processes action
    obs, reward, done, info = env.step(action)

    # Optional: Use info dict for additional processing
    if info.get("error"):
        print(f"Error: {info['error']}")
```

This standardized interface allows agents to interact with any environment that implements the `BaseEnv` interface.
