# Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks

This repository contains code for the paper [Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks](https://arxiv.org/abs/2505.00234). 

## Installation

1. Install [uv](https://docs.astral.sh/uv/)

    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    or follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone this repository: 
    ```bash
    git clone git@github.com:VSAnimator/agent_algo_bench.git
    cd agent_algo_bench
    ```

3. Install dependencies with uv:

    ```bash
    uv venv algoBench --python 3.11
    source algoBench/bin/activate
    uv pip install -r base_requirements.txt
    uv pip install -e .
    ```

---

## Repository Structure

The repository is organized as follows:

- **src/llm_agent/**: Core implementation of the agent framework
  - **env/**: Environment implementations ([detailed documentation](src/llm_agent/env/README.md))
  - **agent/**: Agent implementations
  - **database/**: Database utilities for storing and retrieving in-context examples
  - **llm/**: LLM wrappers and utilities

- **scripts/**: Executable scripts for running experiments
  - **run_agent.py**: Main script for running agents on environments
  - **train_*.sh**: Scripts for training agents on specific environments
  - **test_*.sh**: Scripts for testing agents on specific environments

- **data/**: Data storage for environment data and in-context examples

## Current Experimental Usage

### (Optional) Step 1: Ingest Data into Database

The ```data/starter``` folder currently holds starter databases for alfworld and wordcraft

### Step 2: Self-generate Database from Train Environments

Run the train_alfworld.sh script to generate a database of in-context examples:

```bash
./bash-scripts/train/train_unified.sh --config default:alfworld:rap_flex: --llm gpt-4o-mini
```

### Step 3: Run on Test Environments with Generated Database

```bash
./bash-scripts/test/test_unified.sh --config default:alfworld:rap_flex: --llm gpt-4o-mini
```

## Scripts Usage

The `scripts/` directory contains various utilities for running experiments:

### Main Agent Runner

```bash
python scripts/run_agent.py --llm <llm_name> --agent_type <agent_type> --db_path <database_path> --db_name <database_name> --num_passes <num_passes>
```

Key parameters:
- `--llm`: LLM to use (e.g., openai/gpt-4o-mini)
- `--agent_type`: Agent algorithm to use (e.g., rap, expel)
- `--db_path`: Path to the database of in-context examples
- `--db_name`: Name of the database to use
- `--num_passes`: Number of passes through the environment
- `--env`: Environment to use (default: alfworld)

Example:
```bash
python scripts/run_agent.py --llm openai/gpt-4o-mini --agent_type rap --db_path /data/rl/clone_test/data/alfworld_expel/learning.db --db_name expel_rap_testonly --num_passes 1
```

### Database Ingestion Scripts

The repository includes scripts for ingesting data into the database:

```bash
# For ALFWorld
python src/llm_agent/database/ingest_alfworld.py

# For WebShop
python src/llm_agent/database/ingest_webshop.py

# For WordCraft
python src/llm_agent/database/ingest_wordcraft_logs.py
```

## Options

### Environments Supported

- [Alfworld](https://github.com/alfworld/alfworld) (flag --env alfworld): text-based kitchen simulator. Download data:
    ```bash
    uv pip install -vvv "textworld[pddl]>=1.6.1"
    uv pip install alfworld==0.3.5
    export ALFWORLD_DATA=./data/alfworld
    alfworld-download
    ```

- [Intercode-SQL](https://github.com/princeton-nlp/intercode) (flag --env ic_sql): follow the instructions to set up and run the docker container. 

- [Wordcraft](https://github.com/minqi/wordcraft) (flag --env wordcraft): no additional installation needed

- Add your own environment: see [Adding your own environment](#adding-your-own-environment)

### LLM Options

For API LLMs, we support arbitrary LLMs that are supported by LiteLLM. Make sure to set the API key in the appropriate environment variable.

For local LLMs, set up [VLLM](https://github.com/vllm-project/vllm), and point the script to the port via the flag --vllm_port.

### Data Sources

Many of the included agent algorithms leverage a database of in-context examples to learn. You have the option to start with an empty database or load a pre-existing database. The data sources for the included environments are:

- Alfworld: ```python src/llm_agent/database/ingest_alfworld.py```
- WebShop: ```python src/llm_agent/database/ingest_webshop.py```
- Intercode-SQL: TODO
- WordCraft: ```python src/llm_agent/database/ingest_wordcraft_logs.py```

### Agent Algorithms

The repository supports several agent algorithms:

1. **TODO**

More details on agent algorithms can be found in the paper.

## Extensibility

### Adding Your Own Environment

The `src/llm_agent/env/base_env.py` file defines the interface for environments. To add your own environment:

1. Create a new Python file in `src/llm_agent/env/envs/`
2. Implement a class that inherits from `BaseEnv`
3. Implement the required methods: `reset()`, `step()`, and `get_action_space()`
4. Optionally implement `get_available_actions()`

See the [Environment README](src/llm_agent/env/README.md) for detailed instructions and examples.

### Adding Your Own Agent Algorithm

To add a new agent algorithm:

1. Create a new Python file in `src/llm_agent/agent/`
2. Implement a class that inherits from `BaseAgent`
3. Implement the required methods: `choose_action()`, optionally `analyze_episode()`
4. Register your agent in `src/llm_agent/agent/__init__.py`

See the [Agent README](src/llm_agent/agent/README.md) for detailed instructions and examples.

### Creating Your Own Database

See the [Database README](src/llm_agent/database/README.md) for details on the database interface. We have implemented versions with both SQLite and PostgreSQL backends. 

## License

[MIT License](LICENSE)
