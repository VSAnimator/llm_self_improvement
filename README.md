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

## Current experimental usage

### Self-generate database from train environments

Run the train_alfworld.sh script to generate a database of in-context examples:

```bash
./train_alfworld.sh
```

Zhiqiang, this creates a new database for training runs with 3, 6, and 10 in-context examples. These databases are copies of the alfworld_expel database that has the initial human-curated in-context examples. The script launches 3 parallel processes, each running the training script with a different number of in-context examples. If run correctly, the script will 1) create 3 new databases, 2) run 3 training processes, 3) create 3 new databases with the training results, while saving backup copies of the original databases. There should be no errors, and the databases should contain all of the trajectories from the training runs (can check the trajectory_*.json files to confirm).

### Run on test environments with generated database

```bash
./test_alfworld.sh
```

## Options

### Environments supported

- [Alfworld](https://github.com/alfworld/alfworld) (flag --env alfworld): text-based kitchen simulator. Download data:
    ```bash
    uv pip install -vvv "textworld[pddl]>=1.6.1"
    uv pip install alfworld==0.3.5
    export ALFWORLD_DATA=./data/alfworld
    alfworld-download
    ```
- [WebShop](https://github.com/princeton-nlp/WebShop) (flag --env webshop): follow the instructions to set up the server, the env assumes it is running on port 3000.

- [Intercode-SQL](https://github.com/princeton-nlp/intercode) (flag --env ic_sql): follow the instructions to set up and run the docker container. 

- [Wordcraft](https://github.com/minqi/wordcraft) (flag --env wordcraft): no additional installation needed

- [Gymnasium](https://gymnasium.farama.org) (flag --env gymnasium): Just pass the name of the gymnasium environment as an additional flag --gym_env_name

- Add your own environment: see [Adding your own environment](#adding-your-own-environment)

### LLM options

We use the LiteLLM wrapper. 

API LLMs tested:
1. OpenAI (GPT-4o, GPT-4o-mini)
2. Anthropic (Claude 3.5 Sonnet)
3. Google (Gemini 2 Flash)
4. Meta (Llama 3.1 8B)
5. Together (Llama 3.1 70B)

For API LLMs, make sure to set the API key in the appropriate environment variable.

Local LLMs tested:
1. Llama 3.1 8B

For local LLMs, set up [VLLM](link), and point the script to the port via the flag --vllm_port.

Feel free to try any other API or local LLMs that are supported by LiteLLM. Make sure that the LLM options chosen (ex. structured outputs) are supported by the LLM.

### Data sources

Many of the included agent algorithms leverage a database of in-context examples to learn. You have the option to start with an empty database or load a pre-existing database. The data sources for the included environments are:

- Alfworld: ```python src/llm_agent/database/ingest_alfworld.py```
- WebShop: ```python src/llm_agent/database/ingest_webshop.py```
- Intercode-SQL: TODO
- WordCraft: ```python src/llm_agent/database/ingest_wordcraft_logs.py```

### Agent algorithms

TODO

## Example usage

```bash
python scripts/run_agent_v2.py --llm openai/gpt-4o-mini --agent_type rap --db_path /data/rl/clone_test/data/alfworld_expel/learning.db --db_name expel_rap_testonly --num_passes 1
```

## Extensibility

### Adding your own environment

src/llm_agent/env/base_env.py defines the interface for environments. To add your own environment, you can either subclass the existing environment or implement the interface yourself.

### Adding your own agent algorithm

TODO

### Creating your own database

TODO
