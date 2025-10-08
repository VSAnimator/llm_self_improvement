# Agent Learning Database

This folder contains implementations of the learning database used by agents to store and retrieve episodes and states. The database serves as the memory system for in-context learning, allowing agents to leverage past experiences to improve performance on new tasks.

## Core Functionality

The learning database provides several key capabilities:

### Storage

- **Episodes/Trajectories**: Complete sequences of observations, reasoning, actions, and rewards for a given goal
- **States**: Individual steps within trajectories, including observations, reasoning, and actions

### Retrieval

- **Similarity Search**: Find episodes or states similar to a given query using vector embeddings
- **Exact Matching**: Retrieve episodes by exact environment ID
- **Contrastive Pairs**: Get successful and failed episodes for the same environment
- **Windowed Retrieval**: Extract relevant portions of trajectories around critical states

### Embedding

- **Text Encoding**: Convert text fields to vector embeddings using sentence transformers
- **Vector Storage**: Store embeddings for efficient similarity search
- **Multi-field Search**: Compute similarity across multiple fields (e.g., goal and category)

## Common Interface

Both database implementations share a common interface with these key methods:

### Initialization
```python
def __init__(self, db_path: str = "path/to/db")
```

### Episode Storage
```python
def store_episode(
    self,
    environment_id: str,
    goal: str,
    category: str,
    observations: List[Any],
    reasoning: Optional[List[str]],
    actions: List[Any],
    rewards: List[float],
    plan: Optional[str],
    reflection: Optional[str],
    summary: Optional[str]
)
```

### Similarity Search
```python
def get_similar_entries(
    self,
    key_type: Union[str, List[str]],
    key: Union[str, List[str]],
    k: int = 5,
    outcome: str = None,
    window: int = 1,
    filtered_environment_id: str = None
) -> tuple[List[Dict], List[Dict]]
```

## Implementations

### SQLite Implementation (`learning_db.py`)

The SQLite implementation uses:
- **SQLite**: For relational data storage
- **FAISS**: For vector similarity search
- **SentenceTransformer**: For text embedding generation

Features:
- Simple setup with minimal dependencies
- File-based storage suitable for single-process applications
- Separate database files for trajectories and states
- In-memory index for fast similarity search

Limitations:
- Limited concurrency support
- Performance bottlenecks with large datasets
- No built-in vector operations in SQLite

### PostgreSQL Implementation (`learning_db_postgresql_new.py`)

The PostgreSQL implementation uses:
- **PostgreSQL**: For relational data storage
- **pgvector**: For native vector operations and indexing
- **SentenceTransformer**: For text embedding generation

Features:
- Full ACID compliance with transaction support
- Excellent concurrency for multi-process applications
- Native vector operations for efficient similarity search
- Support for various index types (HNSW, IVFFlat)
- Better performance with large datasets
- Built-in replication and backup capabilities

Advanced capabilities:
- **Concurrent Indexing**: Create indexes without blocking writes using `CREATE INDEX CONCURRENTLY`
- **Vector Indexing**: Efficient similarity search with pgvector's HNSW indexes
- **Transaction Isolation**: MVCC model prevents read/write conflicts
- **Horizontal Scaling**: Support for replication and sharding

## Usage Example

```python
from llm_agent.database.learning_db import LearningDB
# or
from llm_agent.database.learning_db_postgresql_new import LearningDB

# Initialize the database
db = LearningDB(db_path="path/to/db")

# Store an episode
db.store_episode(
    environment_id="env123",
    goal="Find the treasure",
    category="exploration",
    observations=[...],
    reasoning=[...],
    actions=[...],
    rewards=[0, 0, 1],
    plan="First, explore the cave...",
    reflection="I should have checked for traps",
    summary="Successfully found the treasure"
)

# Retrieve similar episodes
success_entries, failure_entries = db.get_similar_entries(
    key_type=["goal", "category"],
    key=["Find the treasure", "exploration"],
    k=5
)
```

## Choosing an Implementation

- Use the **SQLite implementation** for:
  - Development and testing
  - Single-process applications
  - Smaller datasets
  - Environments without PostgreSQL

- Use the **PostgreSQL implementation** for:
  - Production deployments
  - Multi-process applications
  - Large datasets
  - Performance-critical applications
  - Environments requiring high availability
