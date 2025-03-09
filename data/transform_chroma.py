import sqlite3
import json
import uuid
import chromadb
import sys
from datetime import datetime
from chromadb.utils import embedding_functions


# Chroma DB class with batched insertion methods
class ChromaDB:
    def __init__(self, path: str = "./chroma_db"):
        """Initialize Chroma DB with local persistence."""
        self.client = chromadb.PersistentClient(path=path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.zero_embedding = [0.0] * 384

        self.trajectories_collection = self.get_or_create_collection(
            "trajectories", "Trajectories of agent-environment interactions"
        )
        self.goal_collection = self.get_or_create_collection(
            "goal", "Goal of each trajectory"
        )
        self.category_collection = self.get_or_create_collection(
            "category", "Category of each trajectory"
        )
        self.plan_collection = self.get_or_create_collection(
            "plan", "Plan of each trajectory"
        )
        self.reflection_collection = self.get_or_create_collection(
            "reflection", "Reflection of each trajectory"
        )
        self.summary_collection = self.get_or_create_collection(
            "summary", "Summary of each trajectory"
        )

        self.states_collection = self.get_or_create_collection(
            "states", "States of each trajectory"
        )
        self.observation_collection = self.get_or_create_collection(
            "observation", "Observation of each state"
        )
        self.reasoning_collection = self.get_or_create_collection(
            "reasoning", "Reasoning of each state"
        )
        self.action_collection = self.get_or_create_collection(
            "actions", "Actions of each state"
        )

        self.traj_field_collections = {
            "goal": self.goal_collection,
            "category": self.category_collection,
            "plan": self.plan_collection,
            "reflection": self.reflection_collection,
            "summary": self.summary_collection,
        }
        self.state_field_collections = {
            "observation": self.observation_collection,
            "reasoning": self.reasoning_collection,
            "action": self.action_collection,
        }

    def get_or_create_collection(self, name: str, description: str):
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_func,
            # todo, performance tuning
            metadata={
                "hnsw:space": "cosine",
                # "hnsw:construction_ef": 200,
                # "hnsw:M": 48,
                "description": description,
                "created": str(datetime.now()),
            },
        )

    def _insert_trajectory(
        self,
        trajectory_id: str,
        environment_id: str,
        goal: str,
        category: str,
        observations,
        reasoning,
        actions,
        rewards,
        plan,
        reflection,
        summary,
    ) -> int:

        # Convert data to JSON strings for storing in metadata
        observations_str = json.dumps(observations)
        reasoning_str = json.dumps(reasoning)
        actions_str = json.dumps(actions)
        success = bool(rewards and rewards[-1])
        rewards_str = json.dumps(rewards)

        goal = goal if goal else ""
        category = category if category else ""
        plan = plan if plan else ""
        reflection = reflection if reflection else ""
        summary = summary if summary else ""

        fields = {
            "goal": goal,
            "category": category,
            "plan": plan if plan else "",
            "reflection": reflection if reflection else "",
            "summary": summary if summary else "",
        }

        # a "metadata" dictionary for the trajectory doc
        self.trajectories_collection.add(
            ids=[trajectory_id],
            documents=[""],
            embeddings=[self.zero_embedding],
            metadatas=[
                {
                    "environment_id": environment_id,
                    "goal": fields["goal"],
                    "category": fields["category"],
                    "observations": observations_str,
                    "reasoning": reasoning_str,
                    "actions": actions_str,
                    "rewards": rewards_str,
                    "success": success,
                    "plan": fields["plan"],
                    "reflection": fields["reflection"],
                    "summary": fields["summary"],
                }
            ],
        )

        for field_name, text_value in fields.items():
            self.traj_field_collections[field_name].add(
                ids=[trajectory_id],
                documents=[text_value],
                metadatas=[
                    {
                        "trajectory_id": trajectory_id,
                        "success": success,
                    }
                ],
            )

    def _insert_states(self, trajectory_id, observations, actions, reasoning):
        """
        Simulates inserting into 'states' by storing a new Chroma document
        for each state (with separate embeddings for observation, reasoning, action).
        """
        for i in range(len(observations) - 1):
            state = observations[i]
            next_state = observations[i + 1]
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i] if i < len(actions) else ""

            state_id = str(uuid.uuid4())
            self.states_collection.add(
                ids=[state_id],
                documents=[""],
                embeddings=[self.zero_embedding],
                metadatas=[
                    {
                        "trajectory_id": trajectory_id,
                        "position": i,
                        "observation": state,
                        "reasoning": reason_text,
                        "action": action_text,
                        "next_state": next_state,
                    }
                ],
            )

            fields = {
                "observation": state,
                "reasoning": reason_text,
                "action": action_text,
            }
            for field_name, text_value in fields.items():
                self.state_field_collections[field_name].add(
                    ids=[state_id],
                    documents=[text_value],
                    metadatas=[
                        {
                            "trajectory_id": trajectory_id,
                            "position": i,
                        }
                    ],
                )


# Transformation function
def transform_sqlite_to_chroma(sqlite_db_path: str, chroma_db_path: str):
    """Transform SQLite database into Chroma DB."""
    # Initialize Chroma DB
    traj_db = ChromaDB(chroma_db_path)

    # Connect to SQLite
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Fetch all trajectories
    cursor.execute("SELECT * FROM trajectories")
    rows = cursor.fetchall()

    for row in rows:
        # Unpack SQLite row (ignoring embedding fields not used in Chroma)
        (
            id,
            environment_id,
            goal,
            goal_embedding,
            category,
            category_embedding,
            observations_json,
            reasoning_json,
            actions_json,
            rewards_json,
            plan,
            plan_embedding,
            reflection,
            reflection_embedding,
            summary,
            summary_embedding,
        ) = row

        observations = json.loads(observations_json) if observations_json else []
        actions = json.loads(actions_json) if actions_json else []
        reasoning = json.loads(reasoning_json) if reasoning_json else []
        rewards = json.loads(rewards_json) if rewards_json else []

        # Insert trajectory into Chroma DB
        traj_db._insert_trajectory(
            trajectory_id=str(id),
            environment_id=environment_id,
            goal=goal,
            category=category,
            observations=observations,
            reasoning=reasoning,
            actions=actions,
            rewards=rewards,
            plan=plan,
            reflection=reflection,
            summary=summary,
        )

        # Insert states based on trajectory
        traj_db._insert_states(
            trajectory_id=str(id),
            observations=observations,
            actions=actions,
            reasoning=reasoning,
        )

    # Clean up and report
    conn.close()
    print(
        f"Transformed {len(rows)} trajectories and their states "
        f"into Chroma DB at '{chroma_db_path}'."
    )


def transform_chroma_to_sqlite(chroma_db_path: str, sqlite_db_path: str):
    """
    Transform the Chroma DB at `chroma_db_path` back into an SQLite database
    at `sqlite_db_path`. Re-creates or updates the `trajectories` table.
    Embeddings are stored as empty strings or placeholders for simplicity.
    """
    # -- 1) Initialize a connection to the existing Chroma DB --
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=chroma_db_path)
    trajectories_collection = client.get_collection(
        name="trajectories", embedding_function=embedding_func
    )

    results = trajectories_collection.get(include=["metadatas"], limit=None)

    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Ensure we have the same schema as in the original transform
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS trajectories (
        id TEXT PRIMARY KEY,
        environment_id TEXT,
        goal TEXT,
        goal_embedding TEXT,
        category TEXT,
        category_embedding TEXT,
        observations_json TEXT,
        reasoning_json TEXT,
        actions_json TEXT,
        rewards_json TEXT,
        plan TEXT,
        plan_embedding TEXT,
        reflection TEXT,
        reflection_embedding TEXT,
        summary TEXT,
        summary_embedding TEXT
    );
    """
    cursor.execute(create_table_sql)

    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]  # The metadata dict for this trajectory

        # Pull out the stored fields. They match what was inserted in `_insert_trajectory`:
        #   environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary
        environment_id = meta.get("environment_id", "")
        goal = meta.get("goal", "")
        category = meta.get("category", "")
        observations_str = meta.get("observations", "[]")  # JSON string
        reasoning_str = meta.get("reasoning", "")
        actions_str = meta.get("actions", "[]")
        rewards_str = meta.get("rewards", "[]")
        plan = meta.get("plan", "")
        reflection = meta.get("reflection", "")
        summary = meta.get("summary", "")

        # For simplicity in the example, store empty placeholders for embeddings
        goal_embedding = ""
        category_embedding = ""
        plan_embedding = ""
        reflection_embedding = ""
        summary_embedding = ""

        # Insert the row into `trajectories`. We do an upsert or just insert (depending on your needs).
        insert_sql = """
        INSERT OR REPLACE INTO trajectories (
            id,
            environment_id,
            goal,
            goal_embedding,
            category,
            category_embedding,
            observations_json,
            reasoning_json,
            actions_json,
            rewards_json,
            plan,
            plan_embedding,
            reflection,
            reflection_embedding,
            summary,
            summary_embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(
            insert_sql,
            (
                doc_id,
                environment_id,
                goal,
                goal_embedding,
                category,
                category_embedding,
                observations_str,
                reasoning_str,
                actions_str,
                rewards_str,
                plan,
                plan_embedding,
                reflection,
                reflection_embedding,
                summary,
                summary_embedding,
            ),
        )

    conn.commit()
    conn.close()

    print(
        f"Transformed {len(results['ids'])} trajectories from Chroma DB "
        f"('{chroma_db_path}') into SQLite at '{sqlite_db_path}'."
    )


# Run the transformation
if __name__ == "__main__":
    sqlite_db_path = sys.argv[1]
    chroma_db_path = sys.argv[2]
    # new_sqlite_db_path = sys.argv[3]
    transform_sqlite_to_chroma(sqlite_db_path, chroma_db_path)
    # transform_chroma_to_sqlite(chroma_db_path, new_sqlite_db_path)
