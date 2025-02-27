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

        self.trajectories_collection = self.get_or_create_collection("trajectories")
        self.states_collection = self.get_or_create_collection("states")

    def get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name,
            embedding_function=self.embedding_func,
            metadata={
                "hnsw:space": "cosine",
                "description": f"Collection for storing {name}",
                "created": str(datetime.now()),
            },
        )

    def _insert_trajectory(
        self,
        trajectory_id: str,
        environment_id: str,
        goal: str,
        category: str,
        observations: list,
        reasoning: list | None,
        actions: list,
        rewards: list,
        plan: str | None,
        reflection: str | None,
        summary: str | None,
    ) -> int:
        """
        Insert a trajectory into the trajectories_collection.
        Now done in fewer calls for speed.
        """
        # Convert data to JSON strings for metadata
        success = bool(rewards and rewards[-1] > 0)  # success if last reward > 0
        rewards_str = json.dumps(rewards)

        # 1) Add the main trajectory metadata as one document
        main_id = trajectory_id
        main_doc = ""
        main_metadata = {
            "environment_id": environment_id,
            "goal": goal,
            "category": category,
            "observations": observations,
            "reasoning": reasoning,
            "actions": actions,
            "rewards": rewards_str,
            "success": success,
            "plan": plan if plan else "",
            "reflection": reflection if reflection else "",
            "summary": summary if summary else "",
        }

        # 2) Add separate documents for key fields (goal, category, plan, reflection, summary)
        sub_ids = []
        sub_docs = []
        sub_metadatas = []

        fields = {
            "goal": goal or "",
            "category": category or "",
            "plan": plan or "",
            "reflection": reflection or "",
            "summary": summary or "",
        }

        for field_name, text in fields.items():
            doc_id = f"traj_{trajectory_id}_{field_name}"
            sub_ids.append(doc_id)
            sub_docs.append(text)
            sub_metadatas.append(
                {
                    "trajectory_id": trajectory_id,
                    "field_name": field_name,
                    "success": success,
                }
            )

        # Now do batched insertion in two calls:
        # a) main trajectory doc
        self.trajectories_collection.add(
            ids=[main_id],
            documents=[main_doc],
            metadatas=[main_metadata],
        )
        # b) sub-fields
        self.trajectories_collection.add(
            ids=sub_ids,
            documents=sub_docs,
            metadatas=sub_metadatas,
        )

        return 1  # number of trajectories inserted

    def _insert_states(
        self,
        trajectory_id: str,
        observations: list,
        actions: list,
        reasoning: list | None,
    ):
        """
        Insert states into the states_collection based on trajectory data.
        Uses batched inserts to reduce overhead.
        """
        if len(observations) < 2:
            return

        main_ids = []
        main_docs = []
        main_metadatas = []

        # We'll also store sub-fields (observation, reasoning, action) in a separate batch
        sub_ids = []
        sub_docs = []
        sub_metadatas = []

        for i in range(len(observations) - 1):
            state = observations[i]
            next_state = observations[i + 1]
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i] if i < len(actions) else ""

            state_id = str(uuid.uuid4())

            # Main "state" doc
            main_ids.append(state_id)
            main_docs.append("")  # or some summary text if desired
            main_metadatas.append(
                {
                    "trajectory_id": trajectory_id,
                    "position": i,
                    "observation": json.dumps(state),
                    "reasoning": reason_text,
                    "action": action_text,
                    "next_state": json.dumps(next_state),
                }
            )

            # Sub-fields
            fields = {
                "observation": json.dumps(state),
                "reasoning": reason_text,
                "action": action_text,
            }
            for field_name, text in fields.items():
                doc_id = f"state_{state_id}_{field_name}"
                sub_ids.append(doc_id)
                sub_docs.append(text)
                sub_metadatas.append(
                    {
                        "trajectory_id": trajectory_id,
                        "state_id": state_id,
                        "field_name": field_name,
                    }
                )

        # Now do two batched insert calls:
        # a) main states
        self.states_collection.add(
            ids=main_ids,
            documents=main_docs,
            metadatas=main_metadatas,
        )
        # b) sub-fields
        self.states_collection.add(
            ids=sub_ids,
            documents=sub_docs,
            metadatas=sub_metadatas,
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
        reasoning = json.loads(reasoning_json) if reasoning_json else None
        rewards = json.loads(rewards_json) if rewards_json else []

        # Insert trajectory into Chroma DB
        traj_db._insert_trajectory(
            trajectory_id=str(id),
            environment_id=environment_id,
            goal=goal,
            category=category,
            observations=observations_json,
            reasoning=reasoning_json,
            actions=actions_json,
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


# Run the transformation
if __name__ == "__main__":
    sqlite_db_path = sys.argv[1]
    chroma_db_path = sys.argv[2]
    transform_sqlite_to_chroma(sqlite_db_path, chroma_db_path)
