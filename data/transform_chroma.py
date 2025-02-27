import sqlite3
import json
import uuid
import chromadb
import sys


# Helper classes
class Observation:
    def __init__(self, structured):
        self.structured = structured


class Action:
    def __init__(self, text):
        self.text = text


# Chroma DB class with insertion methods
class ChromaDB:
    def __init__(self, path: str = "./chroma_db"):
        """Initialize Chroma DB with local persistence."""
        self.client = chromadb.PersistentClient(path=path)
        self.trajectories_collection = self.client.get_or_create_collection(
            "trajectories"
        )
        self.states_collection = self.client.get_or_create_collection("states")

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
        """Insert a trajectory into the trajectories_collection."""
        # Convert data to JSON strings for metadata
        observations_str = json.dumps([obs.structured for obs in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else ""
        actions_str = json.dumps([act.text for act in actions])
        success = bool(
            rewards and rewards[-1] > 0
        )  # Success if last reward is positive
        rewards_str = json.dumps(rewards)

        # Add trajectory metadata
        self.trajectories_collection.add(
            ids=[trajectory_id],
            documents=[""],
            metadatas=[
                {
                    "environment_id": environment_id,
                    "goal": goal,
                    "category": category,
                    "observations": observations_str,
                    "reasoning": reasoning_str,
                    "actions": actions_str,
                    "rewards": rewards_str,
                    "success": success,
                    "plan": plan if plan else "",
                    "reflection": reflection if reflection else "",
                    "summary": summary if summary else "",
                }
            ],
        )

        # Add separate documents for key fields
        fields = {
            "goal": goal,
            "category": category,
            "plan": plan,
            "reflection": reflection,
            "summary": summary,
        }
        for field_name, text in fields.items():
            if text is None:
                text = ""
            doc_id = f"traj_{trajectory_id}_{field_name}"
            self.trajectories_collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[
                    {
                        "trajectory_id": trajectory_id,
                        "field_name": field_name,
                        "success": success,
                    }
                ],
            )
        return 1  # Number of trajectories inserted

    def _insert_states(
        self,
        trajectory_id: str,
        observations: list,
        actions: list,
        reasoning: list | None,
    ):
        """Insert states into the states_collection based on trajectory data."""
        for i in range(len(observations) - 1):  # Create states as transitions
            state = observations[i].structured
            next_state = observations[i + 1].structured
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i].text if i < len(actions) else ""

            state_id = str(uuid.uuid4())
            self.states_collection.add(
                ids=[state_id],
                documents=[""],
                metadatas=[
                    {
                        "trajectory_id": trajectory_id,
                        "position": i,
                        "observation": json.dumps(state),
                        "reasoning": reason_text,
                        "action": action_text,
                        "next_state": json.dumps(next_state),
                    }
                ],
            )

            # Add separate documents for observation, reasoning, and action
            fields = {
                "observation": json.dumps(state),
                "reasoning": reason_text,
                "action": action_text,
            }
            for field_name, text in fields.items():
                doc_id = f"state_{state_id}_{field_name}"
                self.states_collection.add(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[
                        {
                            "trajectory_id": trajectory_id,
                            "state_id": state_id,
                            "field_name": field_name,
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

        # Parse JSON fields with NULL handling
        observations_structured = (
            json.loads(observations_json) if observations_json else []
        )
        observations = [Observation(struct) for struct in observations_structured]

        actions_texts = json.loads(actions_json) if actions_json else []
        actions = [Action(text) for text in actions_texts]

        rewards = json.loads(rewards_json) if rewards_json else []

        reasoning = json.loads(reasoning_json) if reasoning_json else None

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
        f"Transformed {len(rows)} trajectories and their states into Chroma DB at './chroma_db'."
    )


# Run the transformation
if __name__ == "__main__":
    sqlite_db_path = sys.argv[1]
    chroma_db_path = sys.argv[2]
    transform_sqlite_to_chroma(sqlite_db_path, chroma_db_path)
