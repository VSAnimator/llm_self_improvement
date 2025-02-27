import uuid
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Union, Optional, Any

# Chroma imports
import chromadb
from chromadb.utils import embedding_functions

TRAJECTORY_FIELDS = ["goal", "category", "plan", "reflection", "summary"]
STATE_FIELDS = ["observation", "reasoning", "action"]
RULE_FIELDS = ["name", "content", "context"]


class LearningDB:

    def __init__(self, db_path: str = "data/learning_db", config_type: str = "lite"):
        """
        Initialize a Chroma-based "learning database".
        - embedding_dim is mainly informational here, as SentenceTransformer determines actual dimension.
        - persist_directory is where Chroma will store its local data (DuckDB + parquet).
        """
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        if config_type == "lite":
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        elif config_type == "server":
            self.chroma_client = chromadb.HttpClient(host="localhost", port=8008)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")

        self.trajectories_collection = self.chroma_client.get_or_create_collection(
            name="trajectories",
            embedding_function=self.embedding_func,
            metadata={
                "hnsw:space": "cosine",
                "description": "Trajectories of agent-environment interactions",
                "created": str(datetime.now()),
            },
        )
        self.states_collection = self.chroma_client.get_or_create_collection(
            name="states",
            embedding_function=self.embedding_func,
            metadata={
                "hnsw:space": "cosine",
                "description": "States of each trajectory",
                "created": str(datetime.now()),
            },
        )
        self.rules_collection = self.chroma_client.get_or_create_collection(
            name="rules",
            embedding_function=self.embedding_func,
            metadata={
                "hnsw:space": "cosine",
                "description": "Rules for agent behavior",
                "created": str(datetime.now()),
            },
        )

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
        summary: Optional[str],
    ):
        """
        Insert an episode into our "trajectories" and "states" Chroma collections.
        Replaces the old store_episode from Postgres.
        """
        trajectory_id = str(uuid.uuid4())  # random unique ID
        self._insert_trajectory(
            trajectory_id,
            environment_id,
            goal,
            category,
            observations,
            reasoning,
            actions,
            rewards,
            plan,
            reflection,
            summary,
        )
        self._insert_states(trajectory_id, observations, actions, reasoning)

    def _insert_trajectory(
        self,
        trajectory_id: str,
        environment_id: str,
        goal: str,
        category: str,
        observations: List[Any],
        reasoning: Optional[List[str]],
        actions: List[Any],
        rewards: List[float],
        plan: Optional[str],
        reflection: Optional[str],
        summary: Optional[str],
    ) -> int:

        # Convert data to JSON strings for storing in metadata
        observations_str = json.dumps([obs.structured for obs in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else ""
        actions_str = json.dumps([act.text for act in actions])
        success = bool(rewards and rewards[-1])
        rewards_str = json.dumps(rewards)

        # a "metadata" dictionary for the trajectory doc
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

        fields = {
            "goal": goal,
            "category": category,
            "plan": plan,
            "reflection": reflection,
            "summary": summary,
        }

        for field_name, text_value in fields.items():
            if text_value is None:
                text_value = ""
            doc_id = f"traj_{trajectory_id}_{field_name}"
            self.trajectories_collection.add(
                ids=[doc_id],
                documents=[text_value],
                metadatas=[
                    {
                        "trajectory_id": trajectory_id,
                        "field_name": field_name,
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
                doc_id = f"state_{state_id}_{field_name}"
                self.states_collection.add(
                    ids=[doc_id],
                    documents=[text_value],
                    metadatas=[
                        {
                            "trajectory_id": trajectory_id,
                            "state_id": state_id,
                            "field_name": field_name,
                        }
                    ],
                )

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        collection,
        field_names: List[str],
        query_texts: List[str],
        n_results: int,
        filter: Optional[Dict[str, Any]] = None,
    ):
        candidates = {}
        field_worst_distance = {}

        for fname, qtext in zip(field_names, query_texts):
            results = collection.query(
                query_texts=[qtext],
                n_results=n_results * len(field_names),  # expand search range
                where=(
                    {"$and": [{"field_name": fname}, filter]}
                    if filter
                    else {"field_name": fname}
                ),
                include=["distances", "metadatas"],
            )

            # Safeguard if no results found
            if not results["ids"][0]:
                continue

            tids = [r["trajectory_id"] for r in results["metadatas"][0]]
            distances = results["distances"][0]

            for t, dist in zip(tids, distances):
                if t not in candidates:
                    candidates[t] = {}
                candidates[t][fname] = dist
            field_worst_distance[fname] = distances[-1]

        for k, v in candidates.items():
            candidates[k]["avg_distance"] = np.mean(
                [v.get(fname, field_worst_distance[fname]) for fname in field_names]
            )

        sorted_candidates = sorted(
            candidates.items(), key=lambda x: x[1]["avg_distance"]
        )
        top_k_ids = [sc[0] for sc in sorted_candidates[:n_results]]
        top_k_distances = [
            sc[1]["avg_distance"] for sc in sorted_candidates[:n_results]
        ]
        return top_k_ids, top_k_distances

    def _get_top_k_by_keys(
        self,
        key_type: Union[str, List[str]],
        key: Union[str, List[str]],
        n_results: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ):
        key_type = [key_type] if isinstance(key_type, str) else key_type
        key = [key] if isinstance(key, str) else key

        is_trajectory = any(kt in TRAJECTORY_FIELDS for kt in key_type)
        is_state = any(kt in STATE_FIELDS for kt in key_type)
        is_rule = any(kt in RULE_FIELDS for kt in key_type)
        assert (
            sum([is_trajectory, is_state, is_rule]) == 1
        ), "Invalid or mixed key types."

        if is_trajectory:
            collection = self.trajectories_collection
        elif is_state:
            collection = self.states_collection
        else:  # is_rule
            collection = self.rules_collection

        return self._compute_top_k_nearest_neighbors_by_avg_distance(
            collection,
            key_type,
            key,
            n_results,
            filter,
        )

    def _format_conversion(self, entry, window_start=0, window_end=None):
        entry["observation"] = json.loads(entry["observations"])[
            window_start:window_end
        ]
        entry["reasoning"] = (
            json.loads(entry["reasoning"])
            if entry["reasoning"][window_start:window_end]
            else None
        )
        entry["action"] = json.loads(entry["actions"])[window_start:window_end]
        entry["rewards"] = json.loads(entry["rewards"])[window_start:window_end]
        return entry

    def _exact_match(self, env_id: str, n_results: int, outcome_filter: Dict):
        all_trajectories = self.trajectories_collection.get(
            where=(
                {"$and": [{"environment_id": env_id}, filter]}
                if filter
                else {"environment_id": env_id}
            ),
        )["metadatas"]

        sorted_trajectories = sorted(
            all_trajectories,
            key=lambda x: len(json.loads(x["observations"])),
        )[:n_results]

        success_entries = [
            self._format_conversion(m) for m in sorted_trajectories if m["success"]
        ]
        failure_entries = [
            self._format_conversion(m) for m in sorted_trajectories if not m["success"]
        ]
        return success_entries, failure_entries

    def get_similar_entries(
        self,
        key_type: Union[str, List[str]],
        key: Union[str, List[str]],
        k: int = 5,
        outcome: str = None,
        window: int = 1,
    ) -> List[Dict]:
        key_type = [key_type] if isinstance(key_type, str) else key_type
        key = [key] if isinstance(key, str) else key
        outcome_filter = (
            {"success": outcome in ["winning", "success"]} if outcome else {}
        )

        if key_type == ["environment_id"]:
            return self._exact_match(key[0], k, outcome_filter)

        trajectory_key_types = [kt for kt in key_type if kt in TRAJECTORY_FIELDS]
        state_key_types = [kt for kt in key_type if kt in STATE_FIELDS]

        chosen_t_ids = []
        critical_state_positions = []
        if trajectory_key_types:
            # Filter key+type to only trajectory keys
            key_filtered = [
                v for kt, v in zip(key_type, key) if kt in trajectory_key_types
            ]
            t_ids, t_distances = self._get_top_k_by_keys(
                trajectory_key_types,
                key_filtered,
                k * 3,
                filter=outcome_filter,
            )

            critical_state_ids = {}
            if state_key_types:
                state_only_key_filtered = [
                    v for kt, v in zip(key_type, key) if kt in state_key_types
                ]
                for i, t_id in enumerate(t_ids):
                    s_ids, s_distances = self._get_top_k_by_keys(
                        state_key_types,
                        state_only_key_filtered,
                        n_results=1,
                        filter={"trajectory_id": t_id},
                    )
                    if not s_ids:
                        t_distances[i] += 1.0
                        # missing critical id
                    else:
                        t_distances[i] += min(s_distances)
                        critical_state_ids[t_id] = s_ids[0]

            sort_indices = np.argsort(t_distances)
            chosen_t_ids = [t_ids[i] for i in sort_indices[:k]]

            if critical_state_ids:
                critical_state_ids = {
                    k: v for k, v in critical_state_ids.items() if k in chosen_t_ids
                }
                critical_states = self.states_collection.get(
                    ids=list(critical_state_ids.keys()), include=["metadatas"]
                )["metadatas"]
                critical_state_positions = [cs["position"] for cs in critical_states]

        else:
            assert (
                state_key_types
            ), "At least one key must be a trajectory or state key."
            s_ids, s_distances = self._get_top_k_by_keys(state_key_types, key, k)
            for s_id in s_ids:
                state = self.states_collection.get(ids=[s_id])["metadatas"][0]
                trajectory_id = state["trajectory_id"]
                if trajectory_id not in chosen_t_ids:
                    chosen_t_ids.append(trajectory_id)
                    critical_state_positions.append(state["position"])
                    if len(chosen_t_ids) >= k:
                        break

        if len(chosen_t_ids) == 0:
            return [], []

        similar_entries = self.trajectories_collection.get(ids=chosen_t_ids)[
            "metadatas"
        ]

        for i in range(len(similar_entries)):
            window_start, window_end = 0, None
            if critical_state_positions:
                window_start = max(0, critical_state_positions[i] - window)
                window_end = critical_state_positions[i] + window + 1
            similar_entries[i] = self._format_conversion(
                similar_entries[i], window_start, window_end
            )

        success_entries = [m for m in similar_entries if m["success"]]
        failure_entries = [m for m in similar_entries if not m["success"]]
        return success_entries, failure_entries
