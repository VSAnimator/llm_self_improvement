import uuid
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Union, Optional, Any

# Chroma imports
import chromadb
from chromadb.utils import embedding_functions


class LearningDB:

    def __init__(
        self,
        db_path: str = "data/learning_db",
        config_type: str = "lite",
        port: int = 8008,
    ):
        """
        Initialize a Chroma-based "learning database".
        - embedding_dim is mainly informational here, as SentenceTransformer determines actual dimension.
        - persist_directory is where Chroma will store its local data (DuckDB + parquet).
        """
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # change the dimension acoording to the model
        self.zero_embedding = [0.0] * 384

        if config_type == "lite":
            self.chroma_client = chromadb.PersistentClient(path=db_path)
        elif config_type == "server":
            self.chroma_client = chromadb.HttpClient(host="localhost", port=port)
        else:
            raise ValueError(f"Unknown config_type: {config_type}")

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
        self.rule_field_collections = {}

    def get_or_create_collection(self, name: str, description: str):
        return self.chroma_client.get_or_create_collection(
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
            state = observations[i].structured
            next_state = observations[i + 1].structured
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i].text if i < len(actions) else ""

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

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        collections,
        field_names: List[str],
        query_texts: List[str],
        n_results: int,
        filter: Optional[Dict[str, Any]] = None,
    ):
        candidates = {}
        field_worst_distance = {}

        for fname, qtext in zip(field_names, query_texts):
            results = collections[fname].query(
                query_texts=[qtext],
                n_results=n_results * min(2, len(field_names)),  # expand search range
                where=({**filter} if filter else {}),
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
        key_types = [key_type] if isinstance(key_type, str) else key_type
        keys = [key] if isinstance(key, str) else key

        is_trajectory = any(kt in self.traj_field_collections for kt in key_types)
        is_state = any(kt in self.state_field_collections for kt in key_types)
        is_rule = any(kt in self.rule_field_collections for kt in key_types)
        assert (
            sum([is_trajectory, is_state, is_rule]) == 1
        ), "Invalid or mixed key types."

        if is_trajectory:
            collections = self.traj_field_collections
        elif is_state:
            collections = self.state_field_collections
        else:  # is_rule
            raise NotImplementedError("Rule-based search not yet implemented.")

        return self._compute_top_k_nearest_neighbors_by_avg_distance(
            collections,
            key_types,
            keys,
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
                {"$and": [{"environment_id": env_id}, outcome_filter]}
                if outcome_filter
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

        trajectory_key_types = [
            kt for kt in key_type if kt in self.traj_field_collections
        ]
        state_key_types = [kt for kt in key_type if kt in self.state_field_collections]

        chosen_t_ids = []
        critical_state_positions = {}
        if trajectory_key_types:
            # Filter key+type to only trajectory keys
            key_filtered = [
                v for kt, v in zip(key_type, key) if kt in trajectory_key_types
            ]
            t_ids, t_distances = self._get_top_k_by_keys(
                trajectory_key_types,
                key_filtered,
                k * 2 if state_key_types else k,
                filter=outcome_filter,
            )

            if state_key_types:
                state_only_key_filtered = [
                    v for kt, v in zip(key_type, key) if kt in state_key_types
                ]
                for i, t_id in enumerate(t_ids):
                    t_states = self.states_collection.get(where={"trajectory_id": t_id})
                    state_field_distances = np.zeros(len(t_states["ids"]))
                    for s_field, state_key in zip(
                        state_key_types, state_only_key_filtered
                    ):
                        field_embeds = self.state_field_collections[s_field].get(
                            ids=t_states["ids"], include=["embeddings"]
                        )["embeddings"]
                        query_vec = self.embedding_func(state_key)[0]
                        state_field_distances += [
                            1
                            - np.dot(
                                embd,
                                query_vec,
                            )
                            for embd in field_embeds
                        ]
                    state_field_distances /= len(state_key_types)

                    min_index = np.argmin(state_field_distances)
                    critical_state_positions[t_id] = t_states["metadatas"][min_index][
                        "position"
                    ]
                    t_distances[i] += state_field_distances[min_index]

            sort_indices = np.argsort(t_distances)
            chosen_t_ids = [t_ids[i] for i in sort_indices[:k]]

        else:
            assert (
                state_key_types
            ), "At least one key must be a trajectory or state key."
            s_ids, s_distances = self._get_top_k_by_keys(state_key_types, key, k * 2)
            for s_id in s_ids:
                state = self.states_collection.get(ids=[s_id])["metadatas"][0]
                trajectory_id = state["trajectory_id"]
                if trajectory_id not in chosen_t_ids:
                    chosen_t_ids.append(trajectory_id)
                    critical_state_positions[trajectory_id] = state["position"]
                    if len(chosen_t_ids) >= k:
                        break

        if len(chosen_t_ids) == 0:
            return [], []

        similar_entries = self.trajectories_collection.get(ids=chosen_t_ids)[
            "metadatas"
        ]

        for i, tid in enumerate(chosen_t_ids):
            window_start, window_end = 0, None
            if critical_state_positions:
                window_start = max(0, critical_state_positions[tid] - window)
                window_end = critical_state_positions[tid] + window + 1
            similar_entries[i] = self._format_conversion(
                similar_entries[i], window_start, window_end
            )

        success_entries = [m for m in similar_entries if m["success"]]
        failure_entries = [m for m in similar_entries if not m["success"]]
        return success_entries, failure_entries
