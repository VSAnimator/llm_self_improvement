import uuid
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any

from pymilvus import Milvus, MilvusClient, DataType, FieldSchema, CollectionSchema
from sentence_transformers import SentenceTransformer


class LearningDB:
    def __init__(
        self,
        db_path: str = "data/learning_milvus.db",
    ):
        """
        Initialize a Milvus-based "learning database" with consolidated collections for trajectories and states.

        Args:
            port (int): Port where Milvus server is running (default: 19530).
        """
        # Initialize Milvus client and embedding model
        self.db_file = db_path
        self.milvus_client = MilvusClient(db_path)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_dim = 384
        self.zero_embedding = [
            0.0
        ] * self.embed_dim  # Embedding dimension for all-MiniLM-L6-v2

        # Create or get consolidated collections
        self.trajectories_collection = self._get_or_create_trajectory_collection()
        self.states_collection = self._get_or_create_states_collection()

    def _get_or_create_trajectory_collection(self) -> str:
        collection_name = "trajectories"
        self.trajectory_fields = [
            "goal",
            "category",
            "plan",
            "reflection",
            "summary",
        ]
        if not self.milvus_client.has_collection(collection_name):
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36
                ),
                FieldSchema(
                    name="environment_id", dtype=DataType.VARCHAR, max_length=255
                ),
                FieldSchema(
                    name="observations", dtype=DataType.VARCHAR, max_length=65535
                ),
                FieldSchema(name="reasoning", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="actions", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="rewards", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="success", dtype=DataType.BOOL),
            ]
            for field in self.trajectory_fields:
                fields.append(
                    FieldSchema(name=field, dtype=DataType.VARCHAR, max_length=65535)
                )
                fields.append(
                    FieldSchema(
                        name=f"{field}_vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embed_dim,
                    )
                )
            schema = CollectionSchema(
                fields=fields,
                description="Consolidated trajectories with multiple vector fields",
            )
            self.milvus_client.create_collection(collection_name, schema)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT",
            }
            for field in self.trajectory_fields:
                self.milvus_client.create_index(
                    collection_name, f"{field}_vector", index_params
                )
        return collection_name

    def _get_or_create_states_collection(self) -> str:
        collection_name = "states"
        self.state_fields = ["observation", "reasoning", "action"]
        if not self.milvus_client.has_collection(collection_name):
            fields = [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36
                ),
                FieldSchema(
                    name="trajectory_id", dtype=DataType.VARCHAR, max_length=36
                ),
                FieldSchema(name="position", dtype=DataType.INT32),
                FieldSchema(
                    name="next_state", dtype=DataType.VARCHAR, max_length=65535
                ),
            ]
            for field in self.state_fields:
                fields.append(
                    FieldSchema(name=field, dtype=DataType.VARCHAR, max_length=65535)
                )
                fields.append(
                    FieldSchema(
                        name=f"{field}_vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.embed_dim,
                    )
                )
            schema = CollectionSchema(
                fields=fields,
                description="Consolidated states with multiple vector fields",
            )
            self.milvus_client.create_collection(collection_name, schema)
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT",
            }
            for field in self.state_fields:
                self.milvus_client.create_index(collection_name, field, index_params)
        return collection_name

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
        Insert an episode into the trajectories and states collections.

        Args:
            environment_id (str): Unique identifier for the environment.
            goal (str): Goal of the trajectory.
            category (str): Category of the trajectory.
            observations (List[Any]): List of observation objects.
            reasoning (Optional[List[str]]): List of reasoning texts.
            actions (List[Any]): List of action objects.
            rewards (List[float]): List of rewards.
            plan (Optional[str]): Plan text.
            reflection (Optional[str]): Reflection text.
            summary (Optional[str]): Summary text.
        """

        trajectory_id = str(uuid.uuid4())
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

    def _encode_text(self, text: str) -> List[float]:
        if text:
            return self.embedding_model.encode([text])[0].tolist()
        return self.zero_embedding

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
    ):
        """Insert trajectory data into the trajectories collection."""
        # Convert lists to JSON strings
        observations_str = json.dumps([obs.structured for obs in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else ""
        actions_str = json.dumps([act.text for act in actions])
        rewards_str = json.dumps(rewards)
        success = bool(rewards and rewards[-1])

        # Compute embeddings for trajectory fields
        fields = {
            "goal": goal,
            "category": category,
            "plan": plan if plan else "",
            "reflection": reflection if reflection else "",
            "summary": summary if summary else "",
        }
        embeddings = {f"{k}_vector": self._encode_text(v) for k, v in fields.items()}

        # Insert into trajectories collection
        entity = {
            "id": trajectory_id,
            "environment_id": environment_id,
            "observations": observations_str,
            "reasoning": reasoning_str,
            "actions": actions_str,
            "rewards": rewards_str,
            "success": success,
            **fields,
            **embeddings,
        }
        self.milvus_client.insert(self.trajectories_collection, [entity])

    def _insert_states(
        self,
        trajectory_id: str,
        observations: List[Any],
        actions: List[Any],
        reasoning: Optional[List[str]],
    ):
        """Insert state data into the states collection."""
        for i in range(len(observations) - 1):
            state_id = str(uuid.uuid4())
            observation = observations[i].structured
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i].text if i < len(actions) else ""
            next_state = observations[i + 1].structured

            # Compute embeddings for state fields
            fields = {
                "observation": observation,
                "reasoning": reason_text,
                "action": action_text,
            }
            embeddings = {
                f"{k}_vector": self._encode_text(v) for k, v in fields.items()
            }

            # Insert into states collection
            entity = {
                "id": state_id,
                "trajectory_id": trajectory_id,
                "position": i,
                "next_state": next_state,
                **fields,
                **embeddings,
            }
            self.milvus_client.insert(self.states_collection, [entity])

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        collection_name: str,
        field_names: List[str],
        query_texts: List[str],
        n_results: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[float]]:
        """Compute top-k nearest trajectories based on average distance across fields."""
        candidates = {}
        field_worst_distance = {}

        for fname, qtext in zip(field_names, query_texts):
            query_embedding = self._encode_text(qtext)
            filter_expr = (
                " and ".join([f"{k} == {v}" for k, v in filter.items()])
                if filter
                else None
            )

            results = self.milvus_client.search(
                collection_name=collection_name,
                data=[query_embedding],
                anns_field=f"{fname}_vector",
                filter=filter_expr,
                limit=n_results * min(2, len(field_names)),
                output_fields=["id"],
            )

            if not results[0]:
                continue

            for hit in results[0]:
                tid = hit.entity["id"]
                dist = hit.distance
                if tid not in candidates:
                    candidates[tid] = {}
                candidates[tid][fname] = dist
            field_worst_distance[fname] = results[0][-1].distance if results[0] else 1.0

        for k, v in candidates.items():
            candidates[k]["avg_distance"] = np.mean(
                [
                    v.get(fname, field_worst_distance.get(fname, 1.0))
                    for fname in field_names
                ]
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
    ) -> tuple[List[str], List[float]]:
        """Get top-k trajectory IDs based on key type and query text."""
        key_types = [key_type] if isinstance(key_type, str) else key_type
        keys = [key] if isinstance(key, str) else key

        is_trajectory = all(kt in self.trajectory_fields for kt in key_types)
        is_state = all(kt in self.state_fields for kt in key_types)
        assert is_trajectory or is_state, "Invalid key types."

        if is_trajectory:
            collection_name = self.trajectories_collection
        elif is_state:
            collection_name = self.states_collection

        return self._compute_top_k_nearest_neighbors_by_avg_distance(
            collection_name, key_types, keys, n_results, filter
        )

    def _format_conversion(
        self, entry: Dict, window_start: int = 0, window_end: Optional[int] = None
    ) -> Dict:
        """Format a trajectory entry for return, applying windowing."""
        entry["observation"] = json.loads(entry["observations"])[
            window_start:window_end
        ]
        entry["reasoning"] = (
            json.loads(entry["reasoning"])[window_start:window_end]
            if entry["reasoning"]
            else None
        )
        entry["action"] = json.loads(entry["actions"])[window_start:window_end]
        entry["rewards"] = json.loads(entry["rewards"])[window_start:window_end]
        return entry

    def _exact_match(
        self, env_id: str, n_results: int, outcome_filter: Dict
    ) -> tuple[List[Dict], List[Dict]]:
        """Retrieve trajectories matching an exact environment ID."""
        expr = f"environment_id == '{env_id}'"
        if outcome_filter:
            expr += f" and success == {outcome_filter['success']}"

        results = self.milvus_client.query(
            collection_name=self.trajectories_collection,
            filter=expr,
            output_fields=["*"],
        )
        all_trajectories = [dict(entity) for entity in results]
        sorted_trajectories = sorted(
            all_trajectories, key=lambda x: len(json.loads(x["observations"]))
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
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Retrieve similar trajectory entries based on key type and query.

        Args:
            key_type (Union[str, List[str]]): Type(s) of key (e.g., "goal", "observation").
            key (Union[str, List[str]]): Query text(s) to search for.
            k (int): Number of results to return.
            outcome (str): Filter by "winning" or "success" if specified.
            window (int): Window size around critical state for state-based searches.

        Returns:
            Tuple[List[Dict], List[Dict]]: Success entries and failure entries.
        """
        key_types = [key_type] if isinstance(key_type, str) else key_type
        keys = [key] if isinstance(key, str) else key
        outcome_filter = (
            {"success": outcome in ["winning", "success"]} if outcome else {}
        )

        if key_types == ["environment_id"]:
            return self._exact_match(keys[0], k, outcome_filter)

        trajectory_key_types = [kt for kt in key_types if kt in self.trajectory_fields]
        state_key_types = [kt for kt in key_types if kt in self.state_fields]

        chosen_t_ids = []
        critical_state_positions = {}

        if trajectory_key_types:
            key_filtered = [
                v for kt, v in zip(key_types, keys) if kt in trajectory_key_types
            ]
            t_ids, t_distances = self._get_top_k_by_keys(
                trajectory_key_types,
                key_filtered,
                k * 2 if state_key_types else k,
                outcome_filter,
            )

            if state_key_types:
                state_only_key_filtered = [
                    v for kt, v in zip(key_types, keys) if kt in state_key_types
                ]
                for i, t_id in enumerate(t_ids):
                    t_states = self.milvus_client.query(
                        collection_name=self.states_collection,
                        filter=f"trajectory_id == '{t_id}'",
                        output_fields=["id", "position"],
                    )
                    state_ids = [state["id"] for state in t_states]
                    state_field_distances = np.zeros(len(state_ids))

                    for s_field, state_key in zip(
                        state_key_types, state_only_key_filtered
                    ):
                        query_vec = self._encode_text(state_key)
                        field_embeds = self.milvus_client.query(
                            collection_name=self.states_collection,
                            filter=f"id in {state_ids}",
                            output_fields=["id", f"{s_field}_vector"],
                        )
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
                    critical_state_positions[t_id] = t_states[min_index]["position"]
                    t_distances[i] += state_field_distances[min_index]

            sort_indices = np.argsort(t_distances)
            chosen_t_ids = [t_ids[i] for i in sort_indices[:k]]
        else:
            s_ids, s_distances = self._get_top_k_by_keys(state_key_types, keys, k * 2)
            for s_id in s_ids:
                state = self.milvus_client.query(
                    collection_name=self.states_collection,
                    filter=f"id == '{s_id}'",
                    output_fields=["trajectory_id", "position"],
                )[0]
                trajectory_id = state["trajectory_id"]
                if trajectory_id not in chosen_t_ids:
                    chosen_t_ids.append(trajectory_id)
                    critical_state_positions[trajectory_id] = state["position"]
                    if len(chosen_t_ids) >= k:
                        break

        if not chosen_t_ids:
            return [], []

        similar_entries = self.milvus_client.query(
            collection_name=self.trajectories_collection,
            filter=f"id in {chosen_t_ids}",
            output_fields=["*"],
        )
        similar_entries = [dict(entity) for entity in similar_entries]

        for i, tid in enumerate(chosen_t_ids):
            window_start = max(0, critical_state_positions.get(tid, 0) - window)
            window_end = (
                critical_state_positions.get(tid, 0) + window + 1
                if critical_state_positions
                else None
            )
            similar_entries[i] = self._format_conversion(
                similar_entries[i], window_start, window_end
            )

        success_entries = [m for m in similar_entries if m["success"]]
        failure_entries = [m for m in similar_entries if not m["success"]]
        return success_entries, failure_entries
