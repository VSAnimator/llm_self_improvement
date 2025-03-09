import uuid
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any
import psycopg2
from sentence_transformers import SentenceTransformer


class LearningDB:
    def __init__(
        self,
        db_path: str = "~/.learning_db",
        db_name: str = "learning_db",
    ):
        self.conn = psycopg2.connect(host=db_path, dbname=db_name)
        self.cur = self.conn.cursor()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embed_dim = 384
        self.zero_embedding = [
            0.0
        ] * self.embed_dim  # Embedding dimension for all-MiniLM-L6-v2

        # Define field names for trajectories and states
        self.trajectory_fields = ["goal", "category", "plan", "reflection", "summary"]
        self.state_fields = ["observation", "reasoning", "action"]

        # Create tables and enable pgvector extension if not exists
        self._create_tables_if_not_exist()

    def _create_tables_if_not_exist(self):
        """Create trajectories and states tables with vector columns if they don't exist."""
        # Enable pgvector extension
        self.cur.execute(
            "SELECT EXISTS (SELECT FROM pg_extension WHERE extname = 'vector');"
        )
        if not self.cur.fetchone()[0]:
            self.cur.execute("CREATE EXTENSION vector;")
            self.conn.commit()

        # Create trajectories table if not exists
        self.cur.execute(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'trajectories');"
        )
        if not self.cur.fetchone()[0]:
            self.cur.execute(
                """
                CREATE TABLE trajectories (
                    id VARCHAR(36) PRIMARY KEY,
                    environment_id VARCHAR(255),
                    observations TEXT,
                    reasoning TEXT,
                    actions TEXT,
                    rewards TEXT,
                    success BOOLEAN,
                    goal TEXT,
                    goal_vector vector(384),
                    category TEXT,
                    category_vector vector(384),
                    plan TEXT,
                    plan_vector vector(384),
                    reflection TEXT,
                    reflection_vector vector(384),
                    summary TEXT,
                    summary_vector vector(384)
                );
            """
            )
            self.conn.commit()

        # Create states table if not exists
        self.cur.execute(
            "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'states');"
        )
        if not self.cur.fetchone()[0]:
            self.cur.execute(
                """
                CREATE TABLE states (
                    id VARCHAR(36) PRIMARY KEY,
                    trajectory_id VARCHAR(36),
                    position INTEGER,
                    next_state TEXT,
                    observation TEXT,
                    observation_vector vector(384),
                    reasoning TEXT,
                    reasoning_vector vector(384),
                    action TEXT,
                    action_vector vector(384),
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );
            """
            )
            self.conn.commit()

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
        """Generate a vector embedding for the given text."""
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
        """Insert trajectory data into the trajectories table."""
        observations_str = json.dumps([obs.structured for obs in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else ""
        actions_str = json.dumps([act.text for act in actions])
        rewards_str = json.dumps(rewards)
        success = bool(
            rewards and rewards[-1]
        )  # Assuming last reward indicates success

        fields = {
            "goal": goal,
            "category": category,
            "plan": plan if plan else "",
            "reflection": reflection if reflection else "",
            "summary": summary if summary else "",
        }
        embeddings = {f"{k}_vector": self._encode_text(v) for k, v in fields.items()}

        self.cur.execute(
            """
            INSERT INTO trajectories (
                id, environment_id, observations, reasoning, actions, rewards, success,
                goal, goal_vector, category, category_vector, plan, plan_vector,
                reflection, reflection_vector, summary, summary_vector
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                trajectory_id,
                environment_id,
                observations_str,
                reasoning_str,
                actions_str,
                rewards_str,
                success,
                fields["goal"],
                embeddings["goal_vector"],
                fields["category"],
                embeddings["category_vector"],
                fields["plan"],
                embeddings["plan_vector"],
                fields["reflection"],
                embeddings["reflection_vector"],
                fields["summary"],
                embeddings["summary_vector"],
            ),
        )
        self.conn.commit()

    def _insert_states(
        self,
        trajectory_id: str,
        observations: List[Any],
        actions: List[Any],
        reasoning: Optional[List[str]],
    ):
        """Insert state data into the states table."""
        for i in range(len(observations) - 1):
            state_id = str(uuid.uuid4())
            observation = observations[i].structured
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else ""
            action_text = actions[i].text if i < len(actions) else ""
            next_state = observations[i + 1].structured

            fields = {
                "observation": observation,
                "reasoning": reason_text,
                "action": action_text,
            }
            embeddings = {
                f"{k}_vector": self._encode_text(v) for k, v in fields.items()
            }

            self.cur.execute(
                """
                INSERT INTO states (
                    id, trajectory_id, position, next_state,
                    observation, observation_vector,
                    reasoning, reasoning_vector,
                    action, action_vector
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    state_id,
                    trajectory_id,
                    i,
                    next_state,
                    fields["observation"],
                    embeddings["observation_vector"],
                    fields["reasoning"],
                    embeddings["reasoning_vector"],
                    fields["action"],
                    embeddings["action_vector"],
                ),
            )
        self.conn.commit()

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        table_name: str,
        field_names: List[str],
        query_texts: List[str],
        n_results: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[float]]:
        """
        Compute top-k nearest entries based on average cosine distance across fields.

        Args:
            table_name (str): Name of the table ('trajectories' or 'states').
            field_names (List[str]): Fields to compute similarity on.
            query_texts (List[str]): Query texts to generate embeddings.
            n_results (int): Number of results to return.
            filter (Optional[Dict[str, Any]]): Additional filter conditions.

        Returns:
            Tuple[List[str], List[float]]: List of IDs and their average distances.
        """
        query_embeddings = [self._encode_text(text) for text in query_texts]
        distance_expressions = [
            f"({field}_vector <=> %s::vector)" for field in field_names
        ]
        avg_distance_expr = f"({' + '.join(distance_expressions)}) / {len(field_names)}"

        where_clause = ""
        filter_params = []
        if filter:
            where_conditions = [f"{k} = %s" for k in filter.keys()]
            where_clause = "WHERE " + " AND ".join(where_conditions)
            filter_params = list(filter.values())

        query = f"""
            SELECT id, {avg_distance_expr} AS avg_distance
            FROM {table_name}
            {where_clause}
            ORDER BY avg_distance ASC
            LIMIT %s
        """
        params = query_embeddings + filter_params + [n_results]
        self.cur.execute(query, params)
        results = self.cur.fetchall()
        ids = [row[0] for row in results]
        distances = [row[1] for row in results]
        return ids, distances

    def _get_top_k_by_keys(
        self,
        key_type: Union[str, List[str]],
        key: Union[str, List[str]],
        n_results: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[float]]:
        """Get top-k IDs based on key type and query text."""
        key_types = [key_type] if isinstance(key_type, str) else key_type
        keys = [key] if isinstance(key, str) else key

        is_trajectory = all(kt in self.trajectory_fields for kt in key_types)
        is_state = all(kt in self.state_fields for kt in key_types)
        assert is_trajectory or is_state, "Invalid key types."

        table_name = "trajectories" if is_trajectory else "states"
        return self._compute_top_k_nearest_neighbors_by_avg_distance(
            table_name, key_types, keys, n_results, filter
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
        query = """
            SELECT *
            FROM trajectories
            WHERE environment_id = %s
        """
        params = [env_id]
        if outcome_filter:
            query += " AND success = %s"
            params.append(outcome_filter["success"])
        query += " ORDER BY json_array_length(observations::json) ASC LIMIT %s"
        params.append(n_results)

        self.cur.execute(query, params)
        results = self.cur.fetchall()
        columns = [desc[0] for desc in self.cur.description]
        all_trajectories = [dict(zip(columns, row)) for row in results]

        success_entries = [
            self._format_conversion(m) for m in all_trajectories if m["success"]
        ]
        failure_entries = [
            self._format_conversion(m) for m in all_trajectories if not m["success"]
        ]
        return success_entries, failure_entries

    def get_similar_entries(
        self,
        key_type: Union[str, List[str]],
        key: Union[str, List[str]],
        k: int = 5,
        outcome: str = None,
        window: int = 1,
        filtered_environment_id: str = None,
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
                    # Query states for this trajectory
                    self.cur.execute(
                        """
                        SELECT id, position
                        FROM states
                        WHERE trajectory_id = %s
                    """,
                        (t_id,),
                    )
                    t_states = self.cur.fetchall()
                    state_ids = [row[0] for row in t_states]
                    state_positions = {row[0]: row[1] for row in t_states}

                    if not state_ids:
                        continue

                    # Compute average state distance for each state
                    distance_expressions = [
                        f"({field}_vector <=> %s::vector)" for field in state_key_types
                    ]
                    avg_distance_expr = (
                        f"({' + '.join(distance_expressions)}) / {len(state_key_types)}"
                    )
                    query = f"""
                        SELECT id, {avg_distance_expr} AS avg_distance
                        FROM states
                        WHERE id = ANY(%s)
                        ORDER BY avg_distance ASC
                        LIMIT 1
                    """
                    state_query_embeddings = [
                        self._encode_text(text) for text in state_only_key_filtered
                    ]
                    self.cur.execute(
                        query, tuple(state_query_embeddings) + (state_ids,)
                    )
                    # state_distances = self.cur.fetchall()
                    min_state_id, min_dist = self.cur.fetchone()

                    # Find state with minimum distance
                    # min_dist = float("inf")
                    # min_state_id = None
                    # for state_id, dist in state_distances:
                    #     if dist < min_dist:
                    #         min_dist = dist
                    #         min_state_id = state_id
                    critical_state_positions[t_id] = state_positions[min_state_id]
                    t_distances[i] += min_dist

            sort_indices = np.argsort(t_distances)
            chosen_t_ids = [t_ids[i] for i in sort_indices[:k]]
        else:
            s_ids, s_distances = self._get_top_k_by_keys(state_key_types, keys, k * 2)
            for s_id in s_ids:
                self.cur.execute(
                    """
                    SELECT trajectory_id, position
                    FROM states
                    WHERE id = %s
                """,
                    (s_id,),
                )
                state = self.cur.fetchone()
                trajectory_id, position = state
                if trajectory_id not in chosen_t_ids:
                    chosen_t_ids.append(trajectory_id)
                    critical_state_positions[trajectory_id] = position
                    if len(chosen_t_ids) >= k:
                        break

        if not chosen_t_ids:
            return [], []

        # Retrieve full trajectories
        self.cur.execute(
            """
            SELECT *
            FROM trajectories
            WHERE id = ANY(%s)
        """,
            (chosen_t_ids,),
        )
        results = self.cur.fetchall()
        columns = [desc[0] for desc in self.cur.description]
        similar_entries = [dict(zip(columns, row)) for row in results]

        # Apply windowing
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
