import os
import ast
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any, Tuple

import psycopg2
from psycopg2 import sql

from sentence_transformers import SentenceTransformer

# todo, make a config file for the db
TRAJECTORY_COLS = {
    "goal": "goal_embedding",
    "category": "category_embedding",
    "plan": "plan_embedding",
    "reflection": "reflection_embedding",
    "summary": "summary_embedding",
}

STATE_COLS = {
    "observation": "observation_embedding",
    "reasoning": "reasoning_embedding",
    "action": "action_embedding",
}

RULE_COLS = {
    "name": "name_embedding",
    "content": "rule_embedding",
    "context": "context_embedding",
}


class LearningDB:

    # def __init__(self, db_path: str = "data/learning.db"):
    def __init__(
        self,
        db_name: str = "learning_db",
        db_path: str = "~/.learning_db",
        user: str = "agent",
        password: str = "password",
        embedding_dim: int = 384,
    ):
        """Initialize the learning database and ensure pgvector extension is created."""

        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        host = os.path.join(os.path.expanduser(db_path), "pg_socket")

        self._create_database_if_missing(db_name, user, password, host)
        self.conn = self._connect_to_db(db_name, user, password, host)
        self.cursor = self.conn.cursor()

        # Ensure pgvector extension is created
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create all necessary tables
        self._create_tables()
        self.conn.commit()

    def _connect_to_db(self, db_name, user, password, host):
        """Returns a connection to the specified database."""
        return psycopg2.connect(dbname=db_name, user=user, password=password, host=host)

    def _create_database_if_missing(self, db_name, user, password, host):
        """Check if the database exists and create it if necessary."""
        try:
            with self._connect_to_db("postgres", user, password, host) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,)
                    )
                    if not cur.fetchone():
                        print(f"Database '{db_name}' does not exist. Creating...")
                        cur.execute(f"CREATE DATABASE {db_name} OWNER {user};")
        except psycopg2.Error as e:
            print(f"Error checking/creating database: {e}")
            raise

    def _create_tables(self):
        """Create the trajectories and states tables if they don't exist."""
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS trajectories (
                id SERIAL PRIMARY KEY,
                environment_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                goal_embedding vector({self.embedding_dim}),
                category TEXT NOT NULL,
                category_embedding vector({self.embedding_dim}),
                observations JSON NOT NULL,
                reasoning JSON,
                actions JSON NOT NULL,
                rewards JSON NOT NULL,
                plan TEXT,
                plan_embedding vector({self.embedding_dim}),
                reflection TEXT,
                reflection_embedding vector({self.embedding_dim}),
                summary TEXT,
                summary_embedding vector({self.embedding_dim})
            );
        """
        )
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS states (
                id SERIAL PRIMARY KEY,
                trajectory_id INTEGER NOT NULL,
                observation TEXT NOT NULL,
                observation_embedding vector({self.embedding_dim}),
                reasoning TEXT,
                reasoning_embedding vector({self.embedding_dim}),
                action TEXT NOT NULL,
                action_embedding vector({self.embedding_dim}),
                next_state TEXT NOT NULL,
                FOREIGN KEY (trajectory_id) REFERENCES trajectories (id)
            );
        """
        )
        self._create_rule_table()

    def _encode_text(self, text: Optional[str]) -> Optional[List[float]]:
        return self.model.encode(text).tolist() if text else None

    def store_episode(
        self,
        environment_id: str,
        goal: str,
        category: str,
        observations: List[Any],
        reasoning: List[str],
        actions: List[Any],
        rewards: List[float],
        plan: Optional[str],
        reflection: Optional[str],
        summary: Optional[str],
    ):
        """Insert an episode (trajectory + states) into the database."""
        trajectory_id = self._insert_trajectory(
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

        # todo: is there a need for backing up?

    def _insert_trajectory(
        self,
        environment_id: str,
        goal: str,
        category: str,
        observations: List[Any],
        reasoning: List[str],
        actions: List[Any],
        rewards: List[float],
        plan: Optional[str],
        reflection: Optional[str],
        summary: Optional[str],
    ) -> int:
        """
        Example method to insert a trajectory into the 'trajectories' table,
        encoding text fields into embeddings using SentenceTransformer.
        """
        observations_str = json.dumps(
            [observation.structured for observation in observations]
        )
        reasoning_str = json.dumps(reasoning) if reasoning else None
        actions_str = json.dumps([action.text for action in actions])
        rewards_str = json.dumps(rewards)

        # Generate embeddings for trajectory fields
        goal_embedding = self._encode_text(goal)
        category_embedding = self._encode_text(category)
        plan_embedding = self._encode_text(plan)
        reflection_embedding = self._encode_text(reflection)
        summary_embedding = self._encode_text(summary)

        insert_query = f"""
            INSERT INTO trajectories (
                environment_id, 
                goal, goal_embedding,
                category, category_embedding,
                observations, reasoning, actions, rewards,
                plan, plan_embedding,
                reflection, reflection_embedding,
                summary, summary_embedding
            )
            VALUES (
                %s, 
                %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s
            )
            RETURNING id
        """
        self.cursor.execute(
            insert_query,
            (
                environment_id,
                goal,
                goal_embedding,
                category if category else "",
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
        trajectory_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return trajectory_id

    def _insert_states(self, trajectory_id, observations, actions, reasoning):
        """
        Example method to insert a state into the 'states' table,
        encoding text fields into embeddings using SentenceTransformer.
        """
        for i in range(len(observations) - 1):
            state = observations[i].structured
            next_state = observations[i + 1].structured
            reason_text = reasoning[i] if reasoning and i < len(reasoning) else None
            action_text = actions[i].text if i < len(actions) else None

            self.cursor.execute(
                """
                INSERT INTO states (
                    trajectory_id,
                    observation, observation_embedding,
                    reasoning, reasoning_embedding,
                    action, action_embedding,
                    next_state
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    trajectory_id,
                    state,
                    self._encode_text(state),
                    reason_text,
                    self._encode_text(reason_text),
                    action_text,
                    self._encode_text(action_text),
                    next_state,
                ),
            )
        self.conn.commit()

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        table_name: str,
        embedding_columns: list,  # e.g. ["goal_embedding", "category_embedding"]
        query_embeddings: list,  # list of np.ndarray or list of float
        k: int,
        constraint: Optional[Tuple[str, Any]] = None,
    ):
        """
        Compute top-k nearest neighbors by averaging distances across multiple pgvector columns.
        Returns:
            (top_k_distances, top_k_neighbors)
        """
        # todo, if there is a fix set of combinations, we should store the combinations of vectors instead
        all_distances, all_neighbors = [], []

        # Collect top-k results for each embedding column
        for col_name, query_vec in zip(embedding_columns, query_embeddings):
            # Make sure query_vec is a Python list of floats
            if isinstance(query_vec, np.ndarray):
                query_vec = query_vec.tolist()

            where_clause = (
                f"WHERE {constraint[0]} = {constraint[1]}" if constraint else ""
            )

            self.cursor.execute(
                f"""
                SELECT id, ({col_name} <=> %s::vector) AS distance
                FROM {table_name}
                {where_clause}
                ORDER BY distance ASC
                LIMIT {k}
            """,
                (query_vec,),
            )

            rows = self.cursor.fetchall()
            distances = [r[1] for r in rows]
            neighbors = [r[0] for r in rows]

            all_distances.append(np.array(distances, dtype=np.float32))
            all_neighbors.append(np.array(neighbors, dtype=np.int32))

        # Aggregate distances for each candidate (neighbor)
        candidates = {}  # { neighbor_id: [dist1, dist2, ...] }
        for col_i, (distances, neighbors) in enumerate(
            zip(all_distances, all_neighbors)
        ):
            worst_distance = distances[-1] if len(distances) else 1.0
            for nb, dist in zip(neighbors, distances):
                candidates.setdefault(nb, []).append(dist)
            # Pad any candidates that didn't appear in this column's top-k
            # todo: potential accuracy degradation dur to padding
            for nb in candidates:
                if len(candidates[nb]) < col_i + 1:
                    candidates[nb].append(worst_distance)

        # Compute average distance, then pick top-k
        avg_distances = {
            nb: float(np.mean(dist_list)) for nb, dist_list in candidates.items()
        }
        sorted_candidates = sorted(avg_distances.items(), key=lambda x: x[1])[:k]

        top_k_neighbors = [nb for nb, _ in sorted_candidates]
        top_k_distances = [dist for _, dist in sorted_candidates]
        return np.array(top_k_distances), np.array(top_k_neighbors)

    def _get_top_k_by_keys(
        self, key_type: Union[str, List[str]], key: Union[str, List[str]], k: int = 5
    ):
        """
        Get top-k entries based on one or more key_types and keys, using Postgres + pgvector.
        Returns:
            (list_of_neighbor_ids, list_of_distances)
        """
        # Ensure we have lists for both key_type and key
        key_type = [key_type] if isinstance(key_type, str) else key_type
        key = [key] if isinstance(key, str) else key

        # Determine which table to use
        is_trajectory = all(kt in TRAJECTORY_COLS for kt in key_type)
        is_state = all(kt in STATE_COLS for kt in key_type)
        is_rule = all(kt in RULE_COLS for kt in key_type)
        assert (
            sum([is_trajectory, is_state, is_rule]) == 1
        ), "Invalid or mixed key types."

        if is_trajectory:
            table_name, col_map = "trajectories", TRAJECTORY_COLS
        elif is_state:
            table_name, col_map = "states", STATE_COLS
        else:
            table_name, col_map = "rules", RULE_COLS

        embedding_columns = [col_map[kt] for kt in key_type]
        query_embeddings = [self._encode_text(k_str) for k_str in key]

        # Compute top-k by averaging distance across columns
        distances, neighbors = self._compute_top_k_nearest_neighbors_by_avg_distance(
            table_name, embedding_columns, query_embeddings, k
        )
        return [int(i) for i in neighbors], list(distances)

    def _filter_by_outcome(
        self, ids: List[int], outcome: str
    ) -> Tuple[List[int], List[int]]:
        """
        Filter a list of trajectory IDs by a given outcome (winning or losing),
        leveraging JSON functions to see if the numeric reward value is present.
        Returns: (list_of_matching_ids, list_of_indices_in_ids)
        """
        value = 1 if outcome == "winning" else 0
        query = """
            SELECT id
            FROM trajectories
            WHERE id = ANY(%s)
            AND (rewards::jsonb->>(jsonb_array_length(rewards::jsonb) - 1))::float = %s
        """
        self.cursor.execute(query, (ids, value))
        rows = self.cursor.fetchall()

        trajectory_ids = [row[0] for row in rows]
        indices = [ids.index(tid) for tid in trajectory_ids]
        return trajectory_ids, indices

    def _fetch_trajectory_dict(self, trajectory_id: int) -> Optional[Dict]:
        """Return a dict representing a single trajectory, or None if not found."""
        self.cursor.execute(
            """
            SELECT
                id, environment_id, goal, category,
                observations, reasoning, actions, rewards,
                plan, reflection, summary
            FROM trajectories
            WHERE id = %s
        """,
            (trajectory_id,),
        )
        row = self.cursor.fetchone()
        if not row:
            return None

        rules = self._get_rules_for_id(row[0])
        return {
            "environment_id": row[1],
            "goal": row[2],
            "category": row[3],
            "observation": json.loads(row[4]),
            "reasoning": json.loads(row[5]) if row[5] else None,
            "action": json.loads(row[6]),
            "rewards": json.loads(row[7]),
            "plan": row[8],
            "reflection": row[9],
            "summary": row[10],
            "rules": rules,
        }

    def _fetch_state_ids_for_trajectory(self, trajectory_id: int) -> List[int]:
        """Return all state IDs associated with a given trajectory, in DB order."""
        self.cursor.execute(
            "SELECT id FROM states WHERE trajectory_id = %s", (trajectory_id,)
        )
        rows = self.cursor.fetchall()
        return [r[0] for r in rows]

    def _is_last_reward_successful(self, rewards: List[float]) -> bool:
        """Check if the final reward in a list is 1.0."""
        return bool(rewards and rewards[-1] == 1.0)

    def _exact_match(self, outcome, env_id: str, k: int):

        self.cursor.execute(
            """
            SELECT id
            FROM trajectories
            WHERE environment_id = %s
            ORDER BY char_length(observations) ASC
        """,
            (env_id,),
        )
        rows = self.cursor.fetchall()
        sorted_ids = [r[0] for r in rows]

        if outcome:
            sorted_ids, _ = self._filter_by_outcome(sorted_ids, outcome)

        success_entries, failure_entries = [], []
        for tid in sorted_ids[:k]:
            traj_dict = self._fetch_trajectory_dict(tid)
            if not traj_dict:
                continue
            if self._is_last_reward_successful(traj_dict["rewards"]):
                success_entries.append(traj_dict)
            else:
                failure_entries.append(traj_dict)

        return success_entries, failure_entries

    def get_similar_entries(
        self,
        key_type: Union[str, List[str]],
        key: Union[str, List[str]],
        k: int = 5,
        outcome: str = None,
        window: int = 1,
    ) -> List[Dict]:

        # Ensure lists
        key_type = [key_type] if isinstance(key_type, str) else key_type
        key = [key] if isinstance(key, str) else key

        # 1) If key_type == "environment_id", skip embeddings and do exact matching
        if key_type == ["environment_id"]:
            # We assume there's exactly one env_id in `key`
            return self._exact_match(outcome, key[0], k)

        # 2) Otherwise, do vector search at trajectory or state level
        trajectory_key_types = [kt for kt in key_type if kt in TRAJECTORY_COLS]
        state_key_types = [kt for kt in key_type if kt in STATE_COLS]

        sorted_trajectory_ids = []
        critical_states = {}
        if trajectory_key_types:
            key_filtered = [
                k for kt, k in zip(key_type, key) if kt in trajectory_key_types
            ]
            extended_k = (
                k * (3 if outcome else 1) * (2 if len(state_key_types) > 0 else 1)
            )
            trajectory_ids, trajectory_distances = self._get_top_k_by_keys(
                trajectory_key_types, key_filtered, extended_k
            )

            if outcome:
                trajectory_ids, indices = self._filter_by_outcome(
                    trajectory_ids, outcome
                )
                trajectory_distances = [trajectory_distances[i] for i in indices]

            # If we also have state-level keys, refine with a state-level search
            if state_key_types:
                state_key_embeddings = {
                    kt: self._encode_text(k) for kt, k in zip(state_key_types, key)
                }
                # For each trajectory, retrieve the associated states
                for i, tid in enumerate(trajectory_ids):
                    state_distances, state_ids = (
                        self._compute_top_k_nearest_neighbors_by_avg_distance(
                            "states",
                            [STATE_COLS[kt] for kt in state_key_types],
                            [state_key_embeddings[kt] for kt in state_key_types],
                            k=window,
                            constraint=("trajectory_id", tid),
                        )
                    )
                    trajectory_distances[i] += min(state_distances)
                    critical_states[tid] = state_ids[0]

            # Sort the combined distances from low to high, meaning most similar first
            ranked_indices = np.argsort(trajectory_distances)
            sorted_trajectory_ids = [trajectory_ids[i] for i in ranked_indices]

        else:
            assert state_key_types, "Invalid key types for search."
            state_ids, state_distances = self._get_top_k_by_keys(
                state_key_types, key, k * (2 if outcome else 1)
            )
            for state_id in state_ids:
                self.state_cursor.execute(
                    "SELECT trajectory_id FROM states WHERE id = ?", (state_id,)
                )
                row = self.state_cursor.fetchone()
                if row[0] not in critical_states:
                    critical_states[row[0]] = [state_id]
                    sorted_trajectory_ids.append(row[0])

            if outcome:
                _, indices = self._filter_by_outcome(sorted_trajectory_ids, outcome)
                sorted_trajectory_ids = [
                    sorted_trajectory_ids[i] for i in sorted(indices)
                ]
                critical_states = {
                    tid: critical_states[tid] for tid in sorted_trajectory_ids
                }

        # 3) Fetch the actual entries from the DB based on the IDs
        similar_entries = []
        # If we have trajectory IDs but no state IDs, fetch whole trajectories
        if sorted_trajectory_ids and not critical_states:
            for trajectory_id in sorted_trajectory_ids:
                traj_dict = self._fetch_trajectory_dict(trajectory_id)
                if not traj_dict:
                    continue
                similar_entries.append(traj_dict)
                if len(similar_entries) >= k:
                    break
        # If we have state IDs, fetch windows around those states
        elif critical_states:
            for trajectory_id in sorted_trajectory_ids:
                state_id = critical_states[trajectory_id]
                traj_dict = self._fetch_trajectory_dict(trajectory_id)
                if not traj_dict:
                    continue
                # Find the id of the state in the trajectory
                self.cursor.execute(
                    "SELECT id FROM states WHERE trajectory_id = %s",
                    (trajectory_id,),
                )
                state_ids = [row[0] for row in self.cursor.fetchall()]
                state_id_index = state_ids.index(state_id)
                window_start = max(0, state_id_index - window)
                window_end = min(
                    len(state_ids) + 1, state_id_index + window + 1
                )  # Fixed off-by-one error

                for key in ["observation", "reasoning", "action"]:
                    if traj_dict[key]:
                        traj_dict[key] = traj_dict[key][window_start:window_end]
                similar_entries.append(traj_dict)

                if len(similar_entries) >= k:
                    break

        # Create two separate lists of entries for success vs failure
        success_entries = [
            entry
            for entry in similar_entries
            if self._is_last_reward_successful(entry["rewards"])
        ]
        failure_entries = [
            entry
            for entry in similar_entries
            if not self._is_last_reward_successful(entry["rewards"])
        ]

        return success_entries, failure_entries

    ############################################################
    # implementation for rules to be tested

    def _create_rule_table(self):
        """
        Create the rules table with vector columns for embeddings.
        """
        self.cursor.execute(
            sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS rules (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    name_embedding vector({self.embedding_dim}),
                    rule_content TEXT NOT NULL,
                    rule_embedding vector({self.embedding_dim}),
                    context TEXT,
                    context_embedding vector({self.embedding_dim}),
                    trajectory_ids TEXT,
                    state_ids TEXT
                )
                """
            )
        )

    def store_rule(
        self,
        name: str,
        rule_content: str,
        context: str,
        trajectory_ids: List[int],
        state_ids: List[int],
    ):
        """
        Store a rule in the PostgreSQL 'rules' table with vector embeddings (pgvector).

        Args:
            name: Name of the rule
            rule_content: Full text/content of the rule
            context: Context text (optional) for the rule
            trajectory_ids: List of trajectory IDs associated with the rule
            state_ids: List of state IDs associated with the rule

        Returns:
            The newly inserted rule's primary key (id).
        """
        # Encode embeddings as lists of floats for pgvector
        name_embedding = self.model.encode(name).tolist()
        rule_embedding = self.model.encode(rule_content).tolist()
        context_embedding = self.model.encode(context).tolist()

        trajectory_ids_str = str(trajectory_ids)
        state_ids_str = str(state_ids)

        self.cursor.execute(
            """
            INSERT INTO rules (
                name, name_embedding,
                rule_content, rule_embedding,
                context, context_embedding,
                trajectory_ids,
                state_ids
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                name,
                name_embedding,
                rule_content,
                rule_embedding,
                context,
                context_embedding,
                trajectory_ids_str,
                state_ids_str,
            ),
        )
        # Fetch the newly inserted rule ID
        rule_id = self.cursor.fetchone()[0]
        self.conn.commit()

        return rule_id

    def get_rules(
        self, key_types: List[str], keys: List[str], k: int = 5
    ) -> List[Dict]:
        # If key_types is an empty list, search for all rules
        if not key_types:
            self.cursor.execute(
                "SELECT id, name, rule_content, context, trajectory_ids, state_ids FROM rules"
            )
            rules = self.cursor.fetchall()
            rule_ids = [rule[0] for rule in rules]
        else:
            # Otherwise, use embeddings to search for rules
            rule_ids, rule_distances = self._get_top_k_by_keys(key_types, keys, k)
            self.cursor.execute(
                "SELECT id, name, rule_content, context, trajectory_ids, state_ids FROM rules WHERE id IN ({})".format(
                    ", ".join(map(str, rule_ids))
                )
            )
            rules = self.cursor.fetchall()

        # Turn rules into a list of dictionaries
        rules = [dict(zip(self.cursor.description, rule)) for rule in rules]

        # Let's also fetch corresponding trajectories and states
        for rule in rules:
            trajectory_ids = rule["trajectory_ids"]
            state_ids = rule["state_ids"]
            for trajectory_id in trajectory_ids:
                self.trajectory_cursor.execute(
                    "SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary FROM trajectories WHERE id = %s",
                    (trajectory_id,),
                )
                trajectories = self.trajectory_cursor.fetchall()
                rule["trajectories"] = trajectories
            for state_id in state_ids:
                self.state_cursor.execute(
                    "SELECT id, state, reasoning, action, next_state FROM states WHERE id = %s",
                    (state_id,),
                )
                states = self.state_cursor.fetchall()
                rule["states"] = states

        return rules

    def _get_rules_for_id(self, trajectory_id: int) -> List[Dict]:
        """
        Helper function to get rules for a specific trajectory ID from PostgreSQL.
        """

        self.cursor.execute(
            """
            SELECT name, rule_content, context
            FROM rules
            WHERE trajectory_ids LIKE %s
            """,
            (f"%{trajectory_id}%",),
        )
        rows = self.cursor.fetchall()

        return [{"name": r[0], "content": r[1], "context": r[2]} for r in rows]

    def get_similar_sets(self, n: int, k: int):
        """
        Get similar sets of episodes by finding trajectories with similar goals using vector search.
        """
        # todo, the reward is not general
        self.cursor.execute(
            """
            SELECT id, goal, category, observations, reasoning, actions, plan
            FROM trajectories
            WHERE jsonb_array_length(rewards::jsonb) > 0
            AND EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(rewards) AS reward WHERE reward = '1'
            )
            ORDER BY RANDOM()
            LIMIT %s
            """,
            (n,),
        )
        base_trajectories = self.cursor.fetchall()

        similar_sets = []
        # For each base trajectory, find k similar ones using goal embeddings
        for base_traj in base_trajectories:
            base_goal = base_traj[1]
            base_category = base_traj[2]

            success_entries, _ = self.get_similar_entries(
                key_type=["goal", "category"],
                key=[base_goal, base_category],
                outcome="winning",
                k=k,
            )

            # Skip if no similar entries found
            if not success_entries:
                continue

            # Format entries into similar set
            similar_set = []
            for entry in success_entries:
                similar_set.append(
                    {
                        "goal": entry["goal"],
                        "observation": entry["observation"],
                        "reasoning": entry["reasoning"],
                        "action": entry["action"],
                        "plan": entry["plan"],
                    }
                )
            similar_sets.append(similar_set)

        return similar_sets

    def get_contrastive_pairs(self):
        """Fetch contrastive pairs of successful and failed episodes for each environment_id"""
        # Get all environment IDs

        self.cursor.execute("SELECT DISTINCT environment_id FROM trajectories;")
        env_ids = self.cursor.fetchall()  # list of tuples [(env_id1,), (env_id2,), ...]

        contrastive_pairs = []

        # For each environment ID, get one successful and one failed episode
        for env_id_tuple in env_ids:
            env_id = env_id_tuple[0]
            # todo: again, the reward is not general
            self.cursor.execute(
                """
                SELECT goal, observations, reasoning, actions, plan, LENGTH(observations) as traj_len
                FROM trajectories
                WHERE environment_id = %s
                AND jsonb_array_length(rewards::jsonb) > 0
                AND rewards::text LIKE '%1%'
                ORDER BY traj_len ASC
                LIMIT 1
                """,
                (env_id,),
            )
            success_row = self.cursor.fetchone()

            # Get shortest failed episode
            self.cursor.execute(
                """
                SELECT goal, observations, reasoning, actions, plan, LENGTH(observations) as traj_len
                FROM trajectories
                WHERE environment_id = %s
                AND (
                    jsonb_array_length(rewards::jsonb) = 0
                    OR rewards::text NOT LIKE '%1%'
                    )
                ORDER BY traj_len ASC
                LIMIT 1
                """,
                (env_id,),
            )
            failure_row = self.cursor.fetchone()

            # Only add if we have both success and failure
            if success_row and failure_row:
                success_entry = {
                    "goal": success_row[0],
                    "observation": json.loads(success_row[1]),
                    "reasoning": json.loads(success_row[2]) if success_row[2] else None,
                    "action": json.loads(success_row[3]),
                    "plan": success_row[4],
                }
                failure_entry = {
                    "goal": failure_row[0],
                    "observation": json.loads(failure_row[1]),
                    "reasoning": json.loads(failure_row[2]) if failure_row[2] else None,
                    "action": json.loads(failure_row[3]),
                    "plan": failure_row[4],
                }
                contrastive_pairs.append((success_entry, failure_entry))

        return contrastive_pairs
