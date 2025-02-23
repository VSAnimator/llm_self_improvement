import os
import ast
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any

import psycopg2
from psycopg2 import sql

from sentence_transformers import SentenceTransformer


class LearningDB:

    # def __init__(self, db_path: str = "data/learning.db"):
    def __init__(
        self,
        db_name: str = "learning_db",
        db_path: str = "~/.learning_db",
        user: str = "agent",
        password: str = "password",
        port: int = 5432,
        embedding_dim: int = 384,
    ):
        """
        Initialize the learning database with PostgreSQL and pgvector for embeddings.

        Args:
            db_name: Name of the PostgreSQL database.
            user: Username for the PostgreSQL database.
            password: Password for the PostgreSQL database.
            host: Host address of the PostgreSQL database.
            port: Port number of the PostgreSQL database.
            embedding_dim: Dimension of the embeddings (defaults to 384 for 'all-MiniLM-L6-v2').
        """

        self.embedding_dim = embedding_dim
        host = os.path.join(os.path.expanduser(db_path), "pg_socket")

        self._create_database_if_missing(db_name, user, password, host, port)
        self.conn = self._connect_to_db(db_name, user, password, host, port)
        self.cursor = self.conn.cursor()

        # Ensure pgvector extension is created
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create tables
        self._create_trajectory_table()
        self._create_state_table()
        self._create_rule_table()
        self.conn.commit()
        # print(f"Connected to PostgreSQL database '{db_name}' successfully.")

        # Initialize sentence transformer for encoding text
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _connect_to_db(self, db_name, user, password, host, port):
        """Returns a connection to the specified database."""
        return psycopg2.connect(dbname=db_name, user=user, password=password, host=host)

    def _create_database_if_missing(self, db_name, user, password, host, port):
        """Check if the database exists and create it if necessary."""
        try:
            with self._connect_to_db("postgres", user, password, host, port) as conn:
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

    def _create_trajectory_table(self):
        """
        Create the trajectories table with vector columns for embeddings.
        """
        self.cursor.execute(
            sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id SERIAL PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    goal_embedding vector({self.embedding_dim}),
                    category TEXT NOT NULL,
                    category_embedding vector({self.embedding_dim}),
                    observations TEXT NOT NULL,
                    reasoning TEXT,
                    actions TEXT NOT NULL,
                    rewards TEXT NOT NULL,
                    plan TEXT,
                    plan_embedding vector({self.embedding_dim}),
                    reflection TEXT,
                    reflection_embedding vector({self.embedding_dim}),
                    summary TEXT,
                    summary_embedding vector({self.embedding_dim})
                )
                """
            )
        )

    def _create_state_table(self):
        """
        Create the states table with vector columns for embeddings,
        referencing trajectories(id).
        """
        self.cursor.execute(
            sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS states (
                    id SERIAL PRIMARY KEY,
                    trajectory_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    state_embedding vector({self.embedding_dim}),
                    reasoning TEXT,
                    reasoning_embedding vector({self.embedding_dim}),
                    action TEXT NOT NULL,
                    action_embedding vector({self.embedding_dim}),
                    next_state TEXT NOT NULL,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories (id)
                )
                """
            )
        )

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
        """Store an episode in both trajectory and state databases"""
        # Store trajectory

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
        goal_embedding = self.model.encode(goal).tolist() if goal else None
        category_embedding = self.model.encode(category).tolist() if category else None
        plan_embedding = self.model.encode(plan).tolist() if plan else None
        reflection_embedding = (
            self.model.encode(reflection).tolist() if reflection else None
        )
        summary_embedding = self.model.encode(summary).tolist() if summary else None

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
            # Get raw text for the state
            state = observations[i].structured
            # Encode into a Python list of floats
            state_embedding = self.model.encode(state).tolist()

            # Handle optional reasoning
            reasoning_i = reasoning[i] if reasoning else None
            reasoning_embedding = (
                self.model.encode(reasoning_i).tolist() if reasoning_i else None
            )

            # Action text
            action = actions[i].text
            action_embedding = self.model.encode(action).tolist()

            # Next state text
            next_state = observations[i + 1].structured

            self.cursor.execute(
                """
                INSERT INTO states (
                    trajectory_id,
                    state, state_embedding,
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
                    state_embedding,
                    reasoning_i,
                    reasoning_embedding,
                    action,
                    action_embedding,
                    next_state,
                ),
            )
        # Commit once after the loop
        self.conn.commit()

    def _compute_top_k_nearest_neighbors_by_avg_distance(
        self,
        table_name: str,
        embedding_columns: list,  # e.g. ["goal_embedding", "category_embedding"]
        query_embeddings: list,  # list of np.ndarray or list of float
        k: int,
    ):
        """
        Compute top-k nearest neighbors by averaging distances across several pgvector columns.

        Args:
            table_name (str): Name of the table to query (e.g. 'trajectories', 'rules', etc.).
            embedding_columns (list): The names of the vector columns to search.
            query_embeddings (list): One embedding per column (same order as embedding_columns).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            - distances: np.ndarray of shape (1, k)
            - neighbors: np.ndarray of shape (1, k)
        """

        # todo, if there is a fix set of combinations, we should store the combinations of vectors instead

        all_distances = []
        all_neighbors = []

        # For each embedding column, get the top-k from Postgres
        for col_idx, col_name in enumerate(embedding_columns):
            query_vector = query_embeddings[col_idx]
            # Make sure query_vector is a Python list of floats if needed
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            self.cursor.execute(
                f"""
                SELECT
                    id,
                    ({col_name} <=> %s::vector) AS distance
                FROM {table_name}
                ORDER BY distance ASC
                LIMIT {k}
            """,
                (query_vector,),
            )

            rows = self.cursor.fetchall()
            distances = [r[1] for r in rows]
            neighbors = [r[0] for r in rows]

            all_distances.append(np.array(distances, dtype=np.float32))
            all_neighbors.append(np.array(neighbors, dtype=np.int32))

        candidates = {}  # neighbor_id -> list_of_distances_across_columns

        for col_i in range(len(embedding_columns)):
            col_distances = all_distances[col_i]
            col_neighbors = all_neighbors[col_i]
            worst_distance = col_distances[-1] if len(col_distances) > 0 else 1.0

            for nb, dist in zip(col_neighbors, col_distances):
                if nb not in candidates:
                    candidates[nb] = []
                candidates[nb].append(dist)

            # todo: the padding approach potentially compromises the accuracy of the average distance, can be fixed by more queries
            for nb in candidates:
                if len(candidates[nb]) < col_i + 1:
                    candidates[nb].append(worst_distance)

        avg_distances_dict = {
            nb: float(np.mean(dist_list)) for nb, dist_list in candidates.items()
        }

        sorted_candidates = sorted(avg_distances_dict.items(), key=lambda x: x[1])
        top_k = sorted_candidates[:k]

        top_k_neighbors = [nb for nb, _ in top_k]
        top_k_distances = [dist for _, dist in top_k]
        return np.array(top_k_distances), np.array(top_k_neighbors)

    def _get_top_k_by_keys(
        self, key_type: Union[str, List[str]], key: Union[str, List[str]], k: int = 5
    ):
        """
        Get top-k entries based on one or more key_types and keys, using Postgres+pgvector.
        """
        # Ensure we're dealing with lists
        if isinstance(key_type, str):
            key_type = [key_type]
        if isinstance(key, str):
            key = [key]

        # Decide table & columns
        trajectory_cols = {
            "environment_id": "environment_id_embedding",
            "goal": "goal_embedding",
            "category": "category_embedding",
            "plan": "plan_embedding",
            "reflection": "reflection_embedding",
            "summary": "summary_embedding",
        }

        state_cols = {
            "state": "state_embedding",
            "reasoning": "reasoning_embedding",
            "action": "action_embedding",
        }

        rule_cols = {
            "name": "name_embedding",
            "content": "rule_embedding",
            "context": "context_embedding",
        }

        is_trajectory = all(cols in trajectory_cols for cols in key_type)
        is_state = all(cols in state_cols for cols in key_type)
        is_rule = all(cols in rule_cols for cols in key_type)
        assert (
            sum([is_trajectory, is_state, is_rule]) == 1
        ), "Invalid key types or mix of types"

        if is_trajectory:
            table_name = "trajectories"
            col_map = trajectory_cols
        elif is_state:
            table_name = "states"
            col_map = state_cols
        else:
            table_name = "rules"
            col_map = rule_cols

        embedding_columns = [col_map[kt] for kt in key_type]

        # Encode all keys -> query_embeddings (list of np.ndarray)
        query_embeddings = [self.model.encode(k_str) for k_str in key]

        # Now compute top-k across these columns
        D, I = self._compute_top_k_nearest_neighbors_by_avg_distance(
            table_name, embedding_columns, query_embeddings, k
        )
        return [int(i) for i in I], list(D)

    def _filter_by_outcome(self, ids: list[int], outcome: str):
        # Use float values for the winning/losing outcomes
        # so that '[1.0]' matches actual JSON array elements like 1.0
        winning_value = 1.0 if outcome == "winning" else 0.0

        sql = """
            SELECT id
            FROM trajectories
            WHERE id = ANY(%s)
            AND rewards::jsonb @> %s::jsonb
        """
        self.cursor.execute(sql, (ids, json.dumps([winning_value])))

        rows = self.cursor.fetchall()  # returns list of (id, )

        trajectory_ids = [r[0] for r in rows]

        # Convert those IDs into indices from the original 'ids' list
        indices = [ids.index(tid) for tid in trajectory_ids]
        return trajectory_ids, indices

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

    def _exact_match(self, outcome, key: str, k: int):

        case_expression = """
            CASE
                WHEN jsonb_array_length(rewards::jsonb) > 0
                    AND (rewards ->> (jsonb_array_length(rewards::jsonb) - 1))::float = 1.0
                THEN 1
                ELSE 0
            END AS success
        """

        if outcome is not None:
            # Filter by outcome and get k shortest trajectories
            success_flag = 1 if outcome in ("success", "winning") else 0
            query = f"""
                SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary, char_length(observations) AS traj_len,
                    {case_expression}
                FROM trajectories
                WHERE environment_id = %s
                AND (
                    CASE
                    WHEN jsonb_array_length(rewards::jsonb) > 0
                        AND (rewards ->> (jsonb_array_length(rewards::jsonb) - 1))::float = 1.0
                    THEN 1
                    ELSE 0
                    END
                ) = %s
                ORDER BY traj_len ASC
                LIMIT %s
            """
            self.cursor.execute(query, (key, success_flag, k))
        else:
            # Get k shortest trajectories without filtering outcome
            query = f"""
                SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary, char_length(observations) AS traj_len,
                    {case_expression}
                FROM trajectories
                WHERE environment_id = %s
                ORDER BY traj_len ASC
                LIMIT %s
            """
            self.cursor.execute(query, (key, k))

        rows = self.cursor.fetchall()

        similar_entries = []
        success_labels = []

        for row in rows:
            # Fetch any associated rules
            rules = self._get_rules_for_id(row[0])

            entry = {
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
            similar_entries.append(entry)
            # If there's a valid rewards array, check the last reward for success
            if entry["rewards"] is not None and len(entry["rewards"]) > 0:
                success_labels.append(1 if entry["rewards"][-1] == 1 else 0)
            else:
                success_labels.append(0)

        # Separate success vs failure
        success_entries = [
            similar_entries[i] for i in range(len(success_labels)) if success_labels[i]
        ]
        failure_entries = [
            similar_entries[i]
            for i in range(len(success_labels))
            if not success_labels[i]
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

        # 1) Exact matching for 'environment_id' (no embedding search)
        if key_type == "environment_id":
            return self._exact_match(outcome, key, k)

        # 2) If key_type is NOT just environment_id, do vector search
        # Split out key types into trajectory and state-level key lists
        trajectory_keys = {
            "environment_id",
            "goal",
            "category",
            "plan",
            "reflection",
            "summary",
        }
        trajectory_key_types = [kt for kt in key_type if kt in trajectory_keys]
        state_key_types = [kt for kt in key_type if kt not in trajectory_keys]

        # If we have trajectory-level keys, get top k entries based on them
        # Otherwise, we'll do a state-level search
        trajectory_ids, trajectory_distances = None, None
        state_ids, state_distances = None, None
        if trajectory_key_types:
            # Filter key to trajectory-level keys
            key_filtered = [
                key[i] for i in range(len(key)) if key_type[i] in trajectory_key_types
            ]
            multiplier = (3 if outcome else 1) * (2 if len(state_key_types) > 0 else 1)
            extended_limit = k * multiplier

            trajectory_ids, trajectory_distances = self._get_top_k_by_keys(
                trajectory_key_types, key_filtered, extended_limit
            )
            # Filter by outcome if specified
            if outcome:
                trajectory_ids, indices = self._filter_by_outcome(
                    trajectory_ids, outcome
                )
                trajectory_distances = [trajectory_distances[i] for i in indices]
                # Sort by distances, low to high
                sorted_indices = np.argsort(trajectory_distances)
                trajectory_ids = [trajectory_ids[i] for i in sorted_indices]
                trajectory_distances = [trajectory_distances[i] for i in sorted_indices]

            # Now if state-level keys are present, we'll do a state-level search
            if state_key_types:
                state_distances = []
                state_ids = []
                state_key_embeddings = {
                    kt: self.model.encode([k])[0].reshape(1, -1)
                    for kt, k in zip(key_type, key)
                    if kt in state_key_types
                }
                # For each trajectory, retrieve the associated states
                for trajectory_id in trajectory_ids:
                    self.cursor.execute(
                        """
                        SELECT 
                            id, 
                            state, state_embedding::vector, 
                            reasoning, reasoning_embedding::vector,
                            action, action_embedding::vector, 
                            next_state
                        FROM states
                        WHERE trajectory_id = %s
                        """,
                        (trajectory_id,),
                    )
                    state_rows = self.cursor.fetchall()

                    # Loop through all relevant keys and compute distances
                    state_row_distances = []
                    embedding_col_mapping = {
                        "observation": 2,  # state_embedding column
                        "reasoning": 4,  # reasoning_embedding column
                        "action": 6,  # action_embedding column
                    }

                    for row in state_rows:
                        elem_distances = []
                        for state_key_type in state_key_types:
                            # Get embedding from DB
                            state_embedding = row[embedding_col_mapping[state_key_type]]
                            if state_embedding:
                                elem_distances.append(
                                    1
                                    - np.dot(
                                        np.array(
                                            ast.literal_eval(state_embedding),
                                            dtype=np.float32,
                                        ),
                                        state_key_embeddings[state_key_type][0],
                                    )
                                )
                            else:
                                elem_distances.append(1.0)

                        # Aggregate distances for each state row
                        state_row_distances.append(np.mean(elem_distances))

                    # Pick the row with the largest distance
                    max_index = np.argmax(state_row_distances)
                    state_ids.append(state_rows[max_index][0])
                    state_distances.append(state_row_distances[max_index])

                # Rerank by trajectory and state distances summed
                summed_distances = [
                    trajectory_distances[i] + state_distances[i]
                    for i in range(len(trajectory_distances))
                ]
                ranked_indices = np.argsort(summed_distances)
                trajectory_ids = [trajectory_ids[i] for i in ranked_indices]
                state_ids = [state_ids[i] for i in ranked_indices]
                trajectory_distances = [trajectory_distances[i] for i in ranked_indices]
                state_distances = [state_distances[i] for i in ranked_indices]
        else:
            # If no trajectory-level keys, do a state-level search
            state_ids, state_distances = self._get_top_k_by_keys(
                state_key_types, key, k * (2 if outcome else 1)
            )
            # Filter by outcome if specified
            if outcome:
                state_ids, indices = self._filter_by_outcome(state_ids, outcome)
                state_distances = [state_distances[i] for i in indices]

        # 3) Fetch the actual entries from the DB based on the IDs
        similar_entries = []
        success_labels = []
        outcome_flag = 1 if outcome == "winning" else 0

        # If we have trajectory IDs but no state IDs, fetch whole trajectories
        if trajectory_ids and not state_ids:
            for trajectory_id in trajectory_ids:
                self.cursor.execute(
                    """
                    SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary
                    FROM trajectories
                    WHERE id = %s
                    """,
                    (trajectory_id,),
                )
                row = self.cursor.fetchone()
                if row:
                    rules = self._get_rules_for_id(row[0])

                    entry = {
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

                    if (
                        not outcome
                        or ("rewards" in entry and entry["rewards"][-1] == outcome_flag)
                        or (
                            "trajectory" in entry
                            and entry["trajectory"]["rewards"][-1] == outcome_flag
                        )
                    ):
                        similar_entries.append(entry)
                        success_labels.append(entry["rewards"][-1] == 1)

                    if len(similar_entries) >= k:
                        break
        # If we have state IDs, fetch windows around those states
        elif state_ids:
            # First get trajectory IDs for these states
            trajectory_ids = []
            for state_id in state_ids:
                self.cursor.execute(
                    "SELECT trajectory_id FROM states WHERE id = %s", (state_id,)
                )
                row = self.cursor.fetchone()
                if row:
                    trajectory_ids.append(row[0])

            # Now fetch the states and surrounding context
            for state_id, trajectory_id in zip(state_ids, trajectory_ids):
                # Get the target state
                self.cursor.execute(
                    "SELECT id, state, reasoning, action, next_state FROM states WHERE id = %s",
                    (state_id,),
                )
                state_row = self.cursor.fetchone()

                if state_row:
                    # Get the trajectory info
                    self.cursor.execute(
                        """
                        SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary 
                        FROM trajectories
                        WHERE id = %s
                        """,
                        (trajectory_id,),
                    )
                    trajectory_row = self.cursor.fetchone()
                    if not trajectory_row:
                        continue

                    rules = self._get_rules_for_id(trajectory_row[0])

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

                    # Get the window of states around the target state
                    entry = {
                        "environment_id": trajectory_row[1],
                        "goal": trajectory_row[2],
                        "category": trajectory_row[3],
                        "observation": json.loads(trajectory_row[4])[
                            window_start:window_end
                        ],
                        "reasoning": (
                            json.loads(trajectory_row[5])[window_start:window_end]
                            if trajectory_row[5]
                            else None
                        ),
                        "action": json.loads(trajectory_row[6])[
                            window_start:window_end
                        ],
                        "rewards": json.loads(trajectory_row[7])[
                            window_start:window_end
                        ],
                        "plan": trajectory_row[8],
                        "reflection": trajectory_row[9],
                        "summary": trajectory_row[10],
                        "rules": rules,
                    }

                    rewards = json.loads(
                        trajectory_row[7]
                    )  # Don't want to filter for this

                    if not outcome or (rewards[-1] == outcome_flag):
                        similar_entries.append(entry)
                        success_labels.append(rewards[-1] == 1)

                    if len(similar_entries) >= k:
                        break

        # Create two separate lists of entries for success vs failure
        success_entries = [
            similar_entries[i] for i in range(len(success_labels)) if success_labels[i]
        ]
        failure_entries = [
            similar_entries[i]
            for i in range(len(success_labels))
            if not success_labels[i]
        ]

        return success_entries, failure_entries
