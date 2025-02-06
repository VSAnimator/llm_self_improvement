import os
import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
import json

class LearningDB:
    def __init__(self, db_path: str = "data/learning.db"):
        """Initialize the learning database with SQLite and FAISS indices
        
        Args:
            db_path: Path to store the SQLite database
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize SQLite connections
        self.trajectory_conn = sqlite3.connect(db_path)
        self.trajectory_cursor = self.trajectory_conn.cursor()
        
        self.state_conn = sqlite3.connect(db_path.replace('.db', '_states.db'))
        self.state_cursor = self.state_conn.cursor()
        
        self.rule_conn = sqlite3.connect(db_path.replace('.db', '_rules.db'))
        self.rule_cursor = self.rule_conn.cursor()
        
        # Create trajectory table
        self.trajectory_cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY,
                environment_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                goal_embedding BLOB,
                category TEXT NOT NULL,
                category_embedding BLOB,
                observations TEXT NOT NULL,
                reasoning TEXT,
                actions TEXT NOT NULL,
                rewards TEXT NOT NULL,
                plan TEXT,
                plan_embedding BLOB,
                reflection TEXT,
                reflection_embedding BLOB,
                summary TEXT,
                summary_embedding BLOB
            )
        """)
        
        # Create state table
        self.state_cursor.execute("""
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY,
                trajectory_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                state_embedding BLOB,
                reasoning TEXT,
                reasoning_embedding BLOB,
                action TEXT NOT NULL,
                action_embedding BLOB,
                next_state TEXT NOT NULL,
                FOREIGN KEY(trajectory_id) REFERENCES trajectories(id)
            )
        """)
        
        # Create rule table
        self.rule_cursor.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                name_embedding BLOB,
                rule_content TEXT NOT NULL,
                rule_embedding BLOB,
                context TEXT,
                context_embedding BLOB,
                trajectory_ids TEXT,
                state_ids TEXT
            )
        """)
        
        self.trajectory_conn.commit()
        self.state_conn.commit()
        self.rule_conn.commit()
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS indices
        self.index_path = os.path.dirname(db_path)
        
        # Trajectory-level indices
        self.trajectory_indices = {
            'goal': self._load_or_create_index('goal'),
            'category': self._load_or_create_index('category'),
            'plan': self._load_or_create_index('plan'),
            'reflection': self._load_or_create_index('reflection'),
            'summary': self._load_or_create_index('summary')
        }
        
        # State-level indices
        self.state_indices = {
            'state': self._load_or_create_index('state'),
            'reasoning': self._load_or_create_index('reasoning'),
            'action': self._load_or_create_index('action')
        }
        
        # Rule indices
        self.rule_indices = {
            'name': self._load_or_create_index('rule_name'),
            'context': self._load_or_create_index('rule_context'),
            'content': self._load_or_create_index('rule_content')
        }
        
        # Load id mappings
        self.trajectory_id_mappings = {}
        self.state_id_mappings = {}
        self.rule_id_mappings = {}
        
        for field in self.trajectory_indices.keys():
            mapping_path = os.path.join(self.index_path, f"trajectory_{field}_id_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.trajectory_id_mappings[field] = json.load(f)
            else:
                self.trajectory_id_mappings[field] = {}
                
        for field in self.state_indices.keys():
            mapping_path = os.path.join(self.index_path, f"state_{field}_id_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.state_id_mappings[field] = json.load(f)
            else:
                self.state_id_mappings[field] = {}
                
        for field in self.rule_indices.keys():
            mapping_path = os.path.join(self.index_path, f"rule_{field}_id_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.rule_id_mappings[field] = json.load(f)
            else:
                self.rule_id_mappings[field] = {}

    def _load_or_create_index(self, field: str) -> faiss.IndexFlatIP:
        """Load existing FAISS index or create new one"""
        index_path = os.path.join(self.index_path, f"{field}.index")
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        return faiss.IndexFlatIP(384)

    def _save_index(self, field: str, index: faiss.IndexFlatIP, is_trajectory: bool = True):
        """Save FAISS index to disk"""
        index_path = os.path.join(self.index_path, f"{field}.index")
        faiss.write_index(index, index_path)
        
        prefix = "trajectory_" if is_trajectory else "state_"
        mapping_path = os.path.join(self.index_path, f"{prefix}{field}_id_mapping.json")
        mappings = self.trajectory_id_mappings if is_trajectory else self.state_id_mappings
        with open(mapping_path, 'w') as f:
            json.dump(mappings[field], f)
    
    def _compute_top_k_nearest_neighbors_by_avg_distance(self, indices, query_embeddings, k):
        """
        Compute top-k nearest neighbors by averaging distances across several FAISS indices.
        
        Args:
            indices (list): List of FAISS indices.
            queries (np.ndarray): Query points (shape: [num_queries, dim]).
            k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            np.ndarray: Indices of the top-k nearest neighbors (shape: [num_queries, k]).
            np.ndarray: Averaged distances to the top-k nearest neighbors (shape: [num_queries, k]).
        """
        num_queries = 1
        all_distances = []
        all_neighbors = []
        
        # Query each index
        for i in range(len(indices)):
            distances, neighbors = indices[i].search(query_embeddings[i].reshape(1, -1), k)
            all_distances.append(distances)
            all_neighbors.append(neighbors)
        
        # Combine all results into a single list of candidates per query
        top_k_distances = []
        top_k_neighbors = []
        
        for i in range(num_queries):
            # Collect all neighbors and their distances for query i
            candidates = {}
            for idx in range(len(indices)):
                distances = all_distances[idx][i]
                neighbors = all_neighbors[idx][i]
                worst_distance = distances[-1]  # Worst distance in the top-k of this index
                
                for neighbor, distance in zip(neighbors, distances):
                    if neighbor not in candidates:
                        candidates[neighbor] = []
                    candidates[neighbor].append(distance)
                
                # Add the worst-case distance for neighbors not in this index
                for neighbor in candidates:
                    if len(candidates[neighbor]) < idx + 1:  # If this neighbor didn't appear in the current index
                        candidates[neighbor].append(worst_distance)
            
            # Compute average distances for all candidates
            avg_distances = {neighbor: np.mean(distances) for neighbor, distances in candidates.items()}
            
            # Sort candidates by average distance and select top-k
            sorted_candidates = sorted(avg_distances.items(), key=lambda x: x[1], reverse=True)
            top_k = sorted_candidates[:k]
            
            top_k_neighbors.append([item[0] for item in top_k])
            top_k_distances.append([item[1] for item in top_k])
        
        return np.array(top_k_distances), np.array(top_k_neighbors) 

    def _get_top_k_by_keys(self, key_type: Union[str, List[str]], key: Union[str, List[str]], k: int = 5) -> List[Dict]:
        """Get top k entries based on key_types and keys"""
        # Encode all keys
        key_embeddings = []
        for elem in key:
            key_embeddings.append(self.model.encode([elem])[0])
        
        # Determine if this is a trajectory or state level search
        trajectory_keys = {'environment_id', 'goal', 'category', 'plan', 'reflection', 'summary'} 
        rule_keys = {'name', 'context', 'content'}
        is_trajectory = any(kt in trajectory_keys for kt in key_type)
        is_rule = any(kt in rule_keys for kt in key_type)

        if is_rule:
            indices = self.rule_indices
            mappings = self.rule_id_mappings
        else:
            indices = self.trajectory_indices if is_trajectory else self.state_indices
            mappings = self.trajectory_id_mappings if is_trajectory else self.state_id_mappings

        D, I = self._compute_top_k_nearest_neighbors_by_avg_distance([indices[elem] for elem in key_type], key_embeddings, k)

        # Filter invalid results
        D = [d for d, i in zip(D[0], I[0]) if i != -1]
        I = [i for i in I[0] if i != -1]
        
        if not I:
            return [], []
        
        entry_ids = [mappings[key_type[0]][str(i)] for i in I]
        return entry_ids, D
    
    def _filter_by_outcome(self, ids, outcome):
        cursor = self.trajectory_cursor
        cursor.execute(f"""
            SELECT * FROM trajectories WHERE id IN ({', '.join(map(str, ids))})
            AND CASE 
                WHEN json_array_length(rewards) > 0 AND CAST(json_extract(rewards, '$[#-1]') AS FLOAT) = 1.0 THEN 1
                ELSE 0
            END = {1 if outcome == 'winning' else 0}
        """)
        trajectory_ids = [row[0] for row in cursor.fetchall()]
        # Also get indices of each id in the original list
        indices = [ids.index(id) for id in trajectory_ids]
        return trajectory_ids, indices

    """ Storing in the database """

    def store_rule(self, name: str, rule_content: str, context: str, trajectory_ids: List[int], state_ids: List[int]):
        """Store a rule in the rule database"""
        name_embedding = self.model.encode([name])[0].tobytes()
        rule_embedding = self.model.encode([rule_content])[0].tobytes()
        context_embedding = self.model.encode([context])[0].tobytes()
        self.rule_cursor.execute("""
            INSERT INTO rules (name, name_embedding, rule_content, rule_embedding, context, context_embedding, trajectory_ids, state_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, name_embedding, rule_content, rule_embedding, context, context_embedding, trajectory_ids, state_ids))

        rule_id = self.rule_cursor.lastrowid
        self.rule_conn.commit()

        # Add rule embeddings to FAISS indices
        rule_fields = {
            'name': (name, name_embedding),
            'context': (context, context_embedding),
            'content': (rule_content, rule_embedding)
        }
        
        for field, (value, embedding) in rule_fields.items():
            if value is not None:
                embedding_array = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
                self.rule_indices[field].add(embedding_array)
                curr_size = self.rule_indices[field].ntotal - 1
                self.rule_id_mappings[field][str(curr_size)] = rule_id
                self._save_index(field, self.rule_indices[field], True)

    def store_episode(self, environment_id: str, goal: str, category: str, observations: List[str], reasoning: List[str], 
                     actions: List[str], rewards: List[float], plan: Optional[str],
                     reflection: Optional[str], summary: Optional[str]):
        """Store an episode in both trajectory and state databases"""
        # Store trajectory
        observations_str = json.dumps([observation.structured for observation in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else None
        actions_str = json.dumps([action.text for action in actions])
        rewards_str = json.dumps(rewards)

        # Generate embeddings for trajectory fields
        goal_embedding = self.model.encode([goal])[0].tobytes() if goal else None
        category_embedding = self.model.encode([category])[0].tobytes() if category else None
        plan_embedding = self.model.encode([plan])[0].tobytes() if plan else None
        reflection_embedding = self.model.encode([reflection])[0].tobytes() if reflection else None
        summary_embedding = self.model.encode([summary])[0].tobytes() if summary else None

        # Store trajectory with embeddings
        self.trajectory_cursor.execute("""
            INSERT INTO trajectories (environment_id, goal, goal_embedding, category, category_embedding, 
                                    observations, reasoning, actions, rewards, plan, plan_embedding,
                                    reflection, reflection_embedding, summary, summary_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (environment_id, goal, goal_embedding, category if category else "", category_embedding,
              observations_str, reasoning_str, actions_str, rewards_str, 
              plan, plan_embedding, reflection, reflection_embedding, summary, summary_embedding))
        
        trajectory_id = self.trajectory_cursor.lastrowid
        self.trajectory_conn.commit()

        # Add trajectory embeddings to FAISS indices
        trajectory_fields = {
            'goal': (goal, goal_embedding),
            'category': (category, category_embedding),
            'plan': (plan, plan_embedding),
            'reflection': (reflection, reflection_embedding),
            'summary': (summary, summary_embedding)
        }
        
        for field, (value, embedding) in trajectory_fields.items():
            if value is not None and embedding is not None:
                embedding_array = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
                self.trajectory_indices[field].add(embedding_array)
                curr_size = self.trajectory_indices[field].ntotal - 1
                self.trajectory_id_mappings[field][str(curr_size)] = trajectory_id
                self._save_index(field, self.trajectory_indices[field], True)
        
        # Store individual states with embeddings
        for i in range(len(observations) - 1):
            state = observations[i].structured
            state_embedding = self.model.encode([state])[0].tobytes()
            
            reasoning_i = reasoning[i] if reasoning else None
            reasoning_embedding = self.model.encode([reasoning_i])[0].tobytes() if reasoning_i else None
            
            action = actions[i].text
            action_embedding = self.model.encode([action])[0].tobytes()
            
            next_state = observations[i+1].structured
            
            self.state_cursor.execute("""
                INSERT INTO states (trajectory_id, state, state_embedding, reasoning, reasoning_embedding,
                                  action, action_embedding, next_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (trajectory_id, state, state_embedding, reasoning_i, reasoning_embedding,
                 action, action_embedding, next_state))
            
            state_id = self.state_cursor.lastrowid
            
            # Add state embeddings to FAISS indices
            state_fields = {
                'state': (state, state_embedding),
                'reasoning': (reasoning_i, reasoning_embedding),
                'action': (action, action_embedding)
            }
            
            for field, (value, embedding) in state_fields.items():
                if value is not None:
                    if embedding is not None:
                        embedding_array = np.frombuffer(embedding, dtype=np.float32).reshape(1, -1)
                    else:
                        continue
                    self.state_indices[field].add(embedding_array)
                    curr_size = self.state_indices[field].ntotal - 1
                    self.state_id_mappings[field][str(curr_size)] = state_id
                    self._save_index(field, self.state_indices[field], False)
                    
        self.state_conn.commit()

    """ Retrieving from the database """

    def get_rules(self, key_types: List[str], keys: List[str], k: int = 5) -> List[Dict]:
        # If key_types is an empty list, search for all rules
        if not key_types:
            self.rule_cursor.execute("SELECT id, name, rule_content, context, trajectory_ids, state_ids FROM rules")
            rules = self.rule_cursor.fetchall()
            rule_ids = [rule[0] for rule in rules]
        else:
            # Otherwise, use embeddings to search for rules
            rule_ids, rule_distances = self._get_top_k_by_keys(key_types, keys, k)
            self.rule_cursor.execute("SELECT id, name, rule_content, context, trajectory_ids, state_ids FROM rules WHERE id IN ({})".format(', '.join(map(str, rule_ids))))
            rules = self.rule_cursor.fetchall()

        # Turn rules into a list of dictionaries
        rules = [dict(zip(self.rule_cursor.description, rule)) for rule in rules]

        # Let's also fetch corresponding trajectories and states
        for rule in rules:
            trajectory_ids = rule['trajectory_ids']
            state_ids = rule['state_ids']
            for trajectory_id in trajectory_ids:
                self.trajectory_cursor.execute("SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary FROM trajectories WHERE id = ?", (trajectory_id,))
                trajectories = self.trajectory_cursor.fetchall()
                rule['trajectories'] = trajectories
            for state_id in state_ids:
                self.state_cursor.execute("SELECT id, state, reasoning, action, next_state FROM states WHERE id = ?", (state_id,))
                states = self.state_cursor.fetchall()
                rule['states'] = states

        return rules

    def get_similar_sets(self, n, k):
        """Get similar sets of episodes by finding trajectories with similar goals using embeddings"""
        # Get all successful trajectories
        self.trajectory_cursor.execute("""
            SELECT id, goal, category, observations, reasoning, actions, plan
            FROM trajectories
            WHERE json_array_length(rewards) > 0 
            AND rewards LIKE '%1%'
            ORDER BY RANDOM()
            LIMIT ?
        """, (n,))
        base_trajectories = self.trajectory_cursor.fetchall()

        similar_sets = []
        
        # For each base trajectory, find k similar ones using goal embeddings
        for base_traj in base_trajectories:
            base_goal = base_traj[1]
            base_category = base_traj[2]
            
            # Get similar trajectories using get_similar_entries
            success_entries, _ = self.get_similar_entries(
                key_type=['goal', 'category'],
                key=[base_goal, base_category], 
                outcome='winning',
                k=k
            )
            
            # Skip if no similar entries found
            if not success_entries:
                continue
                
            # Format entries into similar set
            similar_set = []
            for entry in success_entries:
                similar_set.append({
                    'goal': entry['goal'],
                    'observation': entry['observation'],
                    'reasoning': entry['reasoning'],
                    'action': entry['action'],
                    'plan': entry['plan']
                })
            
            # Print all goals in the similar set
            '''
            print("Base goal:", base_goal)
            print("\nSimilar goals:")
            for traj in similar_set:
                print(traj['goal'])
            print('--------------------------------')
            input()
            '''
            
            similar_sets.append(similar_set)

        return similar_sets
    
    def get_contrastive_pairs(self):
        """Fetch contrastive pairs of successful and failed episodes for each environment_id"""
        # Get all environment IDs
        self.trajectory_cursor.execute("""
            SELECT DISTINCT environment_id 
            FROM trajectories
        """)
        env_ids = self.trajectory_cursor.fetchall()

        contrastive_pairs = []
        
        # For each environment ID, get one successful and one failed episode
        for env_id in env_ids:
            env_id = env_id[0]
            
            # Get shortest successful episode
            self.trajectory_cursor.execute("""
                SELECT goal, observations, reasoning, actions, plan, LENGTH(observations) as traj_len
                FROM trajectories 
                WHERE environment_id = ? 
                AND json_array_length(rewards) > 0 
                AND rewards LIKE '%1%'
                ORDER BY traj_len ASC
                LIMIT 1
            """, (env_id,))
            success_row = self.trajectory_cursor.fetchone()
            
            # Get shortest failed episode
            self.trajectory_cursor.execute("""
                SELECT goal, observations, reasoning, actions, plan, LENGTH(observations) as traj_len
                FROM trajectories 
                WHERE environment_id = ?
                AND (json_array_length(rewards) = 0 OR rewards NOT LIKE '%1%')
                ORDER BY traj_len ASC
                LIMIT 1
            """, (env_id,))
            failure_row = self.trajectory_cursor.fetchone()
            
            # Only add if we have both success and failure
            if success_row and failure_row:
                success_entry = {
                    'goal': success_row[0],
                    'observation': json.loads(success_row[1]),
                    'reasoning': json.loads(success_row[2]) if success_row[2] else None,
                    'action': json.loads(success_row[3]),
                    'plan': success_row[4]
                }
                
                failure_entry = {
                    'goal': failure_row[0],
                    'observation': json.loads(failure_row[1]),
                    'reasoning': json.loads(failure_row[2]) if failure_row[2] else None,
                    'action': json.loads(failure_row[3]),
                    'plan': failure_row[4]
                }
                
                contrastive_pairs.append((success_entry, failure_entry))
                
        return contrastive_pairs

    def _get_rules_for_id(self, trajectory_id: int) -> List[Dict]:
        """Helper function to get rules for a trajectory ID"""
        self.rule_cursor.execute("""
            SELECT name, rule_content, context FROM rules 
            WHERE trajectory_ids LIKE ?
        """, (f'%{trajectory_id}%',))
        return [{'name': r[0], 'content': r[1], 'context': r[2]} for r in self.rule_cursor.fetchall()]

    def get_similar_entries(self, key_type: Union[str, List[str]], key: Union[str, List[str]], k: int = 5, outcome: str = None, window: int = 1) -> List[Dict]:
        # For environment_id, use exact matching instead of embedding search
        if key_type == 'environment_id':
            cursor = self.trajectory_cursor
            if outcome is not None:
                # Filter by outcome and get k shortest trajectories
                cursor.execute(f"""
                    SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary, LENGTH(observations) as traj_len,
                    CASE 
                        WHEN json_array_length(rewards) > 0 AND CAST(json_extract(rewards, '$[#-1]') AS FLOAT) = 1.0 THEN 1
                        ELSE 0
                    END as success
                    FROM trajectories 
                    WHERE environment_id = "{key}"
                    AND CASE 
                        WHEN json_array_length(rewards) > 0 AND CAST(json_extract(rewards, '$[#-1]') AS FLOAT) = 1.0 THEN 1
                        ELSE 0
                    END = {1 if (outcome == 'success' or outcome == 'winning') else 0}
                    ORDER BY traj_len ASC
                    LIMIT {k}
                """)
            else:
                # Get k shortest trajectories without filtering outcome
                cursor.execute(f"""
                    SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary, LENGTH(observations) as traj_len,
                    CASE 
                        WHEN json_array_length(rewards) > 0 AND CAST(json_extract(rewards, '$[#-1]') AS FLOAT) = 1.0 THEN 1
                        ELSE 0
                    END as success
                    FROM trajectories 
                    WHERE environment_id = "{key}"
                    ORDER BY traj_len ASC
                    LIMIT {k}
                """)
            
            rows = cursor.fetchall()
            
            similar_entries = []
            success_labels = []
            for row in rows:
                rules = self._get_rules_for_id(row[0])

                entry = {
                    'environment_id': row[1],
                    'goal': row[2],
                    'category': row[3],
                    'observation': json.loads(row[4]),
                    'reasoning': json.loads(row[5]) if row[5] else None,
                    'action': json.loads(row[6]),
                    'rewards': json.loads(row[7]),
                    'plan': row[8],
                    'reflection': row[9],
                    'summary': row[10],
                    'rules': rules
                }
                similar_entries.append(entry)
                success_labels.append(1 if max(json.loads(row[7])) == 1 else 0)
            success_entries = [similar_entries[i] for i in range(len(success_labels)) if success_labels[i]]
            failure_entries = [similar_entries[i] for i in range(len(success_labels)) if not success_labels[i]]
            return success_entries, failure_entries

        # Split out key types into trajectory and state-level key lists
        trajectory_keys = {'environment_id', 'goal', 'category', 'plan', 'reflection', 'summary'} 
        trajectory_key_types = [kt for kt in key_type if kt in trajectory_keys]
        state_key_types = [kt for kt in key_type if kt not in trajectory_keys]

        # If we have trajectory-level keys, get top k entries based on them
        # Otherwise, we'll do a state-level search
        trajectory_ids, trajectory_distances = None, None
        state_ids, state_distances = None, None
        if trajectory_key_types:
            # Filter key to trajectory-level keys
            key_filtered = [key[i] for i in range(len(key)) if key_type[i] in trajectory_key_types]
            trajectory_ids, trajectory_distances = self._get_top_k_by_keys(trajectory_key_types, key_filtered, k * (3 if outcome else 1) * (2 if len(state_key_types) > 0 else 1))
            # Filter by outcome if specified
            if outcome:
                trajectory_ids, indices = self._filter_by_outcome(trajectory_ids, outcome)
                trajectory_distances = [trajectory_distances[i] for i in indices]
                # Sort by distances, high to low
                trajectory_ids = [trajectory_ids[i] for i in np.argsort(trajectory_distances)[::-1]]
                trajectory_distances = [trajectory_distances[i] for i in np.argsort(trajectory_distances)[::-1]]
            # Now if state-level keys are present, we'll do a state-level search
            if state_key_types:
                state_distances = []
                state_ids = []
                state_key_embeddings = {kt: self.model.encode([k])[0].reshape(1, -1) 
                                      for kt, k in zip(key_type, key) 
                                      if kt in state_key_types}
                # For each trajectory, retrieve the associated states
                for trajectory_id in trajectory_ids:
                    self.state_cursor.execute("""
                        SELECT id, state, state_embedding, reasoning, reasoning_embedding, 
                               action, action_embedding, next_state 
                        FROM states WHERE trajectory_id = ?""", (trajectory_id,))
                    state_rows = self.state_cursor.fetchall()
                    
                    # Loop through all relevant keys and compute distances
                    state_row_distances = []
                    embedding_col_mapping = {
                        "observation": 2,  # state_embedding column
                        "reasoning": 4,  # reasoning_embedding column 
                        "action": 6,  # action_embedding column
                    }
                    
                    for state_row in state_rows:
                        elem_distances = []
                        for state_key_type in state_key_types:
                            # Get embedding from DB
                            embedding_bytes = state_row[embedding_col_mapping[state_key_type]]
                            if embedding_bytes:
                                state_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                                elem_distances.append(np.dot(state_embedding, state_key_embeddings[state_key_type][0]))
                            else:
                                elem_distances.append(-1.0)
                                
                        # Aggregate distances for each state row
                        state_row_distances.append(np.mean(elem_distances))
                    # Pick the row with the largest distance
                    max_index = np.argmax(state_row_distances)
                    state_ids.append(state_rows[max_index][0])
                    state_distances.append(state_row_distances[max_index])

                # Rerank by trajectory and state distances summed
                summed_distances = [trajectory_distances[i] + state_distances[i] for i in range(len(trajectory_distances))]
                ranked_indices = np.argsort(summed_distances)[::-1]
                trajectory_ids = [trajectory_ids[i] for i in ranked_indices]
                state_ids = [state_ids[i] for i in ranked_indices]
                trajectory_distances = [trajectory_distances[i] for i in ranked_indices]
                state_distances = [state_distances[i] for i in ranked_indices]
        else:
            # If no trajectory-level keys, do a state-level search
            state_ids, state_distances = self._get_top_k_by_keys(state_key_types, key, k * (2 if outcome else 1))
            # Filter by outcome if specified
            if outcome:
                state_ids, indices = self._filter_by_outcome(state_ids, outcome)
                state_distances = [state_distances[i] for i in indices]

        similar_entries = []
        success_labels = []

        outcome_flag = 1 if outcome == "winning" else 0
        
        # If we have trajectory IDs but no state IDs, fetch whole trajectories
        if trajectory_ids and not state_ids:
            for trajectory_id in trajectory_ids:
                self.trajectory_cursor.execute("SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary FROM trajectories WHERE id = ?", (trajectory_id,))
                row = self.trajectory_cursor.fetchone()
                if row:
                    rules = self._get_rules_for_id(row[0])

                    entry = {
                        'environment_id': row[1],
                        'goal': row[2], 
                        'category': row[3],
                        'observation': json.loads(row[4]),
                        'reasoning': json.loads(row[5]) if row[5] else None,
                        'action': json.loads(row[6]),
                        'rewards': json.loads(row[7]),
                        'plan': row[8],
                        'reflection': row[9],
                        'summary': row[10],
                        'rules': rules
                    }
                    
                    if not outcome or ('rewards' in entry and entry['rewards'][-1] == outcome_flag) or ('trajectory' in entry and entry['trajectory']['rewards'][-1] == outcome_flag):
                        similar_entries.append(entry)
                        success_labels.append(entry['rewards'][-1] == 1)

                    if len(similar_entries) >= k:
                        break
        # If we have state IDs, fetch windows around those states
        elif state_ids:
            # First get trajectory IDs for these states
            trajectory_ids = []
            for state_id in state_ids:
                self.state_cursor.execute("SELECT trajectory_id FROM states WHERE id = ?", (state_id,))
                row = self.state_cursor.fetchone()
                if row:
                    trajectory_ids.append(row[0])
            
            # Now fetch the states and surrounding context
            for state_id, trajectory_id in zip(state_ids, trajectory_ids):
                # Get the target state
                self.state_cursor.execute("SELECT id, state, reasoning, action, next_state FROM states WHERE id = ?", (state_id,))
                state_row = self.state_cursor.fetchone()
                
                if state_row:
                    # Get the trajectory info
                    self.trajectory_cursor.execute("SELECT id, environment_id, goal, category, observations, reasoning, actions, rewards, plan, reflection, summary FROM trajectories WHERE id = ?", (trajectory_id,))
                    trajectory_row = self.trajectory_cursor.fetchone()

                    rules = self._get_rules_for_id(trajectory_row[0])

                    # Find the id of the state in the trajectory
                    self.state_cursor.execute("SELECT id FROM states WHERE trajectory_id = ?", (trajectory_id,))
                    state_ids = [row[0] for row in self.state_cursor.fetchall()]
                    state_id_index = state_ids.index(state_id)
                    window_start = max(0, state_id_index - window)
                    window_end = min(len(state_ids) + 1, state_id_index + window + 1) # Fixed off-by-one error

                    # Get the window of states around the target state
                    entry = {
                        'environment_id': trajectory_row[1],
                        'goal': trajectory_row[2],
                        'category': trajectory_row[3],
                        'observation': json.loads(trajectory_row[4])[window_start:window_end],
                        'reasoning': json.loads(trajectory_row[5])[window_start:window_end] if trajectory_row[5] else None,
                        'action': json.loads(trajectory_row[6])[window_start:window_end],
                        'rewards': json.loads(trajectory_row[7])[window_start:window_end],
                        'plan': trajectory_row[8],
                        'reflection': trajectory_row[9],
                        'summary': trajectory_row[10],
                        'rules': rules
                    }

                    rewards = json.loads(trajectory_row[7]) # Don't want to filter for this
                    
                    if not outcome or (rewards[-1] == outcome_flag):
                        similar_entries.append(entry)
                        success_labels.append(rewards[-1] == 1)

                    if len(similar_entries) >= k:
                        break

        # Create two separate lists of entries for success vs failure
        success_entries = [similar_entries[i] for i in range(len(success_labels)) if success_labels[i]]
        failure_entries = [similar_entries[i] for i in range(len(success_labels)) if not success_labels[i]]
                
        return success_entries, failure_entries