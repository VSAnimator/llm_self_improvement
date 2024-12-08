import os
import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Optional
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
        
        # Create trajectory table
        self.trajectory_cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY,
                environment_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                observations TEXT NOT NULL,
                reasoning TEXT,
                actions TEXT NOT NULL,
                rewards TEXT NOT NULL,
                plan TEXT,
                reflexion TEXT,
                summary TEXT
            )
        """)
        
        # Create state table
        self.state_cursor.execute("""
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY,
                trajectory_id INTEGER NOT NULL,
                state TEXT NOT NULL,
                reasoning TEXT,
                action TEXT NOT NULL,
                next_state TEXT NOT NULL,
                FOREIGN KEY(trajectory_id) REFERENCES trajectories(id)
            )
        """)
        
        self.trajectory_conn.commit()
        self.state_conn.commit()
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS indices
        self.index_path = os.path.dirname(db_path)
        
        # Trajectory-level indices
        self.trajectory_indices = {
            'environment_id': self._load_or_create_index('environment_id'),
            'goal': self._load_or_create_index('goal'),
            'plan': self._load_or_create_index('plan'),
            'reflexion': self._load_or_create_index('reflexion'),
            'summary': self._load_or_create_index('summary')
        }
        
        # State-level indices
        self.state_indices = {
            'state': self._load_or_create_index('state'),
            'reasoning': self._load_or_create_index('reasoning'),
            'action': self._load_or_create_index('action')
        }
        
        # Load id mappings
        self.trajectory_id_mappings = {}
        self.state_id_mappings = {}
        
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

    def store_episode(self, environment_id: str, goal: str, observations: List[str], reasoning: List[str], 
                     actions: List[str], rewards: List[float], plan: Optional[str],
                     reflexion: Optional[str], summary: Optional[str]):
        """Store an episode in both trajectory and state databases"""
        # Store trajectory
        observations_str = json.dumps([observation.structured for observation in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else None
        actions_str = json.dumps([action.text for action in actions])
        rewards_str = json.dumps(rewards)
        
        self.trajectory_cursor.execute("""
            INSERT INTO trajectories (environment_id, goal, observations, reasoning, actions, rewards, plan, reflexion, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (environment_id, goal, observations_str, reasoning_str, actions_str, rewards_str, plan, reflexion, summary))
        trajectory_id = self.trajectory_cursor.lastrowid
        self.trajectory_conn.commit()
        
        # Store individual states
        for i in range(len(observations) - 1):
            self.state_cursor.execute("""
                INSERT INTO states (trajectory_id, state, reasoning, action, next_state)
                VALUES (?, ?, ?, ?, ?)
            """, (trajectory_id, 
                 observations[i].structured,
                 reasoning[i] if reasoning else None,
                 actions[i].text,
                 observations[i+1].structured))
        self.state_conn.commit()
        
        # Add trajectory embeddings
        trajectory_fields = {
            'environment_id': environment_id,
            'goal': goal,
            'plan': plan,
            'reflexion': reflexion,
            'summary': summary
        }
        
        for field, value in trajectory_fields.items():
            if value is not None:
                embedding = self.model.encode([value])[0].reshape(1, -1)
                self.trajectory_indices[field].add(embedding)
                curr_size = self.trajectory_indices[field].ntotal - 1
                self.trajectory_id_mappings[field][str(curr_size)] = trajectory_id
                self._save_index(field, self.trajectory_indices[field], True)
        
        # Add state embeddings
        for i in range(len(observations) - 1):
            state_fields = {
                'state': observations[i].structured,
                'reasoning': reasoning[i] if reasoning else None,
                'action': actions[i].text
            }
            
            for field, value in state_fields.items():
                if value is not None:
                    embedding = self.model.encode([value])[0].reshape(1, -1)
                    self.state_indices[field].add(embedding)
                    curr_size = self.state_indices[field].ntotal - 1
                    self.state_id_mappings[field][str(curr_size)] = trajectory_id
                    self._save_index(field, self.state_indices[field], False)

    def get_similar_entries(self, key_type: str, key: str, k: int = 5, outcome: str = None) -> List[Dict]:
        """Get similar entries based on key_type and key"""
        key_embedding = self.model.encode([key])[0].reshape(1, -1)
        
        # Determine if this is a trajectory or state level search
        trajectory_keys = {'environment_id', 'goal', 'plan', 'reflexion', 'summary'}
        is_trajectory = key_type in trajectory_keys
        
        indices = self.trajectory_indices if is_trajectory else self.state_indices
        mappings = self.trajectory_id_mappings if is_trajectory else self.state_id_mappings
        cursor = self.trajectory_cursor if is_trajectory else self.state_cursor
        
        if outcome:
            D, I = indices[key_type].search(key_embedding, k * 10) # Buffer for losing episodes fetched as well
        else:
            D, I = indices[key_type].search(key_embedding, k)
        
        # Filter invalid results
        D = [d for d, i in zip(D[0], I[0]) if i != -1]
        I = [i for i in I[0] if i != -1]
        
        if not I:
            return []
            
        entry_ids = [mappings[key_type][str(i)] for i in I]
        
        similar_entries = []
        success_labels = []
        for entry_id in entry_ids:
            if is_trajectory:
                cursor.execute("SELECT * FROM trajectories WHERE id = ?", (entry_id,))
                row = cursor.fetchone()
                if row:
                    entry = {
                        'environment_id': row[1],
                        'goal': row[2],
                        'observation': json.loads(row[3]),
                        'reasoning': json.loads(row[4]) if row[4] else None,
                        'action': json.loads(row[5]),
                        'rewards': json.loads(row[6]),
                        'plan': row[7],
                        'reflexion': row[8],
                        'summary': row[9]
                    }
            else:
                cursor.execute("SELECT * FROM states WHERE id = ?", (entry_id,))
                row = cursor.fetchone()
                if row:
                    # Also fetch the entire trajectory for this state
                    self.trajectory_cursor.execute("SELECT * FROM trajectories WHERE id = ?", (row[1],))
                    trajectory_row = self.trajectory_cursor.fetchone()
                    entry = {
                        'trajectory_id': row[1],
                        'state': row[2],
                        'reasoning': row[3],
                        'action': row[4],
                        'next_state': row[5],
                        'trajectory': {
                            'environment_id': trajectory_row[1],
                            'goal': trajectory_row[2],
                            'observation': json.loads(trajectory_row[3]),
                            'reasoning': json.loads(trajectory_row[4]) if trajectory_row[4] else None,
                            'action': json.loads(trajectory_row[5]),
                            'rewards': json.loads(trajectory_row[6]),
                            'plan': trajectory_row[7],
                            'reflexion': trajectory_row[8],
                            'summary': trajectory_row[9]
                        }
                    }

            outcome_flag = 1 if outcome == "winning" else 0
            
            if not outcome or ('rewards' in entry and entry['rewards'][-1] == outcome_flag) or ('trajectory' in entry and entry['trajectory']['rewards'][-1] == outcome_flag):
                similar_entries.append(entry)
                success_labels.append(entry['rewards'][-1] == 1)

            if len(similar_entries) >= k:
                break

        # Create two separate lists of entries for success vs failure
        success_entries = [similar_entries[i] for i in range(len(success_labels)) if success_labels[i]]
        failure_entries = [similar_entries[i] for i in range(len(success_labels)) if not success_labels[i]]
                
        return success_entries, failure_entries