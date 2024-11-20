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
        
        # Initialize SQLite
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Create episodes table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
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
        self.conn.commit()
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS indices for each field
        self.index_path = os.path.dirname(db_path)
        self.field_indices = {
            'environment_id': self._load_or_create_index('environment_id'),
            'goal': self._load_or_create_index('goal'),
            'observations': self._load_or_create_index('observations'), 
            'reasoning': self._load_or_create_index('reasoning'),
            'actions': self._load_or_create_index('actions'),
            'plan': self._load_or_create_index('plan'),
            'reflexion': self._load_or_create_index('reflexion'),
            'summary': self._load_or_create_index('summary')
        }
        
        # Load id mappings
        self.id_mappings = {}
        for field in self.field_indices.keys():
            mapping_path = os.path.join(self.index_path, f"{field}_id_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.id_mappings[field] = json.load(f)
            else:
                self.id_mappings[field] = {}

    def _load_or_create_index(self, field: str) -> faiss.IndexFlatIP:
        """Load existing FAISS index or create new one"""
        index_path = os.path.join(self.index_path, f"{field}.index")
        if os.path.exists(index_path):
            return faiss.read_index(index_path)
        return faiss.IndexFlatIP(384)  # Using sentence-transformer embedding dimension

    def _save_index(self, field: str, index: faiss.IndexFlatIP):
        """Save FAISS index to disk"""
        index_path = os.path.join(self.index_path, f"{field}.index")
        faiss.write_index(index, index_path)
        
        # Save id mapping
        mapping_path = os.path.join(self.index_path, f"{field}_id_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(self.id_mappings[field], f)

    def store_episode(self, environment_id: str, goal: str, observations: List[str], reasoning: List[str], 
                     actions: List[str], rewards: List[float], plan: Optional[str],
                     reflexion: Optional[str], summary: Optional[str]):
        """Store an episode in the database"""
        # Convert lists to strings for storage
        observations_str = json.dumps([observation.structured for observation in observations])
        reasoning_str = json.dumps(reasoning) if reasoning else None
        actions_str = json.dumps([action.text for action in actions])
        rewards_str = json.dumps(rewards)
        
        # Insert into SQLite
        self.cursor.execute("""
            INSERT INTO episodes (environment_id, goal, observations, reasoning, actions, rewards, plan, reflexion, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (environment_id, goal, observations_str, reasoning_str, actions_str, rewards_str, plan, reflexion, summary))
        episode_id = self.cursor.lastrowid
        self.conn.commit()
        
        # Add embeddings to FAISS indices
        fields = {
            'environment_id': environment_id,
            'goal': goal,
            'observations': observations_str,
            'reasoning': reasoning_str,
            'actions': actions_str,
            'plan': plan,
            'reflexion': reflexion,
            'summary': summary
        }
        
        for field, value in fields.items():
            if value is not None:
                embedding = self.model.encode([value])[0]
                embedding = embedding.reshape(1, -1)
                
                # Add to index
                self.field_indices[field].add(embedding)
                
                # Update id mapping
                curr_size = self.field_indices[field].ntotal - 1
                self.id_mappings[field][str(curr_size)] = episode_id
                
                # Save updated index and mapping
                self._save_index(field, self.field_indices[field])

    def get_similar_entries(self, key_type: str, key: str, k: int = 5) -> List[Dict]:
        """Get similar entries from the database based on key_type and key"""
        # Debugging
        print("Key type", key_type)
        print("Key", key)
        input("Press Enter to continue")

        # Get all environment ids
        self.cursor.execute("SELECT DISTINCT environment_id FROM episodes")
        environment_ids = [row[0] for row in self.cursor.fetchall()]
        print("Environment ids", environment_ids)
        input("Press Enter to continue")

        # Get embedding for key
        key_embedding = self.model.encode([key])[0].reshape(1, -1)
        
        # Search FAISS index
        D, I = self.field_indices[key_type].search(key_embedding, k)

        print("D", D)
        print("I", I)
        input("Press Enter to continue")

        # Shorten the lists to remove anything where I is -1
        D = [d for d, i in zip(D[0], I[0]) if i != -1]
        I = [i for i in I[0] if i != -1]

        # Return empty list if I is empty
        if not I:
            return []
        
        # Get episode IDs from mappings
        episode_ids = [self.id_mappings[key_type][str(i)] for i in I]

        print("Episode ids", episode_ids)
        input("Press Enter to continue")
        
        # Retrieve full entries from SQLite
        similar_entries = []
        for episode_id in episode_ids:
            self.cursor.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
            row = self.cursor.fetchone()
            print("Row", row)
            input("Press Enter to continue")
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
                similar_entries.append(entry)
                
        return similar_entries




