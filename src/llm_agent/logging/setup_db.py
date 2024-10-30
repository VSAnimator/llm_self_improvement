import os
import sqlite3
import faiss
import numpy as np
import dataset
from typing import Dict, List, Optional, Tuple
from logging import getLogger

logger = getLogger(__name__)

class LoggingDatabases:
    def __init__(self, env_name: str, state_dim: int = 1536, trajectory_dim: int = 1536):
        """Initialize logging databases for both vector and structured storage
        
        Args:
            env_name: Name of environment to create DBs for
            state_dim: Dimension of state vectors (default 1536 for GPT embeddings)
            trajectory_dim: Dimension of trajectory vectors (default 1536 for GPT embeddings)
        """
        self.env_name = env_name
        self.state_dim = state_dim
        self.trajectory_dim = trajectory_dim
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize FAISS vector stores
        self.state_index = faiss.IndexFlatL2(state_dim)
        self.trajectory_index = faiss.IndexFlatL2(trajectory_dim)
        
        self.state_index_path = f"logs/{env_name}_state_vectors.index"
        self.trajectory_index_path = f"logs/{env_name}_trajectory_vectors.index"
        
        # Load existing vector indices if they exist
        self._load_vector_index(self.state_index_path, 'state', state_dim)
        self._load_vector_index(self.trajectory_index_path, 'trajectory', trajectory_dim)
        
        # Initialize SQLite database using dataset
        self.db_path = f"logs/{env_name}.db"
        self.db = dataset.connect(f'sqlite:///{self.db_path}')
        
        # Create tables if they don't exist
        self.states_table = self.db.create_table('states', primary_id='id')
        self.actions_table = self.db.create_table('actions', primary_id='id')
        self.trajectories_table = self.db.create_table('trajectories', primary_id='id')
        
        # Create indices for faster querying
        if 'states' in self.db:
            self.states_table.create_index(['task'])
        if 'actions' in self.db:
            self.actions_table.create_index(['state_id'])
        if 'trajectories' in self.db:
            self.trajectories_table.create_index(['task'])
        
        logger.info(f"Initialized logging databases for environment: {env_name}")
    
    def _load_vector_index(self, path: str, index_type: str, dim: int):
        """Helper to load vector index from disk"""
        if os.path.exists(path):
            try:
                if index_type == 'state':
                    self.state_index = faiss.read_index(path)
                else:
                    self.trajectory_index = faiss.read_index(path)
                logger.info(f"Loaded existing {index_type} vector index from {path}")
            except Exception as e:
                logger.error(f"Failed to load {index_type} vector index: {str(e)}")
                if index_type == 'state':
                    self.state_index = faiss.IndexFlatL2(dim)
                else:
                    self.trajectory_index = faiss.IndexFlatL2(dim)
    
    def add_state(self, state_vector: np.ndarray, state_data: Dict, task: str) -> int:
        """Add state to both vector and structured storage
        
        Args:
            state_vector: Vector representation of state for similarity search
            state_data: Dictionary of structured state data
            task: Task label for the state
            
        Returns:
            ID of inserted state record
        """
        # Add to FAISS
        self.state_index.add(state_vector.reshape(1, -1))
        vector_id = self.state_index.ntotal - 1
        
        # Add to SQLite
        state_data.update({
            'vector_id': vector_id,
            'task': task
        })
        state_id = self.states_table.insert(state_data)
        
        return state_id
    
    def add_action(self, state_id: int, action_data: Dict) -> int:
        """Add action to structured storage
        
        Args:
            state_id: ID of the state this action was taken in
            action_data: Dictionary of action data
            
        Returns:
            ID of inserted action record
        """
        action_data['state_id'] = state_id
        return self.actions_table.insert(action_data)
    
    def add_trajectory(self, 
                      trajectory_vector: np.ndarray, 
                      trajectory_data: Dict,
                      state_ids: List[int],
                      task: str) -> int:
        """Add trajectory to both vector and structured storage
        
        Args:
            trajectory_vector: Vector representation of full trajectory
            trajectory_data: Dictionary of trajectory metadata
            state_ids: List of state IDs in this trajectory
            task: Task label for the trajectory
            
        Returns:
            ID of inserted trajectory record
        """
        # Add to FAISS
        self.trajectory_index.add(trajectory_vector.reshape(1, -1))
        vector_id = self.trajectory_index.ntotal - 1
        
        # Add to SQLite
        trajectory_data.update({
            'vector_id': vector_id,
            'state_ids': ','.join(map(str, state_ids)),
            'task': task
        })
        return self.trajectories_table.insert(trajectory_data)
    
    def get_state(self, state_id):
        """Retrieve a state by its ID."""
        state = self.states_table.find_one(id=state_id)
        if state is None:
            raise KeyError(f"No state found with ID {state_id}")
        return state
    
    def get_states_for_trajectory(self, trajectory_id):
        """
        Retrieve all states associated with a given trajectory.
        
        Args:
            trajectory_id (int): The ID of the trajectory
            
        Returns:
            list: List of state dictionaries associated with the trajectory
        """
        trajectory = self.trajectories_table.find_one(id=trajectory_id)
        if trajectory is None:
            raise KeyError(f"No trajectory found with ID {trajectory_id}")
            
        # Get list of state IDs from comma-separated string
        state_ids = [int(id) for id in trajectory['state_ids'].split(',')]
        
        # Retrieve all states
        states = []
        for state_id in state_ids:
            state = self.get_state(state_id)
            states.append(state)
            
        return states
    
    def get_action_for_state(self, state_id: int) -> Optional[Dict]:
        """Get the action taken in a given state
        
        Args:
            state_id: ID of the state
            
        Returns:
            Action data dictionary or None if not found
        """
        return self.actions_table.find_one(state_id=state_id)
    
    def search_similar_states(self, query_vector: np.ndarray, k: int = 5) -> Dict:
        """Search for similar states using vector similarity
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            
        Returns:
            Dictionary with distances and state IDs
        """
        distances, vector_indices = self.state_index.search(query_vector.reshape(1, -1), k)
        
        # Map vector IDs to state IDs
        state_ids = []
        valid_distances = []
        for vid, dist in zip(vector_indices[0], distances[0]):
            if vid != -1:  # Skip invalid vector IDs
                state = self.states_table.find_one(vector_id=vid)
                if state:
                    state_ids.append(state['id'])
                    valid_distances.append(dist)
        
        return {
            'distances': valid_distances,
            'state_ids': state_ids  # Now returning actual state IDs instead of vector IDs
        }
    
    def search_similar_trajectories(self, query_vector: np.ndarray, k: int = 5) -> Dict:
        """Search for similar trajectories using vector similarity
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            
        Returns:
            Dictionary with distances and trajectory IDs
        """
        distances, indices = self.trajectory_index.search(query_vector.reshape(1, -1), k)
        return {
            'distances': distances[0].tolist(),
            'vector_ids': indices[0].tolist()
        }
    
    def search_by_task(self, task: str, table: str = 'states') -> List[Dict]:
        """Search for records by task label
        
        Args:
            task: Task label to search for
            table: Which table to search ('states' or 'trajectories')
            
        Returns:
            List of matching records
        """
        if table == 'states':
            return list(self.states_table.find(task=task))
        elif table == 'trajectories':
            return list(self.trajectories_table.find(task=task))
        else:
            raise ValueError("Table must be either 'states' or 'trajectories'")
    
    def save(self):
        """Save vector indices to disk"""
        try:
            faiss.write_index(self.state_index, self.state_index_path)
            faiss.write_index(self.trajectory_index, self.trajectory_index_path)
            logger.info(f"Saved vector indices to disk")
        except Exception as e:
            logger.error(f"Failed to save vector indices: {str(e)}")
    
    def close(self):
        """Close database connections and save state"""
        self.save()
        self.db.close()

