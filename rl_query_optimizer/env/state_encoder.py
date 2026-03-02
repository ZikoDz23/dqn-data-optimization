import numpy as np
import torch

class StateEncoder:
    def __init__(self, config):
        self.max_tables = config.get('max_tables', 20)
        self.feature_dim = self.max_tables * 2 + 1 # join_mask + remaining_mask + cost

    def encode(self, joined_tables, all_tables, current_cost):
        """
        Encodes the current state into a flat vector.
        
        Args:
            joined_tables (set): set of table names/aliases currently joined.
            all_tables (list): list of all table names in the current query.
            current_cost (float): current estimated cost/runtime.
            
        Returns:
            np.array: shape (feature_dim,)
        """
        # Create masks
        join_mask = np.zeros(self.max_tables, dtype=np.float32)
        remaining_mask = np.zeros(self.max_tables, dtype=np.float32)
        
        # Map table names to indices (this mapping must be consistent per query)
        # In a generalized agent, we might map strictly by index in 'all_tables'.
        # Since 'all_tables' changes per query, the position relates to the query-specific table list.
        # This is strictly for the current episode.
        
        for i, table in enumerate(all_tables):
            if i >= self.max_tables: break
            
            if table in joined_tables:
                join_mask[i] = 1.0
            else:
                remaining_mask[i] = 1.0
                
        # Scale cost (log scale is usually better for large values)
        scaled_cost = np.log1p(current_cost) if current_cost > 0 else 0.0
        
        # Concatenate
        state_vec = np.concatenate([join_mask, remaining_mask, [scaled_cost]])
        return state_vec

    def get_input_shape(self):
        return (self.feature_dim,)
