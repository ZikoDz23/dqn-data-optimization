import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ..utils.query_graph import QueryGraph
from .cost_interface import CostInterface

class QueryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, queries=None):
        super(QueryEnv, self).__init__()
        self.config = config
        self.cost_interface = CostInterface(config['database'])
        self.queries = queries if queries else []
        self.current_query = None
        self.query_graph = None
        
        # Max tables assumption for fixed observation space
        self.max_tables = 20
        
        # Observation Space:
        # [joined_mask (N), adjacency_flattened (N*N)? or just focus on mask + cost]
        # User suggested: [joined_relations_mask, remaining_relations_mask, estimated_size, cumulative_cost]
        # Let's do:
        # - Joined Mask (N)
        # - Current Cost (1)
        # - Last Reward (1)
        self.observation_space = spaces.Dict({
            "mask": spaces.Box(low=0, high=1, shape=(self.max_tables,), dtype=np.int8),
            "cost": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        # Action Space:
        # Select a join pair.
        # Since number of tables is dynamic, we can model action as index into valid_joins list?
        # Or a fixed discrete space N*(N-1)/2. 
        # User suggested: N * (N-1) / 2
        num_actions = self.max_tables * (self.max_tables - 1) // 2
        self.action_space = spaces.Discrete(num_actions)

        self.tables = []
        self.joined_tables = set()

    def reset(self, seed=None, query=None):
        super().reset(seed=seed)
        
        if query:
            self.current_query = query
        elif self.queries:
            # Pick strict or random?
            self.current_query = self.queries[0] # simplistic
            
        # Parse query and build graph (mocked for now)
        # self.tables = extract_tables(self.current_query)
        self.tables = ["t1", "t2", "t3"] # Mock
        self.joined_tables = set()
        
        # Initial State
        return self._get_observation(), {}

    def step(self, action):
        # Action map: index -> (i, j)
        pair = self._action_to_pair(action)
        t1_idx, t2_idx = pair
        
        # Check validity
        if t1_idx >= len(self.tables) or t2_idx >= len(self.tables):
            return self._get_observation(), -10.0, False, False, {"error": "Invalid index"}
            
        t1 = self.tables[t1_idx]
        t2 = self.tables[t2_idx]
        
        # Check if already joined or joinable
        # For a left-deep tree, we usually join a new table to the current result.
        # Impl: Mark t1, t2 as joined.
        
        self.joined_tables.add(t1)
        self.joined_tables.add(t2)
        
        # Reward
        cost = self.cost_interface.estimate_cost(list(self.joined_tables), self.current_query)
        reward = -cost 
        
        done = len(self.joined_tables) == len(self.tables)
        
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        mask = np.zeros(self.max_tables, dtype=np.int8)
        for i, t in enumerate(self.tables):
            if t in self.joined_tables:
                mask[i] = 1
        
        # Placeholder cost
        return {
            "mask": mask,
            "cost": np.array([0.0], dtype=np.float32)
        }

    def _action_to_pair(self, action):
        # Convert scalar action to (i, j)
        # k = (2n - 1 - i) * i / 2 + j - i - 1  ... classic upper triangular mapping
        # Naive: iterate
        k = 0
        for i in range(self.max_tables):
            for j in range(i + 1, self.max_tables):
                if k == action:
                    return (i, j)
                k += 1
        return (0, 0)
