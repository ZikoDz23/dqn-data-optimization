import numpy as np
import random
from collections import deque

class SequenceReplayBuffer:
    def __init__(self, capacity, max_seq_len=20):
        self.capacity = capacity
        # Buffer stores episodes: list of transitions
        self.buffer = deque(maxlen=capacity)
        self.max_seq_len = max_seq_len

    def push_episode(self, episode):
        """
        episode: list of (state, action, reward, next_state, done) tuples
        """
        self.buffer.append(episode)

    def sample(self, batch_size):
        # DRQN sampling strategy:
        # 1. Sample 'batch_size' episodes
        # 2. Extract sequences from them (pad if necessary)
        # Here we assume we sample full episodes for query optimization since they are short.
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch_episodes = [self.buffer[i] for i in indices]
        
        # Prepare padding
        # We need to pad to the max length in the batch or global max
        max_len = max(len(ep) for ep in batch_episodes)
        
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        mask_batch = [] # To mask out padding
        
        first_state_shape = batch_episodes[0][0][0].shape
        
        for ep in batch_episodes:
            seq_len = len(ep)
            # Unzip
            s, a, r, ns, d = zip(*ep)
            
            # Pad
            pad_len = max_len - seq_len
            
            s_padded = np.array(s)
            a_padded = np.array(a)
            r_padded = np.array(r)
            ns_padded = np.array(ns)
            d_padded = np.array(d)
            
            if pad_len > 0:
                s_zero = np.zeros((pad_len, *first_state_shape))
                s_padded = np.concatenate([s_padded, s_zero])
                
                ns_zero = np.zeros((pad_len, *first_state_shape))
                ns_padded = np.concatenate([ns_padded, ns_zero])
                
                a_padded = np.concatenate([a_padded, np.zeros(pad_len)])
                r_padded = np.concatenate([r_padded, np.zeros(pad_len)])
                d_padded = np.concatenate([d_padded, np.ones(pad_len)]) # Terminal padding
            
            state_batch.append(s_padded)
            action_batch.append(a_padded)
            reward_batch.append(r_padded)
            next_state_batch.append(ns_padded)
            done_batch.append(d_padded)
            
            # Mask: 1 for valid, 0 for padding
            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
            mask_batch.append(mask)

        return (np.array(state_batch), 
                np.array(action_batch), 
                np.array(reward_batch), 
                np.array(next_state_batch), 
                np.array(done_batch),
                np.array(mask_batch))

    def __len__(self):
        return len(self.buffer)
