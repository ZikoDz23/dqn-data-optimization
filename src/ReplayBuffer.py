from collections import deque
import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    Buffer with Prioritized Experience Replay
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done, td_error):
        """
        Add an experience to the buffer
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append((abs(td_error) + 1e-5) ** self.alpha)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples. Has {len(self.buffer)}, needs {batch_size}")

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        batch = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        return (states, actions, rewards, next_states, dones), weights, indices

    def update_priorities(self, indices, errors):
        """
        Update priorities
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
        self.max_priority = max(self.priorities)

    def size(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self.buffer) >= batch_size
