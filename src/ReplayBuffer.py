from collections import deque
import numpy as np
import random

class PrioritizedReplayBuffer:
    """
    Buffer avec Prioritized Experience Replay
    """
    def __init__(self, capacity, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done, error=1.0):
        """
        Ajoute une expérience au buffer
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(error ** self.alpha)

    def sample(self, batch_size, beta=0.4):
        """
        Echantillonne un batch d'expériences
        """
        if len(self.buffer) == 0:
            raise ValueError("Le buffer est vide. Impossible d'échantillonner.")

        # Normalisation des priorités
        priorities = np.array(self.priorities, dtype=float)
        priorities_sum = priorities.sum()
        if priorities_sum == 0:
            priorities = np.ones_like(priorities) / len(priorities)  
        else:
            priorities = priorities / priorities_sum  

       
        indices = np.random.choice(len(self.buffer), batch_size, p=priorities)
        batch = [self.buffer[i] for i in indices]

       
        weights = (len(self.buffer) * priorities[indices]) ** (-beta)
        weights /= weights.max()  

        return batch, weights, indices

    def update_priorities(self, indices, errors):
        """
        Met à jour les priorités des expériences.
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (error + 1e-5) ** self.alpha

    def size(self):
        """
        Retourne la taille actuelle du buffer.
        """
        return len(self.buffer)