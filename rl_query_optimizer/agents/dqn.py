import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .networks import QNetwork
from ..replay_buffer.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config['rl']
        self.device = torch.device(self.config.get('device', 'cpu'))

        self.policy_net = QNetwork(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(self.config['learning_rate']))
        self.memory = ReplayBuffer(self.config['buffer_capacity'])
        
        self.epsilon = self.config['epsilon_start']
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        # State is numpy array
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def update_epsilon(self):
        self.steps_done += 1
        decay = self.config['epsilon_decay']
        start = self.config['epsilon_start']
        end = self.config['epsilon_end']
        self.epsilon = end + (start - end) * np.exp(-1. * self.steps_done / decay)

    def update(self):
        if len(self.memory) < self.config['batch_size']:
            return

        state, action, reward, next_state, done = self.memory.sample(self.config['batch_size'])
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(state).gather(1, action)

        # V(s') = max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q_values = reward + (1 - done) * self.config['gamma'] * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
