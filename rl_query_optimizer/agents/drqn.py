import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from .networks import DRQNetwork
from ..replay_buffer.sequence_buffer import SequenceReplayBuffer

class DRQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config['rl']
        self.device = torch.device(self.config.get('device', 'cpu'))

        self.policy_net = DRQNetwork(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        self.target_net = DRQNetwork(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=float(self.config['learning_rate']))
        self.memory = SequenceReplayBuffer(self.config['buffer_capacity'])
        
        self.epsilon = self.config['epsilon_start']
        self.steps_done = 0

    def select_action(self, state, hidden, eval_mode=False):
        # DRQN action selection usually involves the history. 
        # But here 'state' is the current observation.
        # 'hidden' is the hidden state (h, c) passed from previous step.
        
        if not eval_mode and random.random() < self.epsilon:
            # Still need to forward to get next hidden state for consistency
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, feat)
                _, next_hidden = self.policy_net(state_t, hidden)
            return random.randrange(self.action_dim), next_hidden
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            # Forward one step
            q_values, next_hidden = self.policy_net(state_t, hidden)
            # q_values shape: (1, 1, actions)
            return q_values.squeeze().argmax().item(), next_hidden

    def update_epsilon(self):
        self.steps_done += 1
        decay = self.config['epsilon_decay']
        start = self.config['epsilon_start']
        end = self.config['epsilon_end']
        self.epsilon = end + (start - end) * np.exp(-1. * self.steps_done / decay)

    def update(self):
        if len(self.memory) < self.config['batch_size']:
            return

        # Sample sequences
        state, action, reward, next_state, done, mask = self.memory.sample(self.config['batch_size'])
        
        state = torch.FloatTensor(state).to(self.device) # (B, Seq, Feat)
        action = torch.LongTensor(action).unsqueeze(-1).to(self.device) # (B, Seq, 1)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(self.device)
        mask = torch.FloatTensor(mask).unsqueeze(-1).to(self.device)

        # Get Q-values for the whole sequence
        q_values_seq, _ = self.policy_net(state)
        
        # Select Q-values for the actions taken
        q_values = q_values_seq.gather(-1, action)

        # Target values
        with torch.no_grad():
            next_q_values_seq, _ = self.target_net(next_state)
            next_max_q = next_q_values_seq.max(-1)[0].unsqueeze(-1)
            target_q_values = reward + (1 - done) * self.config['gamma'] * next_max_q

        # Masked loss (don't train on padding)
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction='none')
        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
