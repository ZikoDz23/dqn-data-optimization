import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DRQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DRQNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # Input to LSTM: [batch, seq, features]
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_dim)
        # hidden is (h_0, c_0)
        out, hidden = self.lstm(x, hidden)
        
        # Take the output of the last time step for Q-value
        # Actually for DRQN we might want Q-values for all steps (if measuring loss on all steps)
        # or just the last step.
        # Usually standard DRQN trains on random sequences and minimizes loss on them.
        
        # If we just want the last Q-values:
        # last_out = out[:, -1, :] 
        # q = self.fc(last_out)
        
        # If we return Q-values for the whole sequence:
        q_seq = self.fc(out) 
        
        return q_seq, hidden
