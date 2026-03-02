import time
import yaml
import torch
from rl_query_optimizer.agents.dqn import DQNAgent
from rl_query_optimizer.agents.drqn import DRQNAgent

def load_config():
    with open("rl_query_optimizer/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_instantiation():
    print("Loading config...")
    config = load_config()
    print(f"Config device: {config['rl'].get('device')}")
    
    start = time.time()
    print("Instantiating DQNAgent...")
    agent = DQNAgent(state_dim=10, action_dim=5, config=config)
    end = time.time()
    print(f"DQNAgent instantiation took {end - start:.4f}s")
    print(f"Agent device: {agent.device}")
    
    start = time.time()
    print("Instantiating DRQNAgent...")
    agent_drqn = DRQNAgent(state_dim=10, action_dim=5, config=config)
    end = time.time()
    print(f"DRQNAgent instantiation took {end - start:.4f}s")
    print(f"Agent device: {agent_drqn.device}")

if __name__ == "__main__":
    test_instantiation()
