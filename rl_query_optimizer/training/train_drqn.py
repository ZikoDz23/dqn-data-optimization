import yaml
import numpy as np
import torch
import os
from ..env.query_env import QueryEnv
from ..agents.drqn import DRQNAgent
from ..env.state_encoder import StateEncoder

def load_config(path="rl_query_optimizer/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_drqn():
    config = load_config()
    
    # Load Queries from disk
    import glob
    query_files = glob.glob("rl_query_optimizer/data/train_queries/*.sql")
    if not query_files:
        print("No queries found! Generating default set...")
        from ..utils.query_generator import QueryGenerator
        gen = QueryGenerator()
        gen.generate_dataset(100, "rl_query_optimizer/data/train_queries")
        query_files = glob.glob("rl_query_optimizer/data/train_queries/*.sql")

    train_queries = []
    for qf in query_files:
        with open(qf, 'r') as f:
            train_queries.append(f.read().strip())

    print(f"Loaded {len(train_queries)} training queries.")
    
    env = QueryEnv(config, queries=train_queries)
    encoder = StateEncoder(config['rl'])
    
    state_dim = encoder.feature_dim
    action_dim = env.action_space.n
    
    agent = DRQNAgent(state_dim, action_dim, config)
    
    num_episodes = config['training']['episodes']
    
    for episode in range(num_episodes):
        raw_state, _ = env.reset()
        state = encoder.encode(env.joined_tables, env.tables, 0)
        
        hidden = None # Reset hidden state at start of episode
        
        done = False
        total_reward = 0
        
        episode_storage = []
        
        while not done:
            action, next_hidden = agent.select_action(state, hidden)
            
            next_raw_state, reward, done, truncated, info = env.step(action)
            
            current_cost = next_raw_state['cost'][0]
            next_state = encoder.encode(env.joined_tables, env.tables, current_cost)
            
            episode_storage.append((state, action, reward, next_state, done))
            
            state = next_state
            hidden = next_hidden
            total_reward += reward
            
        # Store full episode
        agent.memory.push_episode(episode_storage)
        
        loss = agent.update()
        agent.update_epsilon()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        if episode % config['training']['eval_freq'] == 0:
            if not os.path.exists(config['training']['checkpoint_dir']):
                os.makedirs(config['training']['checkpoint_dir'])
            torch.save(agent.policy_net.state_dict(), f"{config['training']['checkpoint_dir']}/drqn_{episode}.pt")

if __name__ == "__main__":
    train_drqn()
