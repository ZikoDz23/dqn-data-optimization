import yaml
import numpy as np
import torch
import os
from ..env.query_env import QueryEnv
from ..agents.dqn import DQNAgent
from ..env.state_encoder import StateEncoder

def load_config(path="rl_query_optimizer/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_dqn():
    config = load_config()
    
    # Load Queries from disk
    import glob
    query_files = glob.glob("rl_query_optimizer/data/train_queries/*.sql")
    if not query_files:
        print("No queries found! Generating default set...")
        # Fallback or auto-generate
        from ..utils.query_generator import QueryGenerator
        gen = QueryGenerator()
        gen.generate_dataset(100, "rl_query_optimizer/data/train_queries")
        query_files = glob.glob("rl_query_optimizer/data/train_queries/*.sql")
        
    train_queries = []
    for qf in query_files:
        with open(qf, 'r') as f:
            train_queries.append(f.read().strip())
            
    print(f"Loaded {len(train_queries)} training queries.") 
    
    print("Initializing Environment...")
    env = QueryEnv(config, queries=train_queries)
    encoder = StateEncoder(config['rl'])
    
    state_dim = encoder.feature_dim
    action_dim = env.action_space.n
    
    print("Initializing Agent...")
    agent = DQNAgent(state_dim, action_dim, config)
    
    num_episodes = config['training']['episodes']
    print(f"Starting training on {config['rl']['device']} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        if episode == 0:
            print("Starting Episode 0...")
        # Curriculum: Switch queries based on episode progress
        # if episode > 1000: env.queries = hard_queries
        
        raw_state, _ = env.reset()
        # Encode state
        state = encoder.encode(env.joined_tables, env.tables, 0)
        
        done = False
        total_reward = 0
        step_count = 0
        max_steps = config['training'].get('max_steps_per_episode', 20)
        
        while not done and step_count < max_steps:
            step_count += 1
            if episode == 0 and step_count % 10 == 0:
                 print(f"Ep 0 Step {step_count}...")
            action = agent.select_action(state)
            
            next_raw_state, reward, done, truncated, info = env.step(action)
            
            # Encode next state
            current_cost = next_raw_state['cost'][0] # Approx
            next_state = encoder.encode(env.joined_tables, env.tables, current_cost)
            
            agent.memory.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            loss = agent.update()
            
        agent.update_epsilon()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
        if episode % config['training']['eval_freq'] == 0:
            # Save model
            if not os.path.exists(config['training']['checkpoint_dir']):
                os.makedirs(config['training']['checkpoint_dir'])
            torch.save(agent.policy_net.state_dict(), f"{config['training']['checkpoint_dir']}/dqn_{episode}.pt")

if __name__ == "__main__":
    train_dqn()
