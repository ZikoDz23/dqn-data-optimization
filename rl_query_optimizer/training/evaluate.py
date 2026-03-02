import yaml
import numpy as np
import torch
from ..env.query_env import QueryEnv
from ..agents.dqn import DQNAgent
from ..agents.drqn import DRQNAgent
from ..env.state_encoder import StateEncoder

def load_config(path="rl_query_optimizer/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_evaluation(agent, env, encoder, episodes=10, mode="agent"):
    total_rewards = []
    execution_times = []
    
    for i in range(episodes):
        raw_state, _ = env.reset()
        state = encoder.encode(env.joined_tables, env.tables, 0)
        
        hidden = None # For DRQN
        done = False
        episode_reward = 0
        
        while not done:
            if mode == "agent":
                if hasattr(agent, 'select_action'):
                    # Check if DRQN or DQN
                    if isinstance(agent, DRQNAgent):
                        action, hidden = agent.select_action(state, hidden, eval_mode=True)
                    else:
                        action = agent.select_action(state, eval_mode=True)
            elif mode == "random":
                action = env.action_space.sample()
            elif mode == "greedy":
                # Heuristic: try all valid joins, pick cheapest
                # This requires env to support 'peek' or we just estimate outside
                # Simplification: Random for now, or access internal cost model
                action = env.action_space.sample() 
            
            next_raw_state, reward, done, truncated, info = env.step(action)
            
            current_cost = next_raw_state['cost'][0]
            state = encoder.encode(env.joined_tables, env.tables, current_cost)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        # Execution time is roughly -reward
        execution_times.append(-episode_reward)
        
    return np.mean(total_rewards), np.mean(execution_times)

def main():
    config = load_config()
    test_queries = ["SELECT * FROM t1, t2, t3 WHERE t1.id = t2.id AND t2.id = t3.id"] 
    env = QueryEnv(config, queries=test_queries)
    encoder = StateEncoder(config['rl'])
    
    # Load Agent
    # agent = DQNAgent(...)
    # agent.policy_net.load_state_dict(...)
    
    print("Evaluating Random Baseline...")
    rand_reward, rand_time = run_evaluation(None, env, encoder, mode="random")
    print(f"Random: Reward={rand_reward}, Time={rand_time}")
    
    # print("Evaluating Agent...")
    # agent_reward, agent_time = run_evaluation(agent, env, encoder, mode="agent")
    # print(f"Agent: Reward={agent_reward}, Time={agent_time}")

if __name__ == "__main__":
    main()
