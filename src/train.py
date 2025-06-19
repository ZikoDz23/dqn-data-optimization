import numpy as np
import random
import keras
import tensorflow as tf
import os
import csv
from datetime import datetime
from src.environment import QueryEnv
from src.dqn import build_dueling_dqn
from src.ReplayBuffer import PrioritizedReplayBuffer
from keras._tf_keras.keras.callbacks import TensorBoard

def train(db_config, episodes=1000, batch_size=32, gamma=0.99, epsilon=1.0,
          epsilon_min=0.01, epsilon_decay=0.995, update_target_freq=1,
          max_steps_per_episode=5):

    env = QueryEnv(db_config)
    state_size = 5
    action_size = len(env.actions)

    model = build_dueling_dqn(state_size, action_size)
    target_model = build_dueling_dqn(state_size, action_size)
    target_model.set_weights(model.get_weights())
    replay_buffer = PrioritizedReplayBuffer(capacity=10000)

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    csv_file = "logs/dqn_best_results_with_history.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["QueryFile", "BestExecutionTime_ms", "BestActionSequence"])

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    summary_writer = tf.summary.create_file_writer(log_dir)

    best_times = {}
    best_actions = {}

    for episode in range(episodes):
        raw_state = env.reset()
        state = np.array(raw_state, dtype=np.float32).reshape(1, state_size)
        total_reward = 0
        done = False
        step_counter = 0

        while not done and step_counter < max_steps_per_episode:
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0])

            raw_next_state, reward, done = env.step(action)
            next_state = np.array(raw_next_state, dtype=np.float32).reshape(1, state_size)
            total_reward += reward

            q_current = model.predict(state, verbose=0)[0][action]
            next_q = target_model.predict(next_state, verbose=0)[0]
            next_action = np.argmax(model.predict(next_state, verbose=0)[0])
            target = reward + gamma * next_q[next_action] * (1 - done)
            td_error = abs(target - q_current)

            replay_buffer.add(state[0], action, reward, next_state[0], done, td_error)
            state = next_state
            step_counter += 1

            if replay_buffer.size() > batch_size:
                (states, actions, rewards, next_states, dones), weights, indices = replay_buffer.sample(batch_size)

                targets = model.predict(states, verbose=0)
                next_q_values = target_model.predict(next_states, verbose=0)
                next_actions = np.argmax(model.predict(next_states, verbose=0), axis=1)

                for i in range(batch_size):
                    targets[i][actions[i]] = rewards[i] + gamma * next_q_values[i][next_actions[i]] * (1 - dones[i])

                model.fit(
                    states,
                    targets,
                    sample_weight=weights,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[tensorboard_callback]
                )

                new_q = model.predict(states, verbose=0)
                td_errors = np.abs(targets - new_q).max(axis=1)
                replay_buffer.update_priorities(indices, td_errors)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % update_target_freq == 0:
            target_model.set_weights(model.get_weights())
            print(f"Episode {episode} → Query: {env.get_query_filename()}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

            with summary_writer.as_default():
                tf.summary.scalar("Total Reward", total_reward, step=episode)
                tf.summary.scalar("Epsilon", epsilon, step=episode)

        #sauvegarde du meilleur temps et des actions pour chaque requête
        query_filename = env.get_query_filename()
        execution_time = round(-reward, 2) if reward < 0 else 0.0
        actions_done = [env.actions[a] if isinstance(a, int) else a for a in env.history]

        if (query_filename not in best_times) or (execution_time < best_times[query_filename]):
            best_times[query_filename] = execution_time
            best_actions[query_filename] = actions_done

    #ecriture finale dans le CSV
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for query, exec_time in best_times.items():
            action_seq = " -> ".join(best_actions[query])
            writer.writerow([query, exec_time, action_seq])

    model.save_weights("models/dueling_dqn.weights.h5")
    env.close()
