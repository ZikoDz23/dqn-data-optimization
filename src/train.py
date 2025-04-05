import numpy as np
import random
import keras._tf_keras
import tensorflow as tf
from datetime import datetime
from src.environment import QueryEnv
from src.dqn import build_dueling_dqn
from src.ReplayBuffer import PrioritizedReplayBuffer

def train(db_config, episodes=500, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, update_target_freq=5):
    env = QueryEnv(db_config)
    state_size = 4
    action_size = 4
    buffer = PrioritizedReplayBuffer(capacity=10000)
    model = build_dueling_dqn(state_size, action_size)
    target_model = build_dueling_dqn(state_size, action_size)
    target_model.set_weights(model.get_weights())

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras._tf_keras.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
    summary_writer = tf.summary.create_file_writer(log_dir)  # Pour écrire des métriques personnalisées

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                q_values = model.predict(np.array([state]), verbose=0)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            total_reward += reward

            q_current = model.predict(np.array([state]), verbose=0)[0][action]
            next_q_values = target_model.predict(np.array([next_state]), verbose=0)[0]
            next_action = np.argmax(model.predict(np.array([next_state]), verbose=0)[0])
            target = reward + gamma * next_q_values[next_action] * (1 - done)
            td_error = abs(target - q_current)

            buffer.add(state, action, reward, next_state, done, td_error)
            state = next_state

            if buffer.size() > batch_size:
                batch, weights, indices = buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                next_states = np.array(next_states)

                targets = model.predict(states, verbose=0)
                next_q_values = target_model.predict(next_states, verbose=0)
                next_actions = np.argmax(model.predict(next_states, verbose=0), axis=1)

                for i in range(batch_size):
                    targets[i][actions[i]] = rewards[i] + gamma * next_q_values[i][next_actions[i]] * (1 - dones[i])

                model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0, callbacks=[tensorboard_callback])
                new_q_values = model.predict(states, verbose=0)
                td_errors = np.abs(targets - new_q_values).max(axis=1)
                buffer.update_priorities(indices, td_errors)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if episode % update_target_freq == 0:
            target_model.set_weights(model.get_weights())
            print(f"Épisode {episode}, Récompense totale : {total_reward}, Epsilon : {epsilon}")
            with summary_writer.as_default():
                tf.summary.scalar("Total Reward", total_reward, step=episode)

    env.close()
    model.save_weights("models/dueling_dqn.h5")