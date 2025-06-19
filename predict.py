import numpy as np
import tensorflow as tf
from src.environment import QueryEnv
from src.dqn import build_dueling_dqn
import csv
from dotenv import load_dotenv
import os

load_dotenv()

# === Database Configuration ===
db_config = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Model and Environment Parameters 
state_size = 5
action_size = 9 
model_path = "models/dueling_dqn.weights.h5"

# === Load the trained model ===
model = build_dueling_dqn(state_size, action_size)
model.load_weights(model_path)

# === Prepare the environment ===
env = QueryEnv(db_config)

# === Result CSV File ===
output_file = "dqn_predictions_final.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["QueryFile", "BestAction", "ExecutionTime_ms"])

    # Loop through all queries
    for i in range(len(env.job_queries)):
        raw_state = env.reset()
        state = np.array(raw_state, dtype=np.float32).reshape(1, state_size)

        # Predict best action
        q_values = model.predict(state, verbose=0)
        action_idx = int(np.argmax(q_values[0]))
        action_name = env.actions[action_idx]

        # Apply the best action
        _, reward, _ = env.step(action_idx)
        exec_time = round(-reward, 3) if reward < 0 else 0.0

        print(f"Query: {env.get_query_filename()} | Action: {action_name} | Execution Time: {exec_time} ms")
        writer.writerow([env.get_query_filename(), action_name, exec_time])

env.close()
print(f"\nResults saved to: {output_file}")
