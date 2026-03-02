import time
import os
import sys

def measure(label, func):
    start = time.time()
    try:
        res = func()
        end = time.time()
        print(f"[{label}] took {end - start:.4f}s")
        return res
    except Exception as e:
        end = time.time()
        print(f"[{label}] FAILED after {end - start:.4f}s: {e}")
        return None

print("Starting diagnostic check...")

# 1. Imports
def import_torch():
    import torch
    return torch

torch = measure("Import torch", import_torch)

# 2. CUDA check
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda")
        # Force initialization
        t = torch.tensor([1.0]).to(device)
        return device
    else:
        print("CUDA is NOT available.")
        return torch.device("cpu")

device = measure("Check/Init CUDA", check_cuda)

# 3. Model creation
def create_model():
    import torch.nn as nn
    model = nn.LSTM(10, 128).to(device)
    return model

measure("Create LSTM on device", create_model)

# 4. File I/O
def load_queries():
    import glob
    files = glob.glob("rl_query_optimizer/data/train_queries/*.sql")
    print(f"Found {len(files)} query files.")
    data = []
    for f in files:
        with open(f, 'r') as fp:
            data.append(fp.read())
    return len(data)

measure("Load 1000 queries", load_queries)

# 5. DB Connect
def connect_db():
    import psycopg2
    import yaml
    with open("rl_query_optimizer/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    conn = psycopg2.connect(
        dbname=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port']
    )
    conn.close()

measure("Connect to DB", connect_db)

print("Diagnostic finished.")
