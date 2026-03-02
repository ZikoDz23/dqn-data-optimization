# RL Query Optimizer

Implementation of a Reinforcement Learning based SQL Query Optimizer (DQN/DRQN).

## Setup
1. Install dependencies:
```bash
pip install -r rl_query_optimizer/requirements.txt
```
2. Configure Database in `rl_query_optimizer/config.yaml`.
3. Ensure PostgreSQL is running and has the JOB database.

## Training
To train the DQN agent:
```bash
python -m rl_query_optimizer.training.train_dqn
```

To train the DRQN agent:
```bash
python -m rl_query_optimizer.training.train_drqn
```

## Evaluation
To evaluate agents vs baselines:
```bash
python -m rl_query_optimizer.training.evaluate
```

## Structure
- `env/`: Gymnasium environment and DB cost interface.
- `agents/`: DQN and DRQN agent implementations.
- `training/`: Training loops.
- `utils/`: SQL parsing and Graph construction.
