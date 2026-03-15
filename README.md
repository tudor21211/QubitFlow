# Quantum Circuit Optimiser — RL + GA Hybrid

A hybrid **Reinforcement Learning + Genetic Algorithm** system that optimises quantum circuits by reducing depth, gate count and estimated error while preserving functional equivalence.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Hybrid Optimiser                           │
│                                                                │
│   ┌──────────────┐      ┌──────────────────────────────────┐  │
│   │  RL Agent     │◄────►│  GA (outer loop)                 │  │
│   │  (PPO/DQN)    │      │  - population of action seqs     │  │
│   │               │      │  - tournament selection           │  │
│   │  Learns which │      │  - crossover + RL-guided mutation │  │
│   │  rewrite rule │      │  - elitism                        │  │
│   │  to apply     │      └──────────────────────────────────┘  │
│   └──────┬───────┘                                             │
│          │                                                     │
│          ▼                                                     │
│   ┌──────────────┐      ┌──────────────────────────────────┐  │
│   │  Environment  │      │  Cost Function                   │  │
│   │  (Gymnasium)  │◄────►│  C = α·depth + β·CX + γ·error   │  │
│   │  State=features│     └──────────────────────────────────┘  │
│   │  Action=rule  │                                            │
│   └──────┬───────┘                                             │
│          │                                                     │
│          ▼                                                     │
│   ┌──────────────────────────────────────────────────────────┐│
│   │  Rewrite Rules (11 total)                                ││
│   │  • Cancel inverse pairs  • Commute CX gates             ││
│   │  • H-X-H → Z sandwich   • H-Z-H → X sandwich           ││
│   │  • Cancel double-H       • Decompose SWAP → 3 CX        ││
│   │  • Merge Rz rotations   • Merge Rx rotations            ││
│   │  • Remove identity rot.  • Commute Z past CX control    ││
│   │  • Commute X past CX target                             ││
│   └──────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Circuit Optimizer/
├── config.py                        # All hyperparameters
├── main.py                          # Entry point (CLI)
├── requirements.txt
├── circuit_optimizer/
│   ├── representation.py            # Feature extraction (12-dim vector)
│   ├── actions.py                   # 11 rewrite rules
│   ├── cost.py                      # C = α·depth + β·CX + γ·error
│   ├── environment.py               # Gymnasium RL environment
│   ├── genetic_algorithm.py         # GA with elitism + tournament
│   ├── rl_agent.py                  # PPO / DQN via Stable-Baselines3
│   ├── hybrid_optimizer.py          # GA + RL integration (Variant A)
│   ├── baseline.py                  # Qiskit transpiler for comparison
│   ├── benchmark.py                 # Evaluation harness
│   ├── visualization.py             # Matplotlib plots
│   └── circuits/
│       └── generators.py            # Test circuit generators
├── models/                          # Saved RL models
└── results/                         # Output plots
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a quick demo

```bash
python main.py --mode demo
```

This optimises one random circuit with all four methods and plots the GA convergence.

### 3. Full pipeline (train + benchmark)

```bash
python main.py --mode full --rl-timesteps 50000 --ga-gens 50
```

### 4. Interactive visualization app

```bash
python main.py --mode visualize --qubits 4 --depth 30 --ga-gens 30 --ga-pop 25
```

This launches a local Streamlit interface that:
- auto-generates random circuits,
- runs Hybrid (GA+RL) when a saved RL model exists,
- falls back to GA when no model is found,
- saves before/after diagrams and metric charts to `results/`.

### 5. Train only / benchmark only

```bash
python main.py --mode train --rl-timesteps 50000
python main.py --mode benchmark
```

## CLI Options

| Flag              | Default   | Description                          |
|-------------------|-----------|--------------------------------------|
| `--mode`          | `full`    | `demo` / `train` / `benchmark` / `visualize` / `full` |
| `--rl-timesteps`  | 50 000    | Total RL training steps              |
| `--ga-gens`       | 50        | GA generations                       |
| `--ga-pop`        | 30        | GA population size                   |
| `--runs`          | 5         | Benchmark repetitions for variance   |
| `--model-path`    | `models/rl_agent` | RL model save/load path       |
| `--qubits`        | 4         | Qubits for random training circuits  |
| `--depth`         | 20        | Depth of random training circuits    |
| `--seed`          | `None`    | Optional seed for reproducible random circuits |

## How It Works

### Step 1 — Representation
Circuits are converted to a **12-dimensional feature vector** (total gates, depth, CX count, per-qubit histogram stats, etc.) that the RL agent observes.

### Step 2 — Actions
The agent chooses from **11 equivalence-preserving rewrite rules**: gate cancellation, rotation merging, commutation, Hadamard sandwiches, SWAP decomposition, and more.

### Step 3 — Cost Function
$$C = \alpha \cdot \text{depth} + \beta \cdot \text{CX count} + \gamma \cdot \text{estimated error}$$
Default weights: α=0.3, β=0.5, γ=0.2.

### Step 4 — RL Environment
A Gymnasium environment where each episode is one circuit. The agent takes rewrite actions, receiving `reward = C_before − C_after`. Episodes end after max steps or stagnation.

### Step 5 — Genetic Algorithm
Evolves *sequences of actions*. Selection via tournament, single-point crossover, and point mutation.

### Step 6 — Hybrid Integration (Variant A)
During GA mutation, each gene has a `rl_mutation_prob` chance of being set by the RL policy instead of randomly — combining global diversity (GA) with local quality (RL).

### Step 7 — Training
PPO (Proximal Policy Optimisation) from Stable-Baselines3. Trains on randomly generated circuits.

### Step 8 — Benchmarking
Four methods compared on 8 test circuits:
- **Original** (no optimisation)
- **Qiskit transpiler** (level 3)
- **GA only**
- **Hybrid GA + RL**

Metrics: cost, depth, gate count, CX count, execution time, variance.

### Step 9 — Generalisation
Train on small random circuits, test on QFT, variational, Grover-like circuits of different sizes.

### Step 10 — Visualisation
Six plot types: GA convergence, cost/depth/CX comparison bars, stability box plots, improvement heatmap.

## Technology Stack

- **Qiskit** — circuit representation, DAG, transpiler baseline
- **Gymnasium** — RL environment interface
- **Stable-Baselines3** — PPO / DQN implementation
- **PyTorch** — neural network backend
- **NumPy** — numerical features
- **Matplotlib + Seaborn** — visualisation
- **NetworkX** — (available for graph analysis)
