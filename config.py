"""
Global configuration for the Quantum Circuit Optimizer.
"""

# ── Cost function weights ──────────────────────────────────────────
ALPHA_DEPTH = 0.3          # weight for circuit depth
BETA_CNOT = 0.5            # weight for CNOT (CX) gate count
GAMMA_ERROR = 0.2          # weight for estimated error rate

# ── RL environment ─────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE = 50  # max rewrite steps per episode
STAGNATION_LIMIT = 10       # stop after N steps without improvement
STATE_DIM = 12              # dimension of feature vector

# ── RL agent (PPO) ─────────────────────────────────────────────────
RL_LEARNING_RATE = 3e-4
RL_GAMMA = 0.99
RL_TOTAL_TIMESTEPS = 50_000
RL_N_STEPS = 256
RL_BATCH_SIZE = 64
RL_N_EPOCHS = 10

# ── Genetic Algorithm ──────────────────────────────────────────────
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 50
GA_ELITE_COUNT = 4
GA_TOURNAMENT_SIZE = 3
GA_CROSSOVER_RATE = 0.7
GA_MUTATION_RATE = 0.3
GA_SEQUENCE_LENGTH = 20     # length of action sequences

# ── Hybrid optimizer ───────────────────────────────────────────────
HYBRID_RL_MUTATION_PROB = 0.25  # probability of using RL during GA mutation
HYBRID_REFINE_TOP_K = 5         # refine top-k elites with RL each generation
HYBRID_REFINE_STEPS = 30        # greedy refine steps per elite

# ── Benchmark ──────────────────────────────────────────────────────
BENCHMARK_RUNS = 5          # repetitions for variance estimation

# ── Noise model (simple depolarizing) ──────────────────────────────
SINGLE_GATE_ERROR = 1e-4
TWO_GATE_ERROR = 1e-2
READOUT_ERROR = 1e-2
