"""
Global configuration for the Quantum Circuit Optimizer.
"""

# ── Cost function weights ──────────────────────────────────────────
ALPHA_DEPTH = 0.3          # weight for circuit depth
BETA_CNOT = 0.5            # weight for CNOT (CX) gate count
GAMMA_ERROR = 0.2          # weight for estimated error rate

# ── Hardware-aware multi-objective cost (Phase 2 foundation) ──────
# Keep these defaults at 0 so legacy behavior remains unchanged until
# you explicitly enable them in experiments.
DELTA_EXEC_TIME = 0.0      # weight for execution-time proxy
EPSILON_ROUTING = 0.0      # weight for coupling-map routing pressure
ZETA_COHERENCE = 0.0       # weight for coherence penalty proxy

ACTIVE_HARDWARE_PROFILE = "sim_default"
ACTIVE_OBJECTIVE_PRESET = "legacy"

OBJECTIVE_PRESETS = {
	"legacy": {
		"alpha_depth": ALPHA_DEPTH,
		"beta_cnot": BETA_CNOT,
		"gamma_error": GAMMA_ERROR,
		"delta_exec_time": 0.0,
		"epsilon_routing": 0.0,
		"zeta_coherence": 0.0,
	},
	"balanced_hw": {
		"alpha_depth": 0.25,
		"beta_cnot": 0.35,
		"gamma_error": 0.20,
		"delta_exec_time": 0.10,
		"epsilon_routing": 0.05,
		"zeta_coherence": 0.05,
	},
	"latency_first": {
		"alpha_depth": 0.20,
		"beta_cnot": 0.25,
		"gamma_error": 0.15,
		"delta_exec_time": 0.30,
		"epsilon_routing": 0.05,
		"zeta_coherence": 0.05,
	},
	"fidelity_first": {
		"alpha_depth": 0.15,
		"beta_cnot": 0.35,
		"gamma_error": 0.30,
		"delta_exec_time": 0.05,
		"epsilon_routing": 0.10,
		"zeta_coherence": 0.05,
	},
}

# Hardware profile values are intentionally lightweight proxies that can
# be replaced with real backend calibration values during experiments.
HARDWARE_PROFILES = {
	"sim_default": {
		"coupling_map": [],
		"gate_error_rates": {
			"single": 1e-4,
			"two": 1e-2,
			"three": 3e-2,
		},
		"gate_durations_ns": {
			"single": 35.0,
			"two": 250.0,
			"three": 700.0,
			"measure": 700.0,
		},
		"avg_t1_ns": 100_000.0,
		"avg_t2_ns": 80_000.0,
	},
	"ibm_like_27q": {
		"coupling_map": [
			(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 11), (11, 14),
			(14, 16), (16, 19), (19, 22), (22, 25), (25, 26),
			(1, 4), (4, 7), (7, 10), (10, 12), (12, 15), (15, 18),
			(18, 21), (21, 23), (23, 24),
			(6, 7), (9, 10), (13, 14), (17, 18), (20, 21),
		],
		"gate_error_rates": {
			"single": 1.2e-4,
			"two": 2.8e-3,
			"three": 8.4e-3,
		},
		"gate_durations_ns": {
			"single": 35.0,
			"two": 320.0,
			"three": 900.0,
			"measure": 900.0,
		},
		"avg_t1_ns": 120_000.0,
		"avg_t2_ns": 95_000.0,
	},
}

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
