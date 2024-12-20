""" ----- Constants ----- """
k_full = [0.125, 0.25, 0.5, 0.75]
k_values = [125, 25, 50, 75]

iterations = [25, 50, 100, 250, 500, 750, 1000]

# File paths for the data
OLD_EXHAUSTIVE_PATH = "../data/exhaustive_v1/results_{}.csv"
EXHAUSTIVE_PATH = "../data/exhaustive/results_{}.csv"
BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first/{}/results_{}.csv"
SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first/{}/results_{}.csv"
WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree/{}/results_{}.csv"
MONTE_CARLO_PATH = "../data/monte_carlo/{}/results_{}.csv"
PARALLEL_HEURISTIC_MONTE_CARLO_PATH = "../data/parallel_heuristic_monte_carlo/{}/results_{}.csv"
SIMULATED_ANNEALING_PATH = "../data/simulated_annealing/{}/results_{}.csv"

# Pregen data file paths
PREGEN_WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree/pregen/results.csv"
PREGEN_MONTE_CARLO_PATH = "../data/monte_carlo/pregen/results.csv"
PREGEN_PARALLEL_HEURISTIC_MONTE_CARLO_PATH = "../data/parallel_heuristic_monte_carlo/pregen/results.csv"
PREGEN_SIMULATED_ANNEALING_PATH = "../data/simulated_annealing/pregen/results.csv"

DATA_FOLDER = "../data/"

SMALL = "small"
BIG = "big"
PREGEN = 'pregen'

# Constants used for the plots
NODE_COUNT = 'Node Count'
SOLUTION_SIZE = 'Solution Size'
GRAPH_SIZE_AXIS = 'Graph Size (|V|)'
TOTAL_WEIGHT = 'Total Weight'
EXECUTION_TIME = 'Execution Time (seconds)'
NUMBER_OF_OPERATIONS = 'Number of Operations'
UPPER_RIGHT = 'upper right'
UPPER_LEFT = 'upper left'

# Constants used to identify the algorithms in the data
OLD_EXHAUSTIVE = 'Old Exhaustive'
EXHAUSTIVE = 'Exhaustive'
BIGGEST_WEIGHT_FIRST = 'Biggest Weight First'
SMALLEST_DEGREE_FIRST = 'Smallest Degree First'
WEIGHT_TO_DEGREE = 'Weight to Degree'
MONTE_CARLO = 'Monte Carlo'
PARALLEL_HEURISTIC_MONTE_CARLO = 'Parallel Heuristic Monte Carlo'
SIMULATED_ANNEALING = 'Simulated Annealing'

ALGORITHMS = [OLD_EXHAUSTIVE, EXHAUSTIVE, BIGGEST_WEIGHT_FIRST, SMALLEST_DEGREE_FIRST, WEIGHT_TO_DEGREE, MONTE_CARLO, PARALLEL_HEURISTIC_MONTE_CARLO, SIMULATED_ANNEALING]

# Labels
EXH = 'Exhaustive'
WMAX = 'WMax - Biggest Weight First'
DMIN = 'DMin - Smallest Degree First'
WDMIX = 'WDMix - Weight to Degree'
MC = 'Monte Carlo'
PHMC = 'PHMC'
SA = 'Simulated Annealing'

LABELS = {
    EXHAUSTIVE: EXH,
    BIGGEST_WEIGHT_FIRST: WMAX,
    SMALLEST_DEGREE_FIRST: DMIN,
    WEIGHT_TO_DEGREE: WDMIX,
    MONTE_CARLO: MC,
    PARALLEL_HEURISTIC_MONTE_CARLO: PHMC,
    SIMULATED_ANNEALING: SA
}

# Define colors for each algorithm
colors = {
    EXHAUSTIVE: 'red',
    BIGGEST_WEIGHT_FIRST: 'orange',
    SMALLEST_DEGREE_FIRST: 'green',
    WEIGHT_TO_DEGREE: 'purple',
    MONTE_CARLO: 'blue',
    PARALLEL_HEURISTIC_MONTE_CARLO: 'cyan',
    SIMULATED_ANNEALING: 'magenta'
}
