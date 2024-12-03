""" ----- Constants ----- """
k_full = [0.125, 0.25, 0.5, 0.75]
k_values = [125, 25, 50, 75]

iterations = [25, 50, 100, 250, 500, 750, 1000]

# File paths for the data
OLD_EXHAUSTIVE_PATH = "../data/exhaustive_v1/exhaustive_v1_p_{}.csv"
EXHAUSTIVE_PATH = "../data/exhaustive/exhaustive_p_{}.csv"
BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_{}.csv"
SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_{}.csv"
WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree_compare/weight_to_degree_compare_p_{}.csv"
MONTE_CARLO_PATH = "../data/monte_carlo_compare/p_{}/results_{}.csv"
HEURISTIC_MONTE_CARLO_PATH = "../data/heuristic_monte_carlo_compare/p_{}/results_{}.csv"

FULL_BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first/biggest_weight_first_p_{}.csv"
FULL_SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first/smallest_degree_first_p_{}.csv"
FULL_WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree/weight_to_degree_p_{}.csv"

DATA_FOLDER = "../data/"

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
THREADED_HEURISTIC_MONTE_CARLO = 'Threaded Heuristic Monte Carlo'
SIMULATED_ANNEALING = 'Simulated Annealing'

ALGORITHMS = [OLD_EXHAUSTIVE, EXHAUSTIVE, BIGGEST_WEIGHT_FIRST, SMALLEST_DEGREE_FIRST, WEIGHT_TO_DEGREE]

RANDOMIZED_ALGORITHMS = [MONTE_CARLO, THREADED_HEURISTIC_MONTE_CARLO, SIMULATED_ANNEALING]

# Labels
EXH = 'Exhaustive'
WMAX = 'WMax - Biggest Weight First'
DMIN = 'DMin - Smallest Degree First'
WDMIX = 'WDMix - Weight to Degree'
MC = 'Monte Carlo'
THMC = 'Threaded Heuristic Monte Carlo (WDMix)'
SA = 'Simulated Annealing'

LABELS = {
    EXHAUSTIVE: EXH,
    BIGGEST_WEIGHT_FIRST: WMAX,
    SMALLEST_DEGREE_FIRST: DMIN,
    WEIGHT_TO_DEGREE: WDMIX,
    MONTE_CARLO: MC,
    THREADED_HEURISTIC_MONTE_CARLO: THMC,
    SIMULATED_ANNEALING: SA
}

# Define colors for each algorithm
colors = {
    EXHAUSTIVE: 'red',
    BIGGEST_WEIGHT_FIRST: 'orange',
    SMALLEST_DEGREE_FIRST: 'green',
    WEIGHT_TO_DEGREE: 'purple',
    MONTE_CARLO: 'blue',
    THREADED_HEURISTIC_MONTE_CARLO: 'cyan',
    SIMULATED_ANNEALING: 'magenta'
}
