from scipy.stats import rankdata
from algorithms import algorithms
import matplotlib.pyplot as plt
from utils import utils
import pandas as pd
import numpy as np
import matplotlib

# This file is a collection of functions that generate the visualizations for the project report
# It is sort of disorganized because I had to move the functions from the Jupyter Notebook to this file

""" ----- Constants ----- """
k_full = [0.125, 0.25, 0.5, 0.75]
k_values = [125, 25, 50, 75]

iterations = [250, 500, 750, 1000]

# File paths for the data
OLD_EXHAUSTIVE_PATH = "../data/exhaustive_v1/exhaustive_v1_p_{}.csv"
EXHAUSTIVE_PATH = "../data/exhaustive/exhaustive_p_{}.csv"
BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_{}.csv"
SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_{}.csv"
WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree_compare/weight_to_degree_compare_p_{}.csv"
MONTE_CARLO_PATH = "../data/randomized_maximum_weight_independent_set_compare/p_{}/results_{}.csv"

FULL_BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first/biggest_weight_first_p_{}.csv"
FULL_SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first/smallest_degree_first_p_{}.csv"
FULL_WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree/weight_to_degree_p_{}.csv"

# Constants used for the plots
SOLUTION_SIZE = 'Solution Size'
GRAPH_SIZE_AXIS = 'Graph Size (|V|)'
TOTAL_WEIGHT = 'Total Weight'
EXECUTION_TIME = 'Execution Time (seconds)'
NUMBER_OF_OPERATIONS = 'Number of Operations'
UPPER_RIGHT = 'upper right'

# Constants used to identify the algorithms in the data
OLD_EXHAUSTIVE = 'Old Exhaustive'
EXHAUSTIVE = 'Exhaustive'
BIGGEST_WEIGHT_FIRST = 'Biggest Weight First'
SMALLEST_DEGREE_FIRST = 'Smallest Degree First'
WEIGHT_TO_DEGREE = 'Weight to Degree'
MONTE_CARLO = 'Monte Carlo'

ALGORITHMS = [OLD_EXHAUSTIVE, EXHAUSTIVE, BIGGEST_WEIGHT_FIRST, SMALLEST_DEGREE_FIRST, WEIGHT_TO_DEGREE]

# Labels
EXH = 'Exhaustive'
WMAX = 'WMax - Biggest Weight First'
DMIN = 'DMin - Smallest Degree First'
WDMIX = 'WDMix - Weight to Degree'
MC = 'Monte Carlo'

LABELS = {
    EXHAUSTIVE: EXH,
    BIGGEST_WEIGHT_FIRST: WMAX,
    SMALLEST_DEGREE_FIRST: DMIN,
    WEIGHT_TO_DEGREE: WDMIX
}

# Define colors for each algorithm
colors = {
    EXHAUSTIVE: 'red',
    BIGGEST_WEIGHT_FIRST: 'orange',
    SMALLEST_DEGREE_FIRST: 'green',
    WEIGHT_TO_DEGREE: 'purple'
}

# Load dataframes for each algorithm and k value
dataframes = {
    OLD_EXHAUSTIVE: {k: pd.read_csv(OLD_EXHAUSTIVE_PATH.format(k)) for k in k_values},
    EXHAUSTIVE: {k: pd.read_csv(EXHAUSTIVE_PATH.format(k)) for k in k_values},
    BIGGEST_WEIGHT_FIRST: {k: pd.read_csv(BIGGEST_WEIGHT_FIRST_PATH.format(k)) for k in k_values},
    SMALLEST_DEGREE_FIRST: {k: pd.read_csv(SMALLEST_DEGREE_FIRST_PATH.format(k)) for k in k_values},
    WEIGHT_TO_DEGREE: {k: pd.read_csv(WEIGHT_TO_DEGREE_PATH.format(k)) for k in k_values},
}

dataframes_randomized = {
    MONTE_CARLO: {k: {i: pd.read_csv(MONTE_CARLO_PATH.format(k, i)) for i in iterations} for k in k_values}
}

dataframes_heuristic_full = {
    BIGGEST_WEIGHT_FIRST: {k: pd.read_csv(FULL_BIGGEST_WEIGHT_FIRST_PATH.format(k)) for k in k_values},
    SMALLEST_DEGREE_FIRST: {k: pd.read_csv(FULL_SMALLEST_DEGREE_FIRST_PATH.format(k)) for k in k_values},
    WEIGHT_TO_DEGREE: {k: pd.read_csv(FULL_WEIGHT_TO_DEGREE_PATH.format(k)) for k in k_values}
}

# Add 'Solution_Size' column to all dataframes
for algorithm, dfs in dataframes.items():
    for k, df in dfs.items():
        df[SOLUTION_SIZE] = df['Solution'].apply(len)
        
        
""" ----- Helper Functions ----- """

# Function to calculate the mean with available data
def calculate_mean(ops_dict, graph_size):
    values = [ops[graph_size] for ops in ops_dict.values() if graph_size < len(ops)]
    return np.mean(values) if values else np.nan

# Calculate the average number of operations and execution time for each graph size across the k values
def calculate_avg_data(ops_dict, time_dict):
    max_graph_size = max(len(ops) for ops in ops_dict.values())
    avg_ops = [calculate_mean(ops_dict, i) for i in range(max_graph_size)]
    avg_time = [calculate_mean(time_dict, i) for i in range(max_graph_size)]
    return avg_ops, avg_time

""" ----- Functions for the various graphs and tables in the report ----- """
def remarks_graphs():
    """ Create the graphs for the remarks section (Fig.1) """

    # Set up subplots side by side
    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Solution Size and Total Weight for each k value
    for k, df in dataframes[EXHAUSTIVE].items():
        axes[0].plot(df[SOLUTION_SIZE], label=f'k={k / 100 if k != 125 else 0.125}')
        axes[1].plot(df[TOTAL_WEIGHT], label=f'k={k / 100 if k != 125 else 0.125}')
        
    # Set labels and legend for both plots
    axes[0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0].set_ylabel(SOLUTION_SIZE)
    axes[0].set_title('Evolution of Solution Size for Different k Values')
    axes[0].legend()

    axes[1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1].set_ylabel(TOTAL_WEIGHT)
    axes[1].set_title('Evolution of Solution Weight for Different k Values')
    axes[1].legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig('../images/evolution_of_solution.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("remarks_graphs() - Done")
    

def exhaustive_comparison_time():
    """ Compare the execution time of the two exhaustive algorithms (Fig.2) """
    
    # Extract data for execution times by graph size and k
    algorithm1_times = {k: dataframes[OLD_EXHAUSTIVE][k][EXECUTION_TIME].to_list() for k in k_values}
    algorithm2_times = {k: dataframes[EXHAUSTIVE][k][EXECUTION_TIME].to_list() for k in k_values}

    # Determine the maximum graph size across all k values
    max_graph_size_v1 = max(len(times) for times in algorithm1_times.values())
    max_graph_size_v2 = max(len(times) for times in algorithm2_times.values())

    # Function to calculate the mean with available data
    def calculate_mean(times_dict, graph_size):
        values = [times[graph_size] for times in times_dict.values() if graph_size < len(times)]
        return np.mean(values) if values else np.nan

    # Calculate the average execution time for each graph size across the k values
    avg_algorithm1_times = [calculate_mean(algorithm1_times, i) for i in range(max_graph_size_v1)]
    avg_algorithm2_times = [calculate_mean(algorithm2_times, i) for i in range(max_graph_size_v2)]

    # Plot Execution Time as a line graph
    plt.figure(figsize=(12, 6))

    # Colors for distinguishing lines
    colors = matplotlib.colormaps["tab10"]

    # Plot average execution times for both algorithms
    plt.plot(range(max_graph_size_v1), avg_algorithm1_times, marker='o', color=colors(0), linestyle='-', linewidth=1.5, label='Exhaustive V1')
    plt.plot(range(max_graph_size_v2), avg_algorithm2_times, marker='s', color=colors(3), linestyle='--', linewidth=1.5, label='Exhaustive V2')

    # Set labels and legend for Execution Time plot
    plt.xlabel(GRAPH_SIZE_AXIS)
    plt.ylabel('Execution Time (s)')
    plt.title('Average Execution Time by Graph Size (for k in [12.5, 25, 50, 75])')

    # Use logarithmic scale for y-axis
    plt.yscale('log')

    # Adjust legend to stay inside the plot
    plt.legend(loc=UPPER_RIGHT, bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Layout adjustments
    plt.tight_layout()

    # Save the plot
    plt.savefig('../images/execution_time_comparison_average.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("exhaustive_comparison_time() - Done")

def exhaustive_comparison_operations():
    """ Compare the number of operations of the two exhaustive algorithms (Fig.3) """
    
    # Extract data for number of operations by graph size and k
    algorithm1_ops = {k: dataframes[OLD_EXHAUSTIVE][k][NUMBER_OF_OPERATIONS] for k in k_values}
    algorithm2_ops = {k: dataframes[EXHAUSTIVE][k][NUMBER_OF_OPERATIONS] for k in k_values}

    # Find the maximum graph size across all k values
    max_graph_size_v1 = max(len(ops) for ops in algorithm1_ops.values())
    max_graph_size_v2 = max(len(ops) for ops in algorithm2_ops.values())

    # Calculate the average number of operations for each graph size across the k values
    avg_algorithm1_ops = [calculate_mean(algorithm1_ops, i) for i in range(max_graph_size_v1)]
    avg_algorithm2_ops = [calculate_mean(algorithm2_ops, i) for i in range(max_graph_size_v2)]

    # Plot Execution Time as a line graph
    plt.figure(figsize=(12, 6))

    # Colors for distinguishing lines
    colors = matplotlib.colormaps["tab10"]  # Use the updated method for color mapping

    # Plot average execution ops for both algorithms
    plt.plot(range(max_graph_size_v1), avg_algorithm1_ops, marker='o', color=colors(0), linestyle='-', linewidth=1.5, label='Exhaustive V1')
    plt.plot(range(max_graph_size_v2), avg_algorithm2_ops, marker='s', color=colors(3), linestyle='--', linewidth=1.5, label='Exhaustive V2')

    # Set labels and legend for Execution Time plot
    plt.xlabel(GRAPH_SIZE_AXIS)
    plt.ylabel(NUMBER_OF_OPERATIONS)
    plt.title('Average Number of Operations by Graph Size (for k in [12.5, 25, 50, 75])')

    # Use logarithmic scale for y-axis
    plt.yscale('log')

    # Adjust legend to stay inside the plot
    plt.legend(loc=UPPER_RIGHT, bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Layout adjustments
    plt.tight_layout()

    # Save the plot
    plt.savefig('../images/number_of_operations_comparison_average.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("exhaustive_comparison_operations() - Done")
    
def greedy_comparison_operations_time():
    """ Compare the number of operations and execution time of the three greedy algorithms (Fig.4) """

    def extract_data_avg(dataframes, column_name):
        """ Extract and calculate average data for all algorithms. """
        return [
            calculate_avg_data(
                {k: dataframes[algorithm][k][column_name] for k in k_values},
                {k: dataframes[algorithm][k][EXECUTION_TIME] for k in k_values}
            )
            for algorithm in [BIGGEST_WEIGHT_FIRST, SMALLEST_DEGREE_FIRST, WEIGHT_TO_DEGREE]
        ]

    # Extract and calculate averages for operations and time
    avg_data = extract_data_avg(dataframes_heuristic_full, NUMBER_OF_OPERATIONS)
    avg_ops = [data[0] for data in avg_data]
    avg_time = [data[1] for data in avg_data]

    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    scale = 100
    x_values = [100 + x * scale for x in range(len(avg_ops[0]))]

    # Titles and labels for plots
    titles = ['Average Number of Operations', 'Average Execution Time (s)']
    y_labels = [NUMBER_OF_OPERATIONS, 'Execution Time (s)']
    data = [avg_ops, avg_time]
    labels = [WMAX, DMIN, WDMIX]

    # Plot data
    for i, ax in enumerate(axes):
        for idx, y_data in enumerate(data[i]):
            ax.plot(x_values, y_data, color=colors[ALGORITHMS[idx+2]], linewidth=1.5, label=labels[idx])
        ax.set_xlabel(GRAPH_SIZE_AXIS)
        ax.set_ylabel(y_labels[i])
        ax.set_title(f'{titles[i]} by Graph Size (for k in [12.5, 25, 50, 75])')
        ax.legend(loc=UPPER_RIGHT, bbox_to_anchor=(0.95, 0.95), fontsize='small')

    plt.tight_layout()
    plt.savefig('../images/greedy_comparison_time_and_operations.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("greedy_comparison_operations_time() - Done")

def solution_comparison():
    """ Compare the solutions found by the algorithms (Fig.5) """
    
    # Visualize one of the graphs and the solutions found by the algorithms
    algorithms_sets = {}

    k = 0.5
    test_graph = utils.create_graph_v4(20, k)

    algorithms_sets[EXHAUSTIVE], _ = algorithms.exhaustive_v2(test_graph)
    algorithms_sets[BIGGEST_WEIGHT_FIRST], _ = algorithms.biggest_weight_first_v2(test_graph)
    algorithms_sets[SMALLEST_DEGREE_FIRST], _ = algorithms.smallest_degree_first_v1(test_graph)
    algorithms_sets[WEIGHT_TO_DEGREE], _ = algorithms.weight_to_degree_v1(test_graph)

    utils.visual_compare_algorithms(test_graph, algorithms_sets, k=k, store_png=True)
    
    # Log the conclusion of the function
    print("solution_comparison() - Done")
    
def exhaustive_vs_greedy_size_weight():
    """ Compare the solution size and total weight of the greedy algorithms with the exhaustive algorithm (Fig.6) """
    """ Also generates the tables for the ranking of the algorithms (Tables 1 and 2) """

    # Create a figure with subplots
    _, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)
    
    # Loop over each k value
    for idx, k in enumerate(k_values):
        
        space = len(dataframes[EXHAUSTIVE][k])

        # Plot Solution Size and Total Weight for each algorithm
        for algorithm in ALGORITHMS[1:]:
            alpha_value = 0.65 if algorithm != EXHAUSTIVE else 1
            dataframes[algorithm][k][SOLUTION_SIZE].head(space).plot(ax=axes[idx, 0], label=LABELS[algorithm], color=colors[algorithm], alpha=alpha_value)
            dataframes[algorithm][k][TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label=LABELS[algorithm], color=colors[algorithm], alpha=alpha_value)
        
        # Set labels for Solution Size plot
        axes[idx, 0].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 0].set_ylabel(SOLUTION_SIZE, fontsize=8)
        axes[idx, 0].set_title(f'Solution Size for k={k}', fontsize=10)
        axes[idx, 0].legend(fontsize=7, loc='upper left')

        # Set labels for Total Weight plot
        axes[idx, 1].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 1].set_ylabel(TOTAL_WEIGHT, fontsize=8)
        axes[idx, 1].set_title(f'Total Weight of Solution for k={k}', fontsize=10)
        axes[idx, 1].legend(fontsize=7, loc='upper left')
        
        size_ranks = np.zeros((4, space))
        weight_ranks = np.zeros((4, space))
        speed_ranks = np.zeros((4, space))

        # Calculate ranks per graph instance, handling ties
        for i in range(space):
            solution_sizes = [dataframes[algorithm][k][SOLUTION_SIZE][i] for algorithm in ALGORITHMS[1:]]
            solution_weights = [dataframes[algorithm][k][TOTAL_WEIGHT][i] for algorithm in ALGORITHMS[1:]]
            solution_speeds = [dataframes[algorithm][k][EXECUTION_TIME][i] for algorithm in ALGORITHMS[1:]]
            
            # Use rankdata with 'min' to rank from largest to smallest (1 is best, 4 is worst)
            size_ranks[:, i] = 5 - rankdata(solution_sizes, method='max')
            weight_ranks[:, i] = 5 - rankdata(solution_weights, method='max')
            speed_ranks[:, i] = rankdata(solution_speeds, method='min')
        
        # Calculate mean ranks across all graph sizes
        size_ranks = np.mean(size_ranks, axis=1)
        weight_ranks = np.mean(weight_ranks, axis=1)
        speed_ranks = np.mean(speed_ranks, axis=1)

        # Prepare ranking table data for current k
        ranking_data_k = {
            'Algorithm': ALGORITHMS[1:],
            'Solution Size Rank': np.round(size_ranks, 2),
            'Total Weight Rank': np.round(weight_ranks, 2),
            'Speed Rank': np.round(speed_ranks, 2),
            'Average Rank': np.round((size_ranks + weight_ranks + speed_ranks) / 3, 2)
        }
        ranking_df_k = pd.DataFrame(ranking_data_k).set_index('Algorithm')
        ranking_df_k = ranking_df_k.sort_values(by='Average Rank')

        # Save table data to CSV
        ranking_df_k.to_csv(f'../data/ranking_table_k_{k}.csv')

    # Save the main figure with all plots
    plt.tight_layout(pad=1.0)
    plt.savefig('../images/solution_size_weight_comparison_all.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("exhaustive_vs_greedy_size_weight() - Done")
    
def exhaustive_vs_greedy_error_ratio_and_accuracy():
    """ Compare the error ratio and accuracy of the greedy algorithms in relation to the exhaustive algorithm (Fig.7) """

    # Initialize the plot grid (2x2 layout)
    _, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 grid
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
    
    # Dictionary to store the results for each k
    accuracy_precision_results = {}

    # Define the colors for each algorithm
    algorithm_colors = {
        WMAX: 'orange',
        DMIN: 'green',
        WDMIX: 'purple'
    }

    # Iterate through each k value
    for idx, k in enumerate(k_values):
        
        # Limit heuristic algorithms to the same solution count as the exhaustive algorithm
        solution_count = len(dataframes[EXHAUSTIVE][k])
        df_biggest_weight_first = dataframes[BIGGEST_WEIGHT_FIRST][k].head(solution_count)
        df_smallest_degree_first = dataframes[SMALLEST_DEGREE_FIRST][k].head(solution_count)
        df_weight_to_degree = dataframes[WEIGHT_TO_DEGREE][k].head(solution_count)
        
        # Structure data in a dictionary for the heuristics
        algorithms_cut = {
            WMAX: df_biggest_weight_first,
            DMIN: df_smallest_degree_first,
            WDMIX: df_weight_to_degree
        }
        
        # Calculate accuracy and precision for solution weight
        accuracy_precision_df_weight = utils.calculate_accuracy_precision(dataframes[EXHAUSTIVE][k], algorithms_cut, metric=TOTAL_WEIGHT)
        accuracy_precision_df_weight['k'] = k  # Add k value for identification
        
        # Store the resulting DataFrame in the dictionary
        accuracy_precision_results[k] = accuracy_precision_df_weight

        # Plot error (the difference between greedy and exhaustive solution weight)
        for name, df in algorithms_cut.items():
            # Calculate error as the difference between exhaustive and greedy solution weight, normalized by exhaustive weight
            error = (dataframes[EXHAUSTIVE][k][TOTAL_WEIGHT].values - df[TOTAL_WEIGHT].values) / dataframes[EXHAUSTIVE][k][TOTAL_WEIGHT].values
            
            # Plot with the respective color and label
            axs[idx].plot(dataframes[EXHAUSTIVE][k]['Node Count'], error, label=name, color=algorithm_colors[name], alpha=0.65)

        # Set plot labels and title
        axs[idx].set_xlabel(GRAPH_SIZE_AXIS, fontsize=12)
        axs[idx].set_ylabel('Error (Ratio)', fontsize=12)
        axs[idx].set_title(f'Error (Greedy - Exhaustive) for k={k}', fontsize=14)
        axs[idx].legend(loc=UPPER_RIGHT, fontsize=10)

        # Set the grid for better readability
        axs[idx].grid(True)
        
    # Concatenate results for all k values into a single DataFrame
    final_accuracy_precision_df = pd.concat(accuracy_precision_results.values(), ignore_index=True)

    # Save to CSV
    final_accuracy_precision_df.to_csv('../data/accuracy_precision_summary.csv', index=False)

    # Adjust layout for the 2x2 grid
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust horizontal and vertical space

    # Save or show the plot
    plt.savefig('../images/error_grid_2x2.png', dpi=300)
    
    # Log the conclusion of the function
    print("exhaustive_vs_greedy_error_ratio_and_accuracy() - Done")
    
def monte_carlo_precision():
    """ Check the results of the Monte Carlo algorithm and its precision, comparing the effect of iterations (Fig.8) """
    
    # Initialize the plot grid (2x2 layout)
    _, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 grid
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

    # Iterate through each k value
    for idx, k in enumerate(k_values):
        
        # Calculate precision by comparing to the exhaustive algorithm data
        exhaustive_weights = dataframes[EXHAUSTIVE][k][TOTAL_WEIGHT].values
        precision_values = []

        for i in iterations:
            monte_carlo_weights = dataframes_randomized[MONTE_CARLO][k][i][TOTAL_WEIGHT].head(len(exhaustive_weights)).values
            precision = np.mean(monte_carlo_weights / exhaustive_weights)
            precision_values.append(precision)
        # Plot the precision values for each iteration count
        axs[idx].plot(iterations, precision_values, marker='o', color='blue', linestyle='-', linewidth=1.5)
        
        # Set plot labels and title
        axs[idx].set_xlabel('Iterations', fontsize=12)
        axs[idx].set_ylabel('Precision', fontsize=12)
        axs[idx].set_title(f'Precision for Monte Carlo with k={k_full[idx]}', fontsize=14)
        
        # Set the grid for better readability
        axs[idx].grid(True)
        
    # Adjust layout for the 2x2 grid
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust horizontal and vertical space

    # Save or show the plot
    plt.savefig('../images/monte_carlo_precision_grid_2x2.png', dpi=300)
    
    # Log the conclusion of the function
    print("monte_carlo_precision() - Done")
    
        
if __name__ == "__main__":
    remarks_graphs()
    exhaustive_comparison_time()
    exhaustive_comparison_operations()
    exhaustive_vs_greedy_size_weight()
    greedy_comparison_operations_time()
    solution_comparison()
    exhaustive_vs_greedy_error_ratio_and_accuracy()
    monte_carlo_precision()