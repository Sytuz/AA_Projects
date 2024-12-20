from algorithms import algorithms
#from scipy.stats import rankdata
import matplotlib.pyplot as plt
from constants import *
from utils import utils
import pandas as pd
import numpy as np
import matplotlib

# Load dataframes for each algorithm and k value
dataframes_small = {
    OLD_EXHAUSTIVE: {k: pd.read_csv(OLD_EXHAUSTIVE_PATH.format(k)) for k in k_full},
    EXHAUSTIVE: {k: pd.read_csv(EXHAUSTIVE_PATH.format(k)) for k in k_full},
    BIGGEST_WEIGHT_FIRST: {k: pd.read_csv(BIGGEST_WEIGHT_FIRST_PATH.format(SMALL,k)) for k in k_full},
    SMALLEST_DEGREE_FIRST: {k: pd.read_csv(SMALLEST_DEGREE_FIRST_PATH.format(SMALL,k)) for k in k_full},
    WEIGHT_TO_DEGREE: {k: pd.read_csv(WEIGHT_TO_DEGREE_PATH.format(SMALL,k)) for k in k_full},
    MONTE_CARLO: {k: pd.read_csv(MONTE_CARLO_PATH.format(SMALL,k)) for k in k_full},
    PARALLEL_HEURISTIC_MONTE_CARLO: {k: pd.read_csv(PARALLEL_HEURISTIC_MONTE_CARLO_PATH.format(SMALL,k)) for k in k_full}, 
    SIMULATED_ANNEALING: {k: pd.read_csv(SIMULATED_ANNEALING_PATH.format(SMALL,k)) for k in k_full}   
}

# For the EXHAUSTIVE algorithm, for every k value, add a column with the solution size
for k in k_full:
    dataframes_small[EXHAUSTIVE][k][SOLUTION_SIZE] = dataframes_small[EXHAUSTIVE][k]['Solution'].apply(lambda x: len(eval(x)))

dataframes_big = {
    BIGGEST_WEIGHT_FIRST: {k: pd.read_csv(BIGGEST_WEIGHT_FIRST_PATH.format(BIG,k)) for k in k_full},
    SMALLEST_DEGREE_FIRST: {k: pd.read_csv(SMALLEST_DEGREE_FIRST_PATH.format(BIG,k)) for k in k_full},
    WEIGHT_TO_DEGREE: {k: pd.read_csv(WEIGHT_TO_DEGREE_PATH.format(BIG,k)) for k in k_full},
    MONTE_CARLO: {k: pd.read_csv(MONTE_CARLO_PATH.format(BIG,k)) for k in k_full},
    PARALLEL_HEURISTIC_MONTE_CARLO: {k: pd.read_csv(PARALLEL_HEURISTIC_MONTE_CARLO_PATH.format(BIG,k)) for k in k_full}, 
    SIMULATED_ANNEALING: {k: pd.read_csv(SIMULATED_ANNEALING_PATH.format(BIG,k)) for k in k_full}   
}

#dataframes_pregen = {
#    WEIGHT_TO_DEGREE: pd.read_csv(PREGEN_WEIGHT_TO_DEGREE_PATH),
#    MONTE_CARLO: pd.read_csv(PREGEN_MONTE_CARLO_PATH),
#    PARALLEL_HEURISTIC_MONTE_CARLO: pd.read_csv(PREGEN_PARALLEL_HEURISTIC_MONTE_CARLO_PATH),
#    SIMULATED_ANNEALING: pd.read_csv(PREGEN_SIMULATED_ANNEALING_PATH)
#}  
        
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
    
def single_metric_comparison(algorithms, filenames, title, metric, output_filename='single_metric_comparison.png', average=None, log_scale=False):
    """
    Compare a single metric across different algorithms, and save the plot as an image.
    
    Args:
    algorithms (list): List of algorithms to compare
    filenames (list): List of filenames for the data of each algorithm
    title (str): Title of the plot
    metric (str): Metric to compare
    output_filename (str): Filename for the output image (optional)
    average (list): List of pd.Series with the average data for each algorithm (optional)
    log_scale (bool): Whether to use a logarithmic scale for the y-axis (optional)
    
    Returns:
    None
    """
    
    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Plot the data for each algorithm
    for algorithm, filename in zip(algorithms, filenames):
        df = pd.read_csv('../data/' + filename)
        if average:
            plt.plot(df[NODE_COUNT], average[algorithm], label=LABELS[algorithm], color=colors[algorithm])
        else:
            plt.plot(df[NODE_COUNT], df[metric], label=LABELS[algorithm], color=colors[algorithm])

    # Set labels and legend for the plot
    plt.xlabel(GRAPH_SIZE_AXIS)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(loc=UPPER_LEFT, fontsize='small')

    # Use logarithmic scale for the y-axis if specified
    if log_scale:
        plt.yscale('log')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'../images/{output_filename}', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print(f"{output_filename.split('/')[-1]} - Done")

def exhaustive_comparison_time_operations():
    """Compare the execution time of the two exhaustive algorithms using helper functions."""
    
    # Use results_average to calculate averages for both algorithms
    avg_algorithm1_times = utils.results_average('exhaustive_v1', k_vals=k_values, metric=EXECUTION_TIME)
    avg_algorithm2_times = utils.results_average('exhaustive', k_vals=k_values, metric=EXECUTION_TIME)
    
    avg_algorithm1_ops = utils.results_average('exhaustive_v1', k_vals=k_values, metric=NUMBER_OF_OPERATIONS)
    avg_algorithm2_ops = utils.results_average('exhaustive', k_vals=k_values, metric=NUMBER_OF_OPERATIONS)

    # Prepare the inputs for the helper function
    algorithms = [OLD_EXHAUSTIVE, EXHAUSTIVE]
    filenames = []  # No filenames needed as we're passing averages directly
    titles = ['Average Execution Time by Graph Size (for k in [12.5, 25, 50, 75])', 'Average Number of Operations by Graph Size (for k in [12.5, 25, 50, 75])']
    metrics = [EXECUTION_TIME, NUMBER_OF_OPERATIONS]
    averages = [{
        OLD_EXHAUSTIVE: avg_algorithm1_times,
        EXHAUSTIVE: avg_algorithm2_times
    },
    {
        OLD_EXHAUSTIVE: avg_algorithm1_ops,
        EXHAUSTIVE: avg_algorithm2_ops
    }]

    # Call the helper function with log scale
    for i in range(2):
        single_metric_comparison(
            algorithms=algorithms,
            filenames=filenames,
            title=titles[i],
            metric=metrics[i],
            output_filename=f'exhaustive_comparison_{metrics[i]}.png',
            average=averages[i],
            log_scale=True
        )

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
    y_labels = [NUMBER_OF_OPERATIONS, ]
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
        axes[idx, 0].legend(fontsize=7, loc=UPPER_LEFT)

        # Set labels for Total Weight plot
        axes[idx, 1].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 1].set_ylabel(TOTAL_WEIGHT, fontsize=8)
        axes[idx, 1].set_title(f'Total Weight of Solution for k={k}', fontsize=10)
        axes[idx, 1].legend(fontsize=7, loc=UPPER_LEFT)
        
        size_ranks = np.zeros((4, space))
        weight_ranks = np.zeros((4, space))
        speed_ranks = np.zeros((4, space))
        
        """
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
        """
        
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
    colors = {
        WDMIX: 'blue',
        MC: 'green',
        PHMC: 'purple',
        SA: 'orange'
    }

    # Iterate through each k value
    for idx, k in enumerate(k_full):
        
        # Limit heuristic algorithms to the same solution count as the exhaustive algorithm
        solution_count = len(dataframes_small[EXHAUSTIVE][k])
        df_weight_to_degree = dataframes_small[WEIGHT_TO_DEGREE][k].head(solution_count)
        df_monte_carlo = dataframes_small[MONTE_CARLO][k].head(solution_count)
        df_parallel_heuristic_monte_carlo = dataframes_small[PARALLEL_HEURISTIC_MONTE_CARLO][k].head(solution_count)
        df_simulated_annealing = dataframes_small[SIMULATED_ANNEALING][k].head(solution_count)
        
        # Structure data in a dictionary for the heuristics
        algorithms_cut = {
            WDMIX: df_weight_to_degree,
            MC: df_monte_carlo,
            PHMC: df_parallel_heuristic_monte_carlo,
            SA: df_simulated_annealing
        }
        
        # Calculate accuracy and precision for solution weight
        accuracy_precision_df_weight = utils.calculate_accuracy_precision(dataframes_small[EXHAUSTIVE][k], algorithms_cut, metric=TOTAL_WEIGHT)
        accuracy_precision_df_weight['k'] = k  # Add k value for identification
        
        # Store the resulting DataFrame in the dictionary
        accuracy_precision_results[k] = accuracy_precision_df_weight

        # Plot error (the difference between greedy and exhaustive solution weight)
        for name, df in algorithms_cut.items():
            # Calculate error as the difference between exhaustive and greedy solution weight, normalized by exhaustive weight
            error = (dataframes_small[EXHAUSTIVE][k][TOTAL_WEIGHT].values - df[TOTAL_WEIGHT].values) / dataframes_small[EXHAUSTIVE][k][TOTAL_WEIGHT].values
            
            # Plot with the respective color and label
            axs[idx].plot(dataframes_small[EXHAUSTIVE][k][NODE_COUNT], error, label=name, color=colors[name], alpha=0.65)

        # Set plot labels and title
        axs[idx].set_xlabel(GRAPH_SIZE_AXIS, fontsize=12)
        axs[idx].set_ylabel('Error (Ratio)', fontsize=12)
        axs[idx].set_title(f'Error (Algorithm - Exhaustive) for k={k}', fontsize=14)
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
    """
    Check the results of the Monte Carlo and Heuristic Monte Carlo algorithms 
    and their precision, comparing the effect of iterations (Fig.8).
    """

    # Initialize the plot with a 2x1 layout
    _, axs = plt.subplots(1, 2, figsize=(18, 7))  # Two plots side by side

    # Define a color palette for the different k values
    colors = plt.cm.tab10(np.linspace(0, 1, len(k_values)))

    # Algorithms to plot
    algorithms = RANDOMIZED_ALGORITHMS
    titles = ['Monte Carlo Precision', 'Heuristic Monte Carlo Precision']

    for plot_idx, algo in enumerate(algorithms):
        # Iterate through each k value
        for idx, k in enumerate(k_values):
            
            # Calculate precision by comparing to the exhaustive algorithm data
            exhaustive_weights = dataframes[EXHAUSTIVE][k][TOTAL_WEIGHT].values
            precision_values = []

            for i in iterations:
                monte_carlo_weights = dataframes[algo][k][i][TOTAL_WEIGHT].head(len(exhaustive_weights)).values
                precision = np.mean(monte_carlo_weights / exhaustive_weights)
                precision_values.append(precision)
            
            # Plot the precision values for each iteration count
            axs[plot_idx].plot(iterations, precision_values, marker='o', linestyle='-', linewidth=1.5, 
                               label=f'k={k_full[idx]}', color=colors[idx])

        # Set plot labels, title, and legend for the current axis
        axs[plot_idx].set_xlabel('Iterations', fontsize=12)
        axs[plot_idx].set_ylabel('Precision', fontsize=12)
        axs[plot_idx].set_title(titles[plot_idx], fontsize=14)
        axs[plot_idx].legend(title='k Values', fontsize=10, loc='best')  # Add a legend
        axs[plot_idx].grid(True)  # Add grid for better readability

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Save or show the plot
    plt.savefig('../images/monte_carlo_precision_comparison.png', dpi=300)
    
    # Log the conclusion of the function
    print("monte_carlo_precision() - Done")
    
def comparison_operations_time():
    """ 
    Compare the number of operations and execution time of the three greedy algorithms 
    and Monte Carlo algorithms (Fig.4 extended).
    """

    space = len(dataframes[PARALLEL_HEURISTIC_MONTE_CARLO][125][25])
    
    def extract_data_avg():
        """ Extract and calculate average data for all algorithms. """
        return [
            calculate_avg_data(
                {k: dataframes[algorithm][k][NUMBER_OF_OPERATIONS].head(space) if algorithm not in RANDOMIZED_ALGORITHMS else dataframes[algorithm][k][50][NUMBER_OF_OPERATIONS].head(space) for k in k_values},
                {k: dataframes[algorithm][k][EXECUTION_TIME].head(space) if algorithm not in RANDOMIZED_ALGORITHMS else dataframes[algorithm][k][50][EXECUTION_TIME].head(space) for k in k_values}
            )
            for algorithm in [
                BIGGEST_WEIGHT_FIRST, 
                SMALLEST_DEGREE_FIRST, 
                WEIGHT_TO_DEGREE, 
                MONTE_CARLO, 
                PARALLEL_HEURISTIC_MONTE_CARLO
            ]
        ]

    # Extract and calculate averages for operations and time
    avg_data = extract_data_avg()
    avg_ops = [data[0] for data in avg_data]
    avg_time = [data[1] for data in avg_data]

    _, axes = plt.subplots(1, 2, figsize=(16, 7))
    x_values = range(1, space+1)

    # Titles and labels for plots
    titles = ['Average Number of Operations', 'Average Execution Time (s)']
    y_labels = [NUMBER_OF_OPERATIONS, ]
    data = [avg_ops, avg_time]
    labels = [WMAX, DMIN, WDMIX, MONTE_CARLO, PARALLEL_HEURISTIC_MONTE_CARLO]

    # Define unique colors for all algorithms
    algorithm_colors = [
        colors[ALGORITHMS[idx + 1]] for idx in range(len(labels)-2)
    ]
    algorithm_colors.extend(['blue', 'cyan'])

    # Plot data
    for i, ax in enumerate(axes):
        for idx, y_data in enumerate(data[i]):
            ax.plot(x_values, y_data, color=algorithm_colors[idx], linewidth=1.5, label=labels[idx])
        ax.set_xlabel(GRAPH_SIZE_AXIS)
        ax.set_ylabel(y_labels[i])
        ax.set_title(f'{titles[i]} by Graph Size (for k in [12.5, 25, 50, 75])')
        ax.legend(loc=UPPER_RIGHT, bbox_to_anchor=(0.95, 0.95), fontsize='small')

    plt.tight_layout()
    plt.savefig('../images/comparison_time_and_operations_extended.png', dpi=300)
    plt.close()

    # Log the conclusion of the function
    print("comparison_operations_time() - Done")
    
def exhaustive_vs_algorithms_size_weight():
    """ 
    Compare the solution size and total weight of the greedy and Monte Carlo algorithms 
    with the exhaustive algorithm (Fig.6 Extended). 
    Also generates the tables for the ranking of all algorithms (Tables 1 and 2 Extended).
    """

    # Function to rank data from largest to smallest
    def rank_largest_to_smallest(data):
        sorted_indices = sorted(range(len(data)), key=lambda x: data[x], reverse=True)
        ranks = [0] * len(data)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1
        return ranks

    # Function to rank data from smallest to largest
    def rank_smallest_to_largest(data):
        sorted_indices = sorted(range(len(data)), key=lambda x: data[x])
        ranks = [0] * len(data)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1
        return ranks

    # Create a figure with subplots
    _, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)

    # Algorithms
    chosen_algorithms = [EXHAUSTIVE, WEIGHT_TO_DEGREE, MONTE_CARLO, PARALLEL_HEURISTIC_MONTE_CARLO, SIMULATED_ANNEALING]

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Loop over each k value
    for idx, k in enumerate(k_full):
        
        space = len(dataframes_small[EXHAUSTIVE][k])

        # Plot Solution Size and Total Weight for each algorithm
        for idx, algorithm in enumerate(chosen_algorithms):
            alpha_value = 0.30 if algorithm != EXHAUSTIVE else 1
            dataframe = dataframes_small[algorithm][k]
            dataframe[SOLUTION_SIZE].head(space).plot(ax=axes[idx, 0], label=LABELS[algorithm], color=colors[idx], alpha=alpha_value)
            dataframe[TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label=LABELS[algorithm], color=colors[idx], alpha=alpha_value)
        
        # Set labels for Solution Size plot
        axes[idx, 0].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 0].set_ylabel(SOLUTION_SIZE, fontsize=8)
        axes[idx, 0].set_title(f'Solution Size for k={k}', fontsize=10)
        axes[idx, 0].legend(fontsize=7, loc=UPPER_LEFT)

        # Set labels for Total Weight plot
        axes[idx, 1].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 1].set_ylabel(TOTAL_WEIGHT, fontsize=8)
        axes[idx, 1].set_title(f'Total Weight of Solution for k={k}', fontsize=10)
        axes[idx, 1].legend(fontsize=7, loc=UPPER_LEFT)
        
        # Initialize arrays for rankings
        size_ranks = np.zeros((5, space))
        weight_ranks = np.zeros((5, space))
        speed_ranks = np.zeros((5, space))

        # Calculate ranks per graph instance, handling ties
        for i in range(space):
            solution_sizes = [dataframes_small[algorithm][k][SOLUTION_SIZE][i] for algorithm in chosen_algorithms]
            solution_weights = [dataframes_small[algorithm][k][TOTAL_WEIGHT][i] for algorithm in chosen_algorithms]
            solution_speeds = [dataframes_small[algorithm][k][EXECUTION_TIME][i] for algorithm in chosen_algorithms]
            
            # Use rankdata with 'min' to rank from largest to smallest (1 is best, N is worst)
            size_ranks[:, i] = rank_largest_to_smallest(solution_sizes)
            weight_ranks[:, i] = rank_largest_to_smallest(solution_weights)
            speed_ranks[:, i] = rank_smallest_to_largest(solution_speeds)
        
        # Calculate mean ranks across all graph sizes
        size_ranks = np.mean(size_ranks, axis=1)
        weight_ranks = np.mean(weight_ranks, axis=1)
        speed_ranks = np.mean(speed_ranks, axis=1)

        # Prepare ranking table data for current k
        ranking_data_k = {
            'Algorithm': chosen_algorithms,
            'Solution Size Rank': np.round(size_ranks, 2),
            'Total Weight Rank': np.round(weight_ranks, 2),
            'Speed Rank': np.round(speed_ranks, 2),
            'Average Rank': np.round((size_ranks + weight_ranks + speed_ranks) / 3, 2)
        }
        ranking_df_k = pd.DataFrame(ranking_data_k).set_index('Algorithm')
        ranking_df_k = ranking_df_k.sort_values(by='Average Rank')

        # Save table data to CSV
        ranking_df_k.to_csv(f'../data/ranking_table_k_{k}_extended.csv')

    # Save the main figure with all plots
    plt.tight_layout(pad=1.0)
    plt.savefig('../images/solution_size_weight_comparison_all_extended.png', dpi=300)
    plt.close()
    
    # Log the conclusion of the function
    print("exhaustive_vs_algorithms_size_weight() - Done")

def quick_algo_graph(algorithm, filename):
    """ 
    Create a quick graph for the specified algorithm and save it to the given filename. 
    This is a helper function to quickly visualize the results of an algorithm. 

    Args:
    algorithm (str) : The algorithm to visualize
    filename (str) : The filename to load the data from and save the plot to
    """

    temp_df = pd.read_csv(DATA_FOLDER + filename)
    
    # Create a figure with subplots
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    x_values = temp_df[NODE_COUNT]

    # Plot Solution Size and Total Weight for the algorithm
    axes[0][0].plot(x_values, temp_df[SOLUTION_SIZE], label=algorithm, color='red')
    axes[0][1].plot(x_values, temp_df[TOTAL_WEIGHT], label=algorithm, color='blue')
    axes[1][0].plot(x_values, temp_df[EXECUTION_TIME], label=algorithm, color='green')
    axes[1][1].plot(x_values, temp_df[NUMBER_OF_OPERATIONS], label=algorithm, color='purple')
    
    # Set labels and legend for both plots
    axes[0][0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0][0].set_ylabel(SOLUTION_SIZE)
    axes[0][0].set_title('Solution Size for ' + algorithm)
    axes[0][0].legend()
    
    axes[0][1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0][1].set_ylabel(TOTAL_WEIGHT)
    axes[0][1].set_title('Total Weight of Solution for ' + algorithm)
    axes[0][1].legend()
    
    axes[1][0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1][0].set_ylabel(EXECUTION_TIME)
    axes[1][0].set_title('Execution Time for ' + algorithm)
    axes[1][0].legend()
    
    axes[1][1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1][1].set_ylabel(NUMBER_OF_OPERATIONS)
    axes[1][1].set_title('Number of Operations for ' + algorithm)
    axes[1][1].legend()
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(DATA_FOLDER + filename.replace('.csv', '.png'), dpi=300)
    plt.close()

def quick_algo_compare_graphs(algorithms, dataset=SMALL, k=None, output_filename='../images/quick_comparison.png'):
    """ 
    Create a quick comparison graph for the specified algorithms and save it to the given filename. 
    This is a helper function to quickly visualize the results of multiple algorithms. 

    Args:
    algorithms (list) : The algorithms to visualize
    filenames (list) : The filenames to load the data from
    """

    # Create a figure with subplots
    _, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Attribute a color to each algorithm
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Set plot labels and titles
    
    for i, algorithm in enumerate(algorithms):
        if dataset == SMALL:
            temp_df = dataframes_small[algorithm]
        elif dataset == BIG:
            temp_df = dataframes_big[algorithm]
        elif dataset == PREGEN:
            temp_df = dataframes_pregen[algorithm]

        x_values = temp_df[k if k else 0.75][NODE_COUNT]
        
        if k:
            temp_df = temp_df[k]
        else:
            temp_df = utils.total_average(temp_df)

        # Plot Solution Size and Total Weight for the algorithm
        axes[0][0].plot(x_values, temp_df[SOLUTION_SIZE], label=algorithm, color=colors[i])
        axes[0][1].plot(x_values, temp_df[TOTAL_WEIGHT], label=algorithm, color=colors[i])
        axes[1][0].plot(x_values, temp_df[EXECUTION_TIME], label=algorithm, color=colors[i])
        axes[1][1].plot(x_values, temp_df[NUMBER_OF_OPERATIONS], label=algorithm, color=colors[i])
    
    # Set labels and legend for both plots
    axes[0][0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0][0].set_ylabel(SOLUTION_SIZE)
    axes[0][0].set_title('Solution Size Comparison')
    axes[0][0].legend()
    
    axes[0][1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0][1].set_ylabel(TOTAL_WEIGHT)
    axes[0][1].set_title('Total Weight Comparison')
    axes[0][1].legend()
    
    axes[1][0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1][0].set_ylabel(EXECUTION_TIME)
    axes[1][0].set_title('Execution Time Comparison')
    axes[1][0].legend()
    
    axes[1][1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1][1].set_ylabel(NUMBER_OF_OPERATIONS)
    axes[1][1].set_title('Number of Operations Comparison')
    axes[1][1].legend()
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def compare_algorithm_k_values(algorithm, dataset=SMALL, output_filename='../images/k_comparison.png'):
    """ 
    Compare the k values of the algorithm and save the plot to the given filename. 
    This is a helper function to quickly visualize the results of an algorithm. 

    Args:
    algorithm (str) : The algorithm to visualize
    output_filename (str) : The filename to save the plot to
    """

    # Create a figure with subplots
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Attribute a color to each k value
    colors = ['red', 'blue', 'green', 'purple']
    
    # Set plot labels and titles
    for i, k in enumerate(k_full):
        if dataset == SMALL:
            temp_df = dataframes_small[algorithm]
        elif dataset == BIG:
            temp_df = dataframes_big[algorithm]
        elif dataset == PREGEN:
            temp_df = dataframes_pregen[algorithm]

        x_values = temp_df[k][NODE_COUNT]
        temp_df = temp_df[k]

        # Plot Execution Time and Number of Operations for the algorithm
        axes[0].plot(x_values, temp_df[EXECUTION_TIME], label=f'k={k}', color=colors[i])
        axes[1].plot(x_values, temp_df[NUMBER_OF_OPERATIONS], label=f'k={k}', color=colors[i])

    # Set labels and legend for both plots
    axes[0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0].set_ylabel(EXECUTION_TIME)
    axes[0].set_title(f'{algorithm} - Execution Time Comparison')
    axes[0].legend()

    axes[1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1].set_ylabel(NUMBER_OF_OPERATIONS)
    axes[1].set_title(f'{algorithm} - Number of Operations Comparison')
    axes[1].legend()

    # Save the plots
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def compare_algorithms_three(algorithms, k=None, dataset=SMALL, output_filename='../images/comparison_3.png'):
    """ 
    Compare the three main metrics of the algorithms and save the plot to the given filename.

    Args:
    algorithms (list) : The algorithms to visualize
    dataset (str) : The dataset to use for the comparison
    output_filename (str) : The filename to save the plot to
    """

    # Create a figure with subplots
    _, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Attribute a color to each algorithm
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Set plot labels and titles
    for i, algorithm in enumerate(algorithms):
        if dataset == SMALL:
            temp_df = dataframes_small[algorithm]
        elif dataset == BIG:
            temp_df = dataframes_big[algorithm]
        elif dataset == PREGEN:
            temp_df = dataframes_pregen[algorithm]

        x_values = temp_df[k if k else 0.75][NODE_COUNT]
        temp_df = temp_df[k] if k else utils.total_average(temp_df)

        alpha = 0.65 if algorithm != EXHAUSTIVE else 1
        # Plot Solution Size, Total Weight, Execution Time for the algorithm
        axes[0].plot(x_values, temp_df[TOTAL_WEIGHT], label=algorithm, color=colors[i], alpha=alpha)
        axes[1].plot(x_values, temp_df[EXECUTION_TIME], label=algorithm, color=colors[i], alpha=alpha)
        axes[2].plot(x_values, temp_df[NUMBER_OF_OPERATIONS], label=algorithm, color=colors[i], alpha=alpha)
    
    # Set labels and legend for all plots
    axes[0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0].set_ylabel(TOTAL_WEIGHT)
    axes[0].set_title('Total Weight Comparison')
    axes[0].legend()
    
    axes[1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1].set_ylabel(EXECUTION_TIME)
    axes[1].set_title('Execution Time Comparison (logarithmic scale)')
    axes[1].set_yscale('log')
    axes[1].legend()
    
    axes[2].set_xlabel(GRAPH_SIZE_AXIS)
    axes[2].set_ylabel(NUMBER_OF_OPERATIONS)
    axes[2].set_title('Number of Operations Comparison (logarithmic scale)')
    axes[2].set_yscale('log')
    axes[2].legend()
    
    # Save the plots
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def rank_algorithms(algorithms, dataset=SMALL, output_filename='../data/ranking_table.csv'):
    """ 
    Rank the algorithms based on their performance and save the results to the given filename.

    Args:
    algorithms (list) : The algorithms to rank
    dataset (str) : The dataset to use for the ranking
    output_filename (str) : The filename to save the ranking to
    """

    # Function to rank data from largest to smallest
    def rank_largest_to_smallest(data):
        sorted_indices = sorted(range(len(data)), key=lambda x: data[x], reverse=True)
        ranks = [0] * len(data)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1
        return ranks

    # Function to rank data from smallest to largest
    def rank_smallest_to_largest(data):
        sorted_indices = sorted(range(len(data)), key=lambda x: data[x])
        ranks = [0] * len(data)
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1
        return ranks

    # Create a DataFrame to store the ranking data
    ranking_data = {
        'Algorithm': algorithms,
        'Average Rank': np.zeros(len(algorithms))
    }
    ranking_df = pd.DataFrame(ranking_data).set_index('Algorithm')

    # Loop over each k value
    for k in k_full:
        if dataset == SMALL:
            temp_dfs = [dataframes_small[algorithm][k] for algorithm in algorithms]
        elif dataset == BIG:
            temp_dfs = [dataframes_big[algorithm][k] for algorithm in algorithms]
        elif dataset == PREGEN:
            temp_dfs = [dataframes_pregen[algorithm][k] for algorithm in algorithms]

        # Calculate ranks per graph instance, handling ties
        size_ranks = np.zeros((len(algorithms), len(temp_dfs[0])))
        weight_ranks = np.zeros((len(algorithms), len(temp_dfs[0])))
        speed_ranks = np.zeros((len(algorithms), len(temp_dfs[0])))

        for i in range(len(temp_dfs[0])):
            solution_sizes = [temp_df[SOLUTION_SIZE][i] for temp_df in temp_dfs]
            solution_weights = [temp_df[TOTAL_WEIGHT][i] for temp_df in temp_dfs]
            solution_speeds = [temp_df[EXECUTION_TIME][i] for temp_df in temp_dfs]

            # Use rankdata with 'min' to rank from largest to smallest (1 is best, N is worst)
            size_ranks[:, i] = rank_largest_to_smallest(solution_sizes)
            weight_ranks[:, i] = rank_largest_to_smallest(solution_weights)
            speed_ranks[:, i] = rank_smallest_to_largest(solution_speeds)

        # Calculate mean ranks across all graph sizes
        size_ranks = np.mean(size_ranks, axis=1)
        weight_ranks = np.mean(weight_ranks, axis=1)

        # Update the ranking DataFrame with the average rank
        ranking_df.loc[algorithms, 'Average Rank'] += (size_ranks + weight_ranks + speed_ranks) / 3

    # Normalize the average rank by the number of k values
    ranking_df['Average Rank'] /= len(k_full)

    # Save the ranking data to a CSV file
    ranking_df.to_csv(output_filename)

    # Log the conclusion of the function
    print("rank_algorithms() - Done")

if __name__ == "__main__":
    # Compare the graphs of the exhaustive, weighted to degree, and all randomized algorithms, using the small dataset    
    #quick_algo_compare_graphs([EXHAUSTIVE, WEIGHT_TO_DEGREE, MONTE_CARLO, PARALLEL_HEURISTIC_MONTE_CARLO, SIMULATED_ANNEALING], dataset=SMALL, k=0.5, output_filename='../images/quick_comparison_small.png')
    # Compare the k values of the monte carlo algorithm using the small dataset
    compare_algorithm_k_values(MONTE_CARLO, dataset=SMALL, output_filename='../images/k_comparison_monte_carlo_small.png')
    compare_algorithm_k_values(MONTE_CARLO, dataset=BIG, output_filename='../images/k_comparison_monte_carlo_big.png')

    # Compare the three main metrics of the algorithms using the small dataset
    compare_algorithms_three([EXHAUSTIVE, WEIGHT_TO_DEGREE, MONTE_CARLO, PARALLEL_HEURISTIC_MONTE_CARLO, SIMULATED_ANNEALING], dataset=SMALL, output_filename='../images/comparison_3_small_average.png')

    # Calculate how much more efficient the parallel heuristic monte carlo algorithm is compared to the monte carlo algorithm
    k = 0.125
    ratio = dataframes_small[MONTE_CARLO][k][EXECUTION_TIME].mean() / dataframes_small[PARALLEL_HEURISTIC_MONTE_CARLO][k][EXECUTION_TIME].mean()
    print(f"The Parallel Heuristic Monte Carlo algorithm is {ratio:.2f} times more efficient than the Monte Carlo algorithm.")

    # Error ratio and accuracy of the greedy algorithms in relation to the exhaustive algorithm
    exhaustive_vs_greedy_error_ratio_and_accuracy()
    