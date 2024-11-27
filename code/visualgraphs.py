from scipy.stats import rankdata
from algorithms import algorithms
import matplotlib.pyplot as plt
from utils import utils
import pandas as pd
import numpy as np
import matplotlib

# This file is a collection of functions that generate the visualizations for the project report
# It is sort of disorganized because I had to move the functions from the Jupyter Notebook to this file

""" Constants and file paths for the visualizations """
k_full = [0.125, 0.25, 0.5, 0.75]
k_values = [125, 25, 50, 75]

# File paths for the data
OLD_EXHAUSTIVE_PATH = "../data/exhaustive_v1/exhaustive_v1_p_{}.csv"
EXHAUSTIVE_PATH = "../data/exhaustive/exhaustive_p_{}.csv"
BIGGEST_WEIGHT_FIRST_PATH = "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_{}.csv"
SMALLEST_DEGREE_FIRST_PATH = "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_{}.csv"
WEIGHT_TO_DEGREE_PATH = "../data/weight_to_degree_compare/weight_to_degree_compare_p_{}.csv"

# Constants used for the plots
SOLUTION_SIZE = 'Solution Size'
GRAPH_SIZE_AXIS = 'Graph Size (|V|)'
TOTAL_WEIGHT = 'Total Weight'
EXECUTION_TIME = 'Execution Time (seconds)'

file_paths = {
    0.125: [EXHAUSTIVE_PATH.format(125),
            BIGGEST_WEIGHT_FIRST_PATH.format(125),
            SMALLEST_DEGREE_FIRST_PATH.format(125),
            WEIGHT_TO_DEGREE_PATH.format(125)],
    0.25: [EXHAUSTIVE_PATH.format(25),
           BIGGEST_WEIGHT_FIRST_PATH.format(25),
           SMALLEST_DEGREE_FIRST_PATH.format(25),
           WEIGHT_TO_DEGREE_PATH.format(25)],
    0.5: [EXHAUSTIVE_PATH.format(50),
          BIGGEST_WEIGHT_FIRST_PATH.format(50),
          SMALLEST_DEGREE_FIRST_PATH.format(50),
          WEIGHT_TO_DEGREE_PATH.format(50)],
    0.75: [EXHAUSTIVE_PATH.format(75),
           BIGGEST_WEIGHT_FIRST_PATH.format(75),
           SMALLEST_DEGREE_FIRST_PATH.format(75),
           WEIGHT_TO_DEGREE_PATH.format(75)]
}

""" Functions for the various graphs and tables in the report """
def remarks_graphs():
    """ Create the graphs for the remarks section (Fig.1) """

    # Load data
    df12_5 = pd.read_csv(EXHAUSTIVE_PATH.format(125))
    df25 = pd.read_csv(EXHAUSTIVE_PATH.format(25))
    df50 = pd.read_csv(EXHAUSTIVE_PATH.format(50))
    df75 = pd.read_csv(EXHAUSTIVE_PATH.format(75))

    # Calculate solution size
    df12_5[SOLUTION_SIZE] = df12_5['Solution'].apply(len)
    df25[SOLUTION_SIZE] = df25['Solution'].apply(len)
    df50[SOLUTION_SIZE] = df50['Solution'].apply(len)
    df75[SOLUTION_SIZE] = df75['Solution'].apply(len)

    # Set up subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Solution Size
    axes[0].plot(df12_5['Solution Size'], label='k=0.125')
    axes[0].plot(df25['Solution Size'], label='k=0.25')
    axes[0].plot(df50['Solution Size'], label='k=0.50')
    axes[0].plot(df75['Solution Size'], label='k=0.75')
    axes[0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0].set_ylabel('Solution Size')
    axes[0].set_title('Evolution of Solution Size for Different k Values')
    axes[0].legend()

    # Plot Total Weight of Solution
    axes[1].plot(df12_5[TOTAL_WEIGHT], label='k=0.125')
    axes[1].plot(df25[TOTAL_WEIGHT], label='k=0.25')
    axes[1].plot(df50[TOTAL_WEIGHT], label='k=0.50')
    axes[1].plot(df75[TOTAL_WEIGHT], label='k=0.75')
    axes[1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1].set_ylabel(TOTAL_WEIGHT)
    axes[1].set_title('Evolution of Solution Weight for Different k Values')
    axes[1].legend()

    # Show the plots
    plt.tight_layout()

    plt.savefig('../images/evolution_of_solution.png', dpi=300)
    plt.show()

def exhaustive_comparison_time():
    """ Compare the execution time of the two exhaustive algorithms (Fig.2) """
    
    # Read data for all graph sizes
    df_v1_125 = pd.read_csv(OLD_EXHAUSTIVE_PATH.format(125))
    df_v1_25 = pd.read_csv(OLD_EXHAUSTIVE_PATH.format(25))
    df_v1_50 = pd.read_csv(OLD_EXHAUSTIVE_PATH.format(50))
    df_v1_75 = pd.read_csv(OLD_EXHAUSTIVE_PATH.format(75))

    df_v1 = {
        125: df_v1_125,
        25: df_v1_25,
        50: df_v1_50,
        75: df_v1_75
    }

    df_v2_125 = pd.read_csv(EXHAUSTIVE_PATH.format(125))
    df_v2_25 = pd.read_csv(EXHAUSTIVE_PATH.format(25))
    df_v2_50 = pd.read_csv(EXHAUSTIVE_PATH.format(50))
    df_v2_75 = pd.read_csv(EXHAUSTIVE_PATH.format(75))

    df_v2 = {
        125: df_v2_125,
        25: df_v2_25,
        50: df_v2_50,
        75: df_v2_75
    }

    # Extract data for execution times by graph size and k
    algorithm1_times = {k: df_v1[k]['Execution Time (seconds)'].to_list() for k in k_values}
    algorithm2_times = {k: df_v2[k]['Execution Time (seconds)'].to_list() for k in k_values}

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
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Layout adjustments
    plt.tight_layout()

    # Save the plot
    plt.savefig('../images/execution_time_comparison_average.png', dpi=300)
    plt.close()

def exhaustive_comparison_operations():
    """ Compare the number of operations of the two exhaustive algorithms (Fig.3) """
    
    # Read data for all graph sizes
    df_v1_125 = pd.read_csv("../data/exhaustive_v1/exhaustive_v1_p_125.csv")
    df_v1_25 = pd.read_csv("../data/exhaustive_v1/exhaustive_v1_p_25.csv")
    df_v1_50 = pd.read_csv("../data/exhaustive_v1/exhaustive_v1_p_50.csv")
    df_v1_75 = pd.read_csv("../data/exhaustive_v1/exhaustive_v1_p_75.csv")

    df_v1 = {
        125: df_v1_125,
        25: df_v1_25,
        50: df_v1_50,
        75: df_v1_75
    }

    df_v2_125 = pd.read_csv("../data/exhaustive/exhaustive_p_125.csv")
    df_v2_25 = pd.read_csv("../data/exhaustive/exhaustive_p_25.csv")
    df_v2_50 = pd.read_csv("../data/exhaustive/exhaustive_p_50.csv")
    df_v2_75 = pd.read_csv("../data/exhaustive/exhaustive_p_75.csv")

    df_v2 = {
        125: df_v2_125,
        25: df_v2_25,
        50: df_v2_50,
        75: df_v2_75
    }

    k_values = [125, 25, 50, 75]  # Use actual values

    # Extract data for number of operations by graph size and k
    algorithm1_ops = {k: df_v1[k]['Number of Operations'] for k in k_values}
    algorithm2_ops = {k: df_v2[k]['Number of Operations'] for k in k_values}

    # Find the maximum graph size across all k values
    max_graph_size_v1 = max(len(ops) for ops in algorithm1_ops.values())
    max_graph_size_v2 = max(len(ops) for ops in algorithm2_ops.values())

    # Function to calculate the mean with available data
    def calculate_mean(ops_dict, graph_size):
        values = [ops[graph_size] for ops in ops_dict.values() if graph_size < len(ops)]
        return np.mean(values) if values else np.nan

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
    plt.ylabel('Number of Operations')
    plt.title('Average Number of Operations by Graph Size (for k in [12.5, 25, 50, 75])')

    # Use logarithmic scale for y-axis
    plt.yscale('log')

    # Adjust legend to stay inside the plot
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Layout adjustments
    plt.tight_layout()

    # Save the plot
    plt.savefig('../images/number_of_operations_comparison_average.png', dpi=300)
    plt.close()
    
def greedy_comparison_operations_time():
    """ Compare the number of operations and execution time of the three greedy algorithms (Fig.4) """

    # Read data for all graph sizes for the three algorithms
    df_biggest_weight_first_125 = pd.read_csv("../data/biggest_weight_first/biggest_weight_first_p_125.csv")
    df_biggest_weight_first_25 = pd.read_csv("../data/biggest_weight_first/biggest_weight_first_p_25.csv")
    df_biggest_weight_first_50 = pd.read_csv("../data/biggest_weight_first/biggest_weight_first_p_50.csv")
    df_biggest_weight_first_75 = pd.read_csv("../data/biggest_weight_first/biggest_weight_first_p_75.csv")

    df_smallest_degree_first_125 = pd.read_csv("../data/smallest_degree_first/smallest_degree_first_p_125.csv")
    df_smallest_degree_first_25 = pd.read_csv("../data/smallest_degree_first/smallest_degree_first_p_25.csv")
    df_smallest_degree_first_50 = pd.read_csv("../data/smallest_degree_first/smallest_degree_first_p_50.csv")
    df_smallest_degree_first_75 = pd.read_csv("../data/smallest_degree_first/smallest_degree_first_p_75.csv")

    df_weight_to_degree_125 = pd.read_csv("../data/weight_to_degree/weight_to_degree_p_125.csv")
    df_weight_to_degree_25 = pd.read_csv("../data/weight_to_degree/weight_to_degree_p_25.csv")
    df_weight_to_degree_50 = pd.read_csv("../data/weight_to_degree/weight_to_degree_p_50.csv")
    df_weight_to_degree_75 = pd.read_csv("../data/weight_to_degree/weight_to_degree_p_75.csv")

    # Store data for each algorithm
    df_biggest_weight_first = {
        125: df_biggest_weight_first_125,
        25: df_biggest_weight_first_25,
        50: df_biggest_weight_first_50,
        75: df_biggest_weight_first_75
    }

    df_smallest_degree_first = {
        125: df_smallest_degree_first_125,
        25: df_smallest_degree_first_25,
        50: df_smallest_degree_first_50,
        75: df_smallest_degree_first_75
    }

    df_weight_to_degree = {
        125: df_weight_to_degree_125,
        25: df_weight_to_degree_25,
        50: df_weight_to_degree_50,
        75: df_weight_to_degree_75
    }

    k_values = [125, 25, 50, 75]  # Use actual values

    # Extract data for time and number of operations by graph size and k
    def extract_data(df_dict, k_values, column_name):
        return {k: df_dict[k][column_name] for k in k_values}

    biggest_weight_first_ops = extract_data(df_biggest_weight_first, k_values, 'Number of Operations')
    smallest_degree_first_ops = extract_data(df_smallest_degree_first, k_values, 'Number of Operations')
    weight_to_degree_ops = extract_data(df_weight_to_degree, k_values, 'Number of Operations')

    # Extract Execution Time data (Assuming there's a 'Execution Time' column in the CSVs)
    biggest_weight_first_time = extract_data(df_biggest_weight_first, k_values, 'Execution Time (seconds)')
    smallest_degree_first_time = extract_data(df_smallest_degree_first, k_values, 'Execution Time (seconds)')
    weight_to_degree_time = extract_data(df_weight_to_degree, k_values, 'Execution Time (seconds)')

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

    avg_biggest_weight_first_ops, avg_biggest_weight_first_time = calculate_avg_data(biggest_weight_first_ops, biggest_weight_first_time)
    avg_smallest_degree_first_ops, avg_smallest_degree_first_time = calculate_avg_data(smallest_degree_first_ops, smallest_degree_first_time)
    avg_weight_to_degree_ops, avg_weight_to_degree_time = calculate_avg_data(weight_to_degree_ops, weight_to_degree_time)

    # Plotting both graphs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors for distinguishing lines
    colors = matplotlib.colormaps["tab10"]  # Use the updated method for color mapping

    scale = 100

    # Plot 1: Number of Operations
    axes[0].plot([100 +x * scale for x in range(len(avg_biggest_weight_first_ops))], avg_biggest_weight_first_ops, color=colors(1), linewidth=1.5, label='WMax - Biggest Weight First')
    axes[0].plot([100 +x * scale for x in range(len(avg_smallest_degree_first_ops))], avg_smallest_degree_first_ops, color=colors(2), linewidth=1.5, label='DMin - Smallest Degree First')
    axes[0].plot([100 +x * scale for x in range(len(avg_weight_to_degree_ops))], avg_weight_to_degree_ops, color=colors(4), linewidth=1.5, label='WDMix - Weight to Degree')

    axes[0].set_xlabel(GRAPH_SIZE_AXIS)
    axes[0].set_ylabel('Number of Operations')
    axes[0].set_title('Average Number of Operations by Graph Size (for k in [12.5, 25, 50, 75])')
    axes[0].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Plot 2: Execution Time
    axes[1].plot([100 +x * scale for x in range(len(avg_biggest_weight_first_time))], avg_biggest_weight_first_time, color=colors(1), linewidth=1.5, label='WMax - Biggest Weight First')
    axes[1].plot([100 +x * scale for x in range(len(avg_smallest_degree_first_time))], avg_smallest_degree_first_time, color=colors(2), linewidth=1.5, label='DMin - Smallest Degree First')
    axes[1].plot([100 +x * scale for x in range(len(avg_weight_to_degree_time))], avg_weight_to_degree_time, color=colors(4), linewidth=1.5, label='WDMix - Weight to Degree')

    axes[1].set_xlabel(GRAPH_SIZE_AXIS)
    axes[1].set_ylabel('Execution Time (s)')
    axes[1].set_title('Average Execution Time by Graph Size (for k in [12.5, 25, 50, 75])')
    axes[1].legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize='small')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('../images/greedy_comparison_time_and_operations.png', dpi=300)
    plt.close()

def solution_comparison():
    """ Compare the solutions found by the algorithms (Fig.5) """
    
    # Visualize one of the graphs and the solutions found by the algorithms
    algorithms_sets = {}

    k = 0.5
    test_graph = utils.create_graph_v4(20, k)

    algorithms_sets['Exhaustive'], _ = algorithms.exhaustive_v2(test_graph)
    algorithms_sets['Biggest Weight First'], _ = algorithms.biggest_weight_first_v2(test_graph)
    algorithms_sets['Smallest Degree First'], _ = algorithms.smallest_degree_first_v1(test_graph)
    algorithms_sets['Weight to Degree'], _ = algorithms.weight_to_degree_v1(test_graph)

    utils.visual_compare_algorithms(test_graph, algorithms_sets, k=k, store_png=True)
    
def exhaustive_vs_greedy_size_weight():
    """ Compare the solution size and total weight of the greedy algorithms with the exhaustive algorithm (Fig.6) """
    """ Also generates the tables for the ranking of the algorithms (Tables 1 and 2) """

    # Create a figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)

    # Loop over each k value
    for idx, k in enumerate(k_full):
        # Load data for each algorithm
        df1 = pd.read_csv(file_paths[k][0])
        df2 = pd.read_csv(file_paths[k][1])
        df3 = pd.read_csv(file_paths[k][2])
        df4 = pd.read_csv(file_paths[k][3])

        space = 39 if k == 0.125 else 55 if k == 0.25 else 118 if k == 0.5 else 238

        # Calculate Solution Size and Total Weight for each dataframe
        df1['Solution Size'] = df1['Solution'].apply(len)
        df2['Solution Size'] = df2['Solution'].apply(len)
        df3['Solution Size'] = df3['Solution'].apply(len)
        df4['Solution Size'] = df4['Solution'].apply(len)

        # Plot Solution Size
        df2['Solution Size'].head(space).plot(ax=axes[idx, 0], label='WMax - Biggest Weight First', color='orange', alpha=0.65)
        df3['Solution Size'].head(space).plot(ax=axes[idx, 0], label='DMin - Smallest Degree First', color='green', alpha=0.65)
        df4['Solution Size'].head(space).plot(ax=axes[idx, 0], label='WDMix -Weight to Degree', color='purple', alpha=0.65)
        df1['Solution Size'].head(space).plot(ax=axes[idx, 0], label='Exhaustive', color='red', linewidth=2, zorder=10)

        # Set labels for Solution Size plot
        axes[idx, 0].set_xlabel(GRAPH_SIZE_AXIS, fontsize=8)
        axes[idx, 0].set_ylabel('Solution Size', fontsize=8)
        axes[idx, 0].set_title(f'Solution Size for k={k}', fontsize=10)
        axes[idx, 0].legend(fontsize=7, loc='upper left')

        # Plot Total Weight
        df2[TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label='WMax - Biggest Weight First', color='orange', alpha=0.65)
        df3[TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label='DMin - Smallest Degree First', color='green', alpha=0.65)
        df4[TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label='WDMix - Weight to Degree', color='purple', alpha=0.65)
        df1[TOTAL_WEIGHT].head(space).plot(ax=axes[idx, 1], label='Exhaustive', color='red', linewidth=2, zorder=10)

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
            solution_sizes = [df1['Solution Size'][i], df2['Solution Size'][i], df3['Solution Size'][i], df4['Solution Size'][i]]
            solution_weights = [df1[TOTAL_WEIGHT][i], df2[TOTAL_WEIGHT][i], df3[TOTAL_WEIGHT][i], df4[TOTAL_WEIGHT][i]]
            solution_speeds = [df1['Execution Time (seconds)'][i], df2['Execution Time (seconds)'][i], df3['Execution Time (seconds)'][i], df4['Execution Time (seconds)'][i]]
            
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
            'Algorithm': ['Exhaustive', 'Biggest Weight First', 'Smallest Degree First', 'Weight to Degree'],
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
    
def exhaustive_vs_greedy_error_ratio():
    """ Compare the error ratio of the greedy algorithms in relation to the exhaustive algorithm (Fig.7) """

    # Define k values and file paths
    k_values = [0.125, 0.25, 0.5, 0.75]


    # Initialize the plot grid (2x2 layout)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # 2x2 grid
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier iteration

    # Define the colors for each algorithm
    algorithm_colors = {
        'WMax - Biggest Weight First': 'orange',
        'DMin - Smallest Degree First': 'green',
        'WDMix - Weight to Degree': 'purple'
    }

    # Iterate through each k value
    for idx, (k, paths) in enumerate(file_paths.items()):
        # Load the CSV files for each algorithm
        df_exhaustive = pd.read_csv(paths[0])
        df_biggest_weight_first = pd.read_csv(paths[1])
        df_smallest_degree_first = pd.read_csv(paths[2])
        df_weight_to_degree = pd.read_csv(paths[3])
        
        # Limit heuristic algorithms to the same solution count as the exhaustive algorithm
        solution_count = len(df_exhaustive)  # Get the solution count for exhaustive
        df_exhaustive = df_exhaustive.head(solution_count)
        df_biggest_weight_first = df_biggest_weight_first.head(solution_count)
        df_smallest_degree_first = df_smallest_degree_first.head(solution_count)
        df_weight_to_degree = df_weight_to_degree.head(solution_count)
        
        # Structure data in a dictionary for the heuristics
        algorithms = {
            'WMax - Biggest Weight First': df_biggest_weight_first,
            'DMin - Smallest Degree First': df_smallest_degree_first,
            'WDMix - Weight to Degree': df_weight_to_degree
        }

        # Plot error (the difference between greedy and exhaustive solution weight)
        for name, df in algorithms.items():
            # Calculate error as the difference between exhaustive and greedy solution weight, normalized by exhaustive weight
            error = (df_exhaustive[TOTAL_WEIGHT].values - df[TOTAL_WEIGHT].values) / df_exhaustive[TOTAL_WEIGHT].values
            
            # Plot with the respective color and label
            axs[idx].plot(df_exhaustive['Node Count'], error, label=name, color=algorithm_colors[name], alpha=0.65)

        # Set plot labels and title
        axs[idx].set_xlabel(GRAPH_SIZE_AXIS, fontsize=12)
        axs[idx].set_ylabel('Error (Ratio)', fontsize=12)
        axs[idx].set_title(f'Error (Greedy - Exhaustive) for k={k}', fontsize=14)
        axs[idx].legend(loc='upper right', fontsize=10)

        # Set the grid for better readability
        axs[idx].grid(True)

    # Adjust layout for the 2x2 grid
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)  # Adjust horizontal and vertical space

    # Save or show the plot
    plt.savefig('../images/error_grid_2x2.png', dpi=300)
    plt.show()
    
def exhaustive_vs_greedy_deviations_accuracy():
    """ Generates the content for the tables of deviations and accuracy for the greedy algorithms (Tables 3 and 4) """

    # Define k values and file paths
    k_values = [0.125, 0.25, 0.5, 0.75]
    file_paths = {
        0.125: ["../data/exhaustive/exhaustive_p_125.csv",
                "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_125.csv",
                "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_125.csv",
                "../data/weight_to_degree_compare/weight_to_degree_compare_p_125.csv"],
        0.25: ["../data/exhaustive/exhaustive_p_25.csv",
            "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_25.csv",
            "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_25.csv",
            "../data/weight_to_degree_compare/weight_to_degree_compare_p_25.csv"],
        0.5: ["../data/exhaustive/exhaustive_p_50.csv",
            "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_50.csv",
            "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_50.csv",
            "../data/weight_to_degree_compare/weight_to_degree_compare_p_50.csv"],
        0.75: ["../data/exhaustive/exhaustive_p_75.csv",
            "../data/biggest_weight_first_compare/biggest_weight_first_compare_p_75.csv",
            "../data/smallest_degree_first_compare/smallest_degree_first_compare_p_75.csv",
            "../data/weight_to_degree_compare/weight_to_degree_compare_p_75.csv"]
    }

    # Dictionary to store the results for each k
    accuracy_precision_results = {}

    # Loop over each k value
    for k in k_values:
        # Load data for each algorithm
        df_exhaustive = pd.read_csv(file_paths[k][0])
        df_biggest_weight_first = pd.read_csv(file_paths[k][1])
        df_smallest_degree_first = pd.read_csv(file_paths[k][2])
        df_weight_to_degree = pd.read_csv(file_paths[k][3])
        
        # Limit heuristic algorithms to the same solution count as the exhaustive algorithm
        solution_count = len(df_exhaustive)  # Get the solution count for exhaustive
        df_biggest_weight_first = df_biggest_weight_first.head(solution_count)
        df_smallest_degree_first = df_smallest_degree_first.head(solution_count)
        df_weight_to_degree = df_weight_to_degree.head(solution_count)
        
        # Create a dictionary of the heuristic algorithms
        algorithms = {
            'Biggest Weight First': df_biggest_weight_first,
            'Smallest Degree First': df_smallest_degree_first,
            'Weight to Degree': df_weight_to_degree
        }
        
        # Calculate accuracy and precision for solution weight
        accuracy_precision_df_weight = utils.calculate_accuracy_precision(df_exhaustive, algorithms, metric=TOTAL_WEIGHT)
        accuracy_precision_df_weight['k'] = k  # Add k value for identification
        
        # Store the resulting DataFrame in the dictionary
        accuracy_precision_results[k] = accuracy_precision_df_weight

    # Concatenate results for all k values into a single DataFrame
    final_accuracy_precision_df = pd.concat(accuracy_precision_results.values(), ignore_index=True)

    # Save to CSV
    final_accuracy_precision_df.to_csv('../data/accuracy_precision_summary.csv', index=False)
    
if __name__ == "__main__":
    remarks_graphs()
    exhaustive_comparison_time()
    exhaustive_comparison_operations()
    exhaustive_vs_greedy_size_weight()
    greedy_comparison_operations_time()
    solution_comparison()
    exhaustive_vs_greedy_error_ratio()
    exhaustive_vs_greedy_deviations_accuracy()