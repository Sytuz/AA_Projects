import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import threading
import random
import pickle
import time
import csv
import os

# Utility class with required functions
class utils:

    @staticmethod
    # Function to create a graph with n nodes and edge probability p, with a fixed seed for structure and weight generation
    def create_graph_v4(n, p, graph_seed=108122, weight_seed=108122):
        # Create a graph with a fixed seed for structure
        G = nx.erdos_renyi_graph(n, p, seed=graph_seed)
        
        # Set up a separate Random instance for consistent node weights
        rand_instance = random.Random(weight_seed)
        for i in range(n):
            G.nodes[i]['weight'] = rand_instance.randint(1, 100)
        
        return G

    @staticmethod
    def graph_creation_and_save(n, p, step, output_dir='../graphs', file_name='graphs', max_workers=2):
        """
        Function to create and save graphs to a file, with progress tracking using tqdm.

        Args:
            n (int): Maximum size of graphs.
            p (float): Edge probability.
            step (int): Step size for graph increments.
            output_dir (str): Directory to save the graphs.
            file_name (str): Base file name for saving.
            max_workers (int): Number of workers (not used here but included for future expansion).

        Returns:
            str: Path to the output file.
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # File to store all graphs for the given p
        output_filename = os.path.join(output_dir, f"{file_name}_p_{p}.pkl")
        
        # Define sizes from 0 to n, in increments of step
        sizes = range(step, n + 1, step)
        
        # Create a dictionary to store the graphs
        graph_data = {}
        
        # Generate graphs with tqdm progress bar
        print("Starting graph generation...")
        with tqdm(total=len(sizes), desc=f"Generating Graphs for k={p}", unit="graph") as pbar:
            for size in sizes:
                graph_data[size] = utils.create_graph_v4(size, p)
                pbar.update(1)  # Update the progress bar after each iteration

        # Save all the graphs to a file (overwrite the file)
        with open(output_filename, 'wb') as f:
            pickle.dump(graph_data, f)
        
        print("All graphs generated and saved.")
        return output_filename

    @staticmethod
    # Function to load graphs from a file
    def load_graphs(filename):
        # Load all graphs from a single file
        with open(filename, 'rb') as f:
            graphs = pickle.load(f)
        return graphs

    @staticmethod
    # Function to generate all subsets of a given set
    def all_subsets(s):
        return itertools.chain(*map(lambda x: itertools.combinations(s, x), range(0, len(s)+1)))

    @staticmethod
    # Function to check if a given subset is an independent set in a graph
    def is_independent_set(G, subset):
        for u, v in itertools.combinations(subset, 2):
            if G.has_edge(u, v):
                return False
        return True

    @staticmethod
    # Function to pretty print the graph
    def graph_print(G):
        data = G.nodes.data()
        print("Node | Weight")
        print("-----+-------")
        for node in data:
            print(node[0], "   |", node[1]['weight'])
        print()

    @staticmethod
    def stress_test(func, p, max_time_minutes, filename="stress_test_results.csv", save_results=True, n_max=1000, sample_size=5, stored_graphs=True, iterations=None):
        """Runs a stress test for a single p value and optionally saves the results to a specified CSV file."""
        
        max_time_seconds = max_time_minutes * 60
        stored_graphs_filename = f"../graphs/graphs_p_{p}.pkl" if "compare" not in filename else f"../graphs/small_graphs_p_{p}.pkl"
        graphs = None
        if stored_graphs:
            graphs = list(utils.load_graphs(stored_graphs_filename).values())
            n_max = len(graphs) * sample_size

        # If save_results is True, open the file for writing
        if save_results:
            filename = f"../data/{filename}"
            file = open(filename, mode='w', newline='')
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(["Node Count", "Number of Operations", "Total Weight", "Execution Time (seconds)", "Solution Size", "Solution"])
        else:
            file = None
            writer = None

        print("  n  | No of Operations | Total Weight | Time (s) | Solution Size | Solution ")
        
        # Stress test loop
        n = sample_size
        while n <= n_max:
            if stored_graphs:
                G = graphs[n // sample_size - 1]
            else:
                # Generate graph with n nodes and edge probability p
                G = utils.create_graph_v4(n, p)
            
            # Measure the time taken by the function on the generated graph
            result_container = {}
            elapsed_time = 0

            def run_func():
                """Wrapper function to execute and store the results."""
                if iterations is not None:
                    result_container['result'], result_container['op'] = func(G, iterations=iterations)
                else:
                    result_container['result'], result_container['op'] = func(G)

            # Create a thread to run the function
            thread = threading.Thread(target=run_func)
            start = time.time()
            thread.start()
            thread.join(timeout=max_time_seconds)
            end = time.time()

            # Check if the thread finished
            if thread.is_alive():
                print(f"Execution for n = {n} exceeded the time limit of {max_time_minutes} minutes. Stopping stress test.")
                thread.join()  # Ensure thread is cleaned up
                break
            else:
                elapsed_time = end - start

            # Calculate results
            result = result_container.get('result', set())
            op = result_container.get('op', 0)
            total_weight = sum(G.nodes[node]['weight'] for node in result)
            solution_size = len(result)
            result_str = str(result)
            
            # Write results to CSV if saving is enabled
            if save_results:
                writer.writerow([n, op, total_weight, elapsed_time, solution_size, result_str])
            
            # Print the result for the current size
            print(f"{n:4} | {op:16} | {total_weight:12} | {elapsed_time:8.6f} | {solution_size:13} | {result_str} ")
            
            # Increment node count for next iteration
            n += sample_size

        # Close the file if it was opened
        if file:
            file.close()


    @staticmethod
    def full_stress_test(func, p_values=[0.125, 0.25, 0.5, 0.75], max_time_minutes=2, 
                        base_filename="full_stress_test_results", save_results=True, 
                        n_max=1000, sample_size=5, stored_graphs=True, 
                        iterations=None):
        """
        Runs stress tests for a function across multiple p values and optionally iterations.
        Results are saved in a structured directory format.

        Args:
            func: The function to test.
            p_values: A list of edge probabilities to test.
            max_time_minutes: Maximum time allowed for a single test.
            base_filename: Base name for the results directory.
            save_results: Whether to save the results to CSV files.
            n_max: Maximum number of nodes in a test graph.
            sample_size: Increment size for nodes in each test.
            stored_graphs: Whether to use pre-generated graphs.
            max_iterations: Maximum iterations for algorithms that require them (optional).
            iteration_step: Incremental step for iterations (required if max_iterations is given).
        """

        # Base directory for all results
        base_folder = f"../data/{base_filename}"
        
        if save_results:
            # Clean up and recreate the base folder
            if os.path.exists(base_folder):
                for item in os.listdir(base_folder):
                    item_path = os.path.join(base_folder, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        for sub_item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, sub_item)
                            if os.path.isfile(sub_item_path):
                                os.remove(sub_item_path)
                        os.rmdir(item_path)

            else:
                os.makedirs(base_folder)
        
        # Loop over each edge probability (p value)
        for p in p_values:
            print(f"Starting stress tests for p = {p}")

            # Create a folder for the current p value
            p_folder = os.path.join(base_folder, f"p_{int(p * 1000) if p == 0.125 else int(p * 100)}")
            os.makedirs(p_folder, exist_ok=True)

            if iterations is not None:
                # Iteration-based testing
                for iteration_count in iterations:
                    print(f"  Testing with {iteration_count} iterations...")

                    # Generate a unique filename for this test
                    filename = os.path.join(p_folder, f"results_{iteration_count}.csv")

                    # Run the stress test and save results
                    utils.stress_test(
                        func, p, max_time_minutes, filename=filename,
                        save_results=save_results, n_max=n_max,
                        sample_size=sample_size, stored_graphs=stored_graphs,
                        iterations=iteration_count
                    )
            else:                
                # Generate a unique filename for this test
                filename = os.path.join(p_folder, "results.csv")

                # Run the stress test and save results
                utils.stress_test(
                    func, p, max_time_minutes, filename=filename,
                    save_results=save_results, n_max=n_max,
                    sample_size=sample_size, stored_graphs=stored_graphs
                )
            
            print(f"Completed stress tests for p = {p}.\n")


    @staticmethod
    def visualize_solution(G, mwis_set, store_png=False):
        """Visualizes the input graph and the solution set."""
        # Create a color map where MWIS nodes are red and others are blue
        node_colors = ['red' if node in mwis_set else 'blue' for node in G.nodes]
        
        # Create labels to show the weight of each node
        node_labels = {node: G.nodes[node].get('weight', 1) for node in G.nodes}  # Default weight is 1 if not specified
        
        # Define layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph with node colors and labels
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=500, font_size=10)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='white', font_weight='bold')
        
        # Save the graph as a PNG file if store_png is True
        if store_png:
            plt.savefig('../images/MWIS_solution.png')
        
        # Display the graph
        plt.show()

    @staticmethod
    def visual_compare_algorithms(G, algorithm_mwis_sets, k=None, store_png=False):
        """
        Visualizes and compares the MWIS results of different algorithms.
        
        Parameters:
        - G: The input graph.
        - algorithm_mwis_sets: Dictionary where keys are algorithm names (str) and values are sets of nodes 
                               in the MWIS for each algorithm.
        - store_png: Boolean indicating whether to save the plot as a PNG file.
        """
        # Define layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Number of algorithms
        num_algorithms = len(algorithm_mwis_sets)
        
        # Determine the layout of subplots in a 2x2 grid
        rows = (num_algorithms + 1) // 2  # Number of rows (2 columns per row)
        
        # Set up subplots in a 2x2 or equivalent grid
        fig, axes = plt.subplots(rows, 2, figsize=(12, 6 * rows))
        axes = axes.flatten()  # Flatten axes array for easier indexing
        
        for i, (algo_name, mwis_set) in enumerate(algorithm_mwis_sets.items()):
            # Select the current subplot
            ax = axes[i]
            
            # Determine node colors: red for MWIS nodes, blue for others
            node_colors = ['red' if node in mwis_set else 'blue' for node in G.nodes]
            
            # Create labels showing the weight of each node
            node_labels = {node: G.nodes[node].get('weight', 1) for node in G.nodes}  # Default weight is 1
            
            # Draw the graph with specific colors and labels for this algorithm's solution
            nx.draw(G, pos, ax=ax, with_labels=False, node_color=node_colors, node_size=500, edgecolors='black', font_size=10)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='white', font_weight='bold', ax=ax)
            
            # Calculate the total weight of the MWIS set
            total_weight = sum(G.nodes[node]['weight'] for node in mwis_set)
            
            # Set title for each subplot with the algorithm name and total weight
            ax.set_title(f"{algo_name} Solution\nTotal Weight: {total_weight}")
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Add an overall title for the entire figure
        fig.suptitle(f"MWIS Comparison for Graph with {len(G.nodes)} Nodes and {k*100}% Density", fontsize=16, fontweight='bold')
        
        # Adjust layout before adding the suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        
        # Save the figure as PNG if specified
        if store_png:
            fig.savefig('../images/MWIS_comparison.png', dpi=300)  # Use fig.savefig instead of plt.savefig
        else:        
            # Show the plot
            plt.show()
            
    @staticmethod
    # Define a function to calculate accuracy and precision for a given k
    def calculate_accuracy_precision(df_exhaustive, df_comparative, metric='Total Weight'):
        """
        Calculate accuracy and precision metrics for each heuristic algorithm compared to the exhaustive solution.

        Parameters:
        df_exhaustive (pd.DataFrame): DataFrame of the exhaustive solution.
        df_comparative (dict): Dictionary of DataFrames for each heuristic algorithm.
        metric (str): The metric column to compare (e.g., 'Total Weight').

        Returns:
        pd.DataFrame: DataFrame containing average, minimum, and maximum deviations for accuracy, and precision.
        """
        # Initialize a list to hold results
        results = []

        # Calculate accuracy and precision for each algorithm
        for name, df in df_comparative.items():
            # Compute deviations for accuracy
            deviations = np.abs(df[metric].values - df_exhaustive[metric].values)
            average_accuracy = np.mean(deviations)    # Mean deviation for accuracy
            max_deviation = np.max(deviations)        # Maximum deviation
            
            # Accuracy is a percantage equal to the number of zero deviations divided by the total number of deviations
            accuracy = deviations[deviations == 0].size / deviations.size
            
            # Append the calculated metrics for each algorithm
            results.append({
                'Algorithm': name,
                'Average Deviation': average_accuracy,
                'Maximum Deviation': max_deviation,
                'Accuracy': accuracy,
            })

        # Convert the list of results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    
    def plot_deviation_standard_deviation(df_exhaustive, df_comparative, metric='Total Weight', k_value=0.125):
        """
        Calculate the deviations between the exhaustive solution and heuristic algorithms,
        and plot the standard deviation of these deviations across all graph instances.

        Parameters:
        df_exhaustive (pd.DataFrame): DataFrame of the exhaustive solution.
        df_comparative (dict): Dictionary of DataFrames for each heuristic algorithm.
        metric (str): The metric column to compare (e.g., 'Total Weight').
        k_value (float): The value of k for which the plot is being generated.
        """
        # Initialize a figure for plotting
        plt.figure(figsize=(10, 6))

        # Loop through each heuristic algorithm
        for name, df in df_comparative.items():
            # Calculate deviations
            deviations = np.abs(df[metric].values - df_exhaustive[metric].values)

            # Calculate standard deviation of deviations for each graph instance
            std_deviation = np.std(deviations, axis=0)

            # Plot the standard deviation
            plt.plot(std_deviation, label=name)

        # Set plot labels and title
        plt.xlabel('Graph Instance', fontsize=12)
        plt.ylabel('Standard Deviation of Deviation', fontsize=12)
        plt.title(f'Standard Deviation of Deviations for k={k_value}', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)

        # Save or show the plot
        plt.tight_layout()
        plt.savefig(f'../images/deviation_standard_deviation_k_{k_value}.png', dpi=300)
        plt.show()
