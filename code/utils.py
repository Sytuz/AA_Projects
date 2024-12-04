from concurrent.futures import ThreadPoolExecutor, TimeoutError
import matplotlib.pyplot as plt
from constants import k_values
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import threading
import random
import json
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
    def graph_creation(n, k, step, save_data=None, num_threads=4):
        """
        Function to create graphs with parallel processing, a timeout for each graph, 
        and save to a JSONL file. Writing occurs only after all graphs are generated.

        Args:
            n (int): Maximum size of graphs.
            k (float): Edge probability.
            step (int): Step size for graph increments.
            save_data (Dict): Dictionary with output directory and file name.
            num_threads (int): Number of threads to use for graph generation.
            timeout (int): Timeout in seconds for each graph generation task.

        Returns:
            str or Dict: The path to the JSONL file if `save_data` is provided; otherwise, a dictionary of graphs.
        """
        def generate_graph(size):
            """Generates a graph and returns it as a JSON object."""
            graph = utils.create_graph_v4(size, k)
            adjacency_list = {node: list(graph.neighbors(node)) for node in graph.nodes}
            node_weights = [graph.nodes[node]['weight'] for node in graph.nodes]
            return {
                "graph_name": f"graph_size_{size}_{k}",
                "size": size,
                "adjacency_list": adjacency_list,
                "node_weights": node_weights
            }

        # Define sizes
        sizes = list(range(step, n + 1, step))

        
        # Progress bar setup
        print("Starting graph generation...")
        graph_data = []

        with tqdm(total=len(sizes), desc=f"Generating Graphs for k={k}", unit="graph") as pbar:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit tasks for all sizes, respecting the timeout flag
                futures = {}
                for size in sizes:
                    future = executor.submit(generate_graph, size)
                    futures[future] = size

                # Process completed futures
                for future in futures:
                    try:
                        graph_json = future.result()
                        graph_data.append(graph_json)
                    except Exception as e:
                        print(f"Graph generation failed for size {futures[future]}: {e}")
                    finally:
                        pbar.update(1)

        if save_data:
            # Ensure the output directory exists
            os.makedirs(save_data['output_dir'], exist_ok=True)
            output_filename = os.path.join(save_data['output_dir'], f"{save_data['file_name']}_{k}.jsonl")
            
            # Write all graphs to JSONL file
            with open(output_filename, 'w') as file:
                for graph_json in graph_data:
                    file.write(json.dumps(graph_json) + '\n')
            
            print("All graphs generated and saved.")
            return output_filename

        print("All graphs generated.")
        return graph_data
    
    @staticmethod
    def create_graph_from_txt_to_jsonl(file_path='../graphs/test_graph.txt', graph_name="test_graph"):
        """
        Reads a graph from a text file and returns a JSONL line representation
        with adjacency list, node weights, and graph name.

        Args:
            file_path (str): Path to the input text file.
            graph_name (str): Name of the graph.

        Returns:
            str: A JSON-formatted string representing the graph in JSONL format.
        """
        with open(file_path, 'r') as f:
            # Read the file and split into lines
            lines = f.readlines()
            
            # Parse number of nodes and edges
            num_nodes = int(lines[0].strip())
            num_edges = int(lines[1].strip())
            
            # Parse the adjacency list
            adjacency_list = {i: [] for i in range(num_nodes)}
            for line in lines[2:]:
                u, v = map(int, line.strip().split())
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
            
            # Generate random weights for each node
            rand_instance = random.Random(108122)
            weights = [rand_instance.randint(1, 100) for _ in range(num_nodes)]
            
            # Construct the JSON object
            graph_data = {
                "graph_name": graph_name,
                "adjacency_list": adjacency_list,
                "node_weights": weights
            }
            
            # Return as a JSONL-compatible string
            return json.dumps(graph_data)

    @staticmethod
    def load_graphs(filename):
        """
        Load graphs from a JSONL file.

        Args:
            filename (str): Path to the JSONL file.

        Returns:
            List of graph data (as dictionaries).
        """
        graph_data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                graph_data.append(json.loads(line.strip()))
        return graph_data

    @staticmethod
    # Function to generate all subsets of a given set
    def all_subsets(s):
        return itertools.chain(*map(lambda x: itertools.combinations(s, x), range(0, len(s)+1)))

    @staticmethod
    # Function to check if a given subset is an independent set in a graph
    def is_independent_set(G, subset):
        for u, v in itertools.combinations(subset, 2):
            if G.has_edge(u, v):
                return False;
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

    @staticmethod
    def graphs_txt_to_jsonl(input_dir='../graphs/graph_dataset', output_filename='../graphs/graph_dataset.jsonl'):
        """
        Convert a dataset of graphs stored in text files to a single JSON Lines file.

        Parameters:
        - input_dir (str): Directory containing text files with graph data.
        - output_filename (str): Output JSON Lines file to store the graph dataset.
        """
        # Open the output file in write mode
        with open(output_filename, 'w') as f:
            # Loop through each file in the input directory
            for file in os.listdir(input_dir):
                if file.endswith('.txt'):  # Ensure only text files are processed
                    graph_name = os.path.splitext(file)[0]  # Use the file name without extension as graph name
                    file_path = os.path.join(input_dir, file)
                    
                    try:
                        # Convert the graph to a JSONL line
                        graph_jsonl = utils.create_graph_jsonl(file_path, graph_name)
                        
                        # Write the JSONL line to the output file
                        f.write(graph_jsonl + '\n')
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")

        print(f"Graph dataset saved to {output_filename}")
        
    @staticmethod
    def results_average(foldername, k_vals=k_values, metric='Total Weight'):
        """
        Calculate the average metric value for each graph size in the results DataFrame.

        Parameters:
        - foldername (str): Name of the folder containing the results CSV files.
        - k_vals (list): List of k values for which to calculate the average metric.
        - metric (str): The metric column to calculate the average.

        Returns:
        - pd.Series: Series containing the average metric value for each graph size.
        """
        
        # Initialize an empty dictionary to store the average metric values
        average_values = {}
        
        # Loop through each k value
        for k in k_vals:
            # Load the results DataFrame for the current k value
            results_df = pd.read_csv(f'../data/{foldername}/results_{k}.csv')
            
            # Calculate the average metric value for each graph size
            average_values[k] = results_df.groupby('Node Count')[metric].mean().values
        
        # Convert the dictionary to a pandas Series
        return pd.Series(average_values)
