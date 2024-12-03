from utils import utils
from tqdm import tqdm
import networkx as nx
import numpy as np
import threading
import json
import time
import csv
import os

class stressTester:

    @staticmethod
    def run_stress_test_iteration(func, G, max_time_seconds, iterations, writer, graph_name=None):
        """Runs a single iteration of the stress test and writes the result."""
        # Use preloaded graph or generate a new one

        result_container = {}
        elapsed_time = 0

        # Run the test in a thread to enforce timeout
        def run_func():
            if iterations is not None:
                result_container['result'], result_container['operations'] = func(G, iterations=iterations)
            else:
                result_container['result'], result_container['operations'] = func(G)

        thread = threading.Thread(target=run_func)
        start = time.time()
        thread.start()
        thread.join(timeout=max_time_seconds)
        end = time.time()

        if thread.is_alive():
            print(f"Execution for n = {G.number_of_nodes()} exceeded the time limit of {max_time_seconds / 60:.2f} minutes. Stopping stress test.")
            thread.join()  # Ensure thread is cleaned up
            return

        elapsed_time = end - start
        result = result_container.get('result', set())
        operations = result_container.get('operations', 0)
        total_weight = sum(G.nodes[node]['weight'] for node in result)
        solution_size = len(result)
        result_str = str(result)

        # Save the results
        row = [G.number_of_nodes(), operations, total_weight, elapsed_time, solution_size, result_str]
        if graph_name:
            row.insert(0, graph_name)
        writer.writerow(row)

    @staticmethod
    def stress_test(func, max_time_minutes=1, dataset_file="../graphs/graph_dataset.jsonl", base_filename="stress_test_results", iterations=None, generate=None):
        """
        Runs stress tests for a function across multiple graphs in a dataset.

        Args:
            func: Function to test.
            max_time_minutes: Maximum time allowed for a single test.
            dataset_file: File containing the dataset of graphs.
            base_filename: Base directory name for saving results
            iterations: Number of iterations for algorithms that require them (optional).
            generate: Dictionary with parameters for generating graphs (optional).
            |   n_max: Maximum number of nodes in generated graphs.
            |   step: Increment size for nodes in each test.
            |   k: Edge probability for generated graphs.
        """
        # Base directory for all results
        base_folder = f"../data/{base_filename}"

        # Max time in seconds
        max_time_seconds = max_time_minutes * 60

        # Create the base folder if it doesn't exist
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        if generate:
            # Generate graphs for a full test
            n_max = generate.get('n_max', 1000)
            k = generate.get('k', 0.5)
            step = generate.get('sample_size', 5)

            graph_dataset = utils.graph_creation(n_max, k, step)

        # Prepare to write results to a single CSV file
        filename = os.path.join(base_folder, "results.csv" if not generate else f"results_{k}.csv")
        print(f"Writing results to {filename}")
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            header = ["Node Count", "Number of Operations", "Total Weight", "Execution Time (seconds)", "Solution Size", "Solution"]
            if not generate:
                header.insert(0, "Graph Name")

            # Write CSV header
            writer.writerow(header)
            
        if generate:
            with open(filename, mode='a', newline='') as result_file:
                writer = csv.writer(result_file)

                # Wrapping the file iterator with tqdm for better progress tracking
                for _, G in tqdm(graph_dataset.items(), desc="Stress Testing Generated Graphs"):
                        # Run the stress test on the generated graph
                        stressTester.run_stress_test_iteration(func, G, max_time_seconds, iterations, writer)
        else:
            # Process each line in the JSONL dataset
            with open(dataset_file, 'r', encoding='utf-8') as jsonl_file, open(filename, mode='a', newline='') as result_file:
                writer = csv.writer(result_file)

                # Wrapping the file iterator with tqdm for better progress tracking
                for line in tqdm(jsonl_file, desc="Stress Testing Graphs", unit='graph', total=sum(1 for _ in open(dataset_file))):
                    try:
                        # Load the graph data from the current JSONL line
                        graph_data = json.loads(line)
                        graph_name = graph_data.get('graph_name', None)
                        adjacency_matrix = np.array(graph_data['adjacency_matrix'])
                        node_weights = graph_data['node_weights']

                        # Reconstruct the graph using networkx
                        G = nx.Graph(adjacency_matrix)

                        for i, weight in enumerate(node_weights):
                            G.nodes[i]['weight'] = weight

                        # Run the stress test on the graph
                        stressTester.run_stress_test_iteration(func, G, max_time_seconds, iterations, writer, graph_name)

                    except Exception as e:
                        print(f"Failed to process graph: {e}")
                

    @staticmethod
    def _execute_function(func, G, max_time_seconds, iterations=None):
        """
        Executes the given function with a timeout and captures results.

        Args:
            func: The function to execute.
            G: The graph input for the function.
            max_time_seconds: Maximum allowed execution time.
            iterations: Number of iterations for algorithms that require them (optional).

        Returns:
            Tuple: (result, elapsed_time, operations).
        """
        result_container = {}
        start_time = time.time()

        def run_func():
            if iterations is not None:
                result_container['result'], result_container['op'] = func(G, iterations=iterations)
            else:
                result_container['result'], result_container['op'] = func(G)

        thread = threading.Thread(target=run_func)
        thread.start()
        thread.join(timeout=max_time_seconds)

        elapsed_time = time.time() - start_time
        if thread.is_alive():
            thread.join()  # Ensure thread cleanup
            return None, None, None  # Indicate timeout

        result = result_container.get('result', set())
        operations = result_container.get('op', 0)
        return result, elapsed_time, operations