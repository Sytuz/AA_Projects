import signal
from utils import utils
from tqdm import tqdm
import networkx as nx
import numpy as np
import json
import time
import csv
import os


class TimeoutException(Exception):
    """Custom exception to handle timeouts."""
    pass


def timeout_handler(signum, frame):
    """Handler for the timeout signal."""
    raise TimeoutException("Execution exceeded the allowed time limit.")


class stressTester:

    @staticmethod
    def run_stress_test_iteration(func, G, max_time_seconds, iterations, writer, graph_name=None):
        """Runs a single iteration of the stress test and writes the result."""
        result = set()
        operations = 0
        elapsed_time = 0

        try:
            # Set up the signal for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_time_seconds)  # Set the timeout in seconds

            start = time.time()
            if iterations is not None:
                result, operations = func(G, iterations=iterations)
            else:
                result, operations = func(G)
            elapsed_time = time.time() - start

            # Disable the alarm after successful execution
            signal.alarm(0)
        except TimeoutException:
            print(f"Execution for n = {G.number_of_nodes()} exceeded the time limit of {max_time_seconds / 60:.2f} minutes. Stopping stress test.")
            return False  # Indicate timeout
        except Exception as e:
            print(f"Error during function execution: {e}")
            raise e
            return False  # Treat errors as timeouts for safety

        total_weight = sum(G.nodes[node]['weight'] for node in result)
        solution_size = len(result)
        result_str = str(result)

        # Save the results
        row = [G.number_of_nodes(), operations, total_weight, elapsed_time, solution_size, result_str]
        if graph_name:
            row.insert(0, graph_name)
        writer.writerow(row)

        return True  # Indicate success

    @staticmethod
    def stress_test(func, max_time_minutes=1, dataset_file="../graphs/graph_dataset.jsonl", dirname="stress_test_results", filename="results", iterations=None, generate=None):
        """
        Runs stress tests for a function across multiple graphs in a dataset.

        Args:
            func: Function to test.
            max_time_minutes: Maximum time allowed for a single test.
            dataset_file: File containing the dataset of graphs.
            dirname: Base directory name for saving results
            iterations: Number of iterations for algorithms that require them (optional).
            generate: Dictionary with parameters for generating graphs (optional).
            |   n_max: Maximum number of nodes in generated graphs.
            |   step: Increment size for nodes in each test.
            |   k: Edge probability for generated graphs.
        """
        # Base directory for all results
        base_folder = f"../data/{dirname}"

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
        filename = os.path.join(base_folder, f"{filename}.csv" if not generate else f"{filename}_{k}.csv")
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

                for _, G in tqdm(graph_dataset.items(), desc="Stress Testing Generated Graphs"):
                    # Run the stress test on the generated graph
                    success = stressTester.run_stress_test_iteration(func, G, max_time_seconds, iterations, writer)
                    if not success:  # Stop further testing on timeout
                        print("Timeout reached, stopping further testing.")
                        break
        else:
            # Process each line in the JSONL dataset
            with open(dataset_file, 'r', encoding='utf-8') as jsonl_file, open(filename, mode='a', newline='') as result_file:
                writer = csv.writer(result_file)

                for line in tqdm(jsonl_file, desc="Stress Testing Graphs", unit='graph', total=sum(1 for _ in open(dataset_file))):
                    try:
                        # Load the graph data from the current JSONL line
                        graph_data = json.loads(line)
                        graph_name = graph_data.get('graph_name', None)
                        adjacency_list = graph_data['adjacency_list']
                        node_weights = graph_data['node_weights']

                        # Convert keys of adjacency list to integers
                        adjacency_list = {int(k): v for k, v in adjacency_list.items()}

                        # Reconstruct the graph using networkx
                        G = nx.Graph(adjacency_list)

                        for i, weight in enumerate(node_weights):
                            G.nodes[i]['weight'] = weight

                        # Run the stress test on the graph
                        success = stressTester.run_stress_test_iteration(func, G, max_time_seconds, iterations, writer, graph_name)

                        if not success:  # Stop further testing on timeout
                            print("Timeout reached, stopping further testing.")
                            break

                    except Exception as e:
                        print(f"Failed to process graph: {e}")
                        raise e
