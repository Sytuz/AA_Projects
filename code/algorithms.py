from concurrent.futures import ThreadPoolExecutor
from utils import utils
import networkx as nx
from tqdm import tqdm
import hashlib
import time
import math
import random
import hashlib
import csv
import os

class algorithms:
    # - G: the graph to test (networkx graph object)

    @staticmethod
    # Exhaustive search algorithm to find the maximum weight independent set
    # Checks all possible subsets of the graph's nodes
    def exhaustive_v1(G):
        op = 0
        max_weight = 0
        max_set = set()
        for subset in utils.all_subsets(G.nodes):
            op += 1
            if utils.is_independent_set(G, subset):
                weight = sum([G.nodes[node]['weight'] for node in subset])
                op += len(subset) - 1  # Count the number of additions made
                if weight > max_weight:
                    max_weight = weight
                    max_set = subset
            op += len(subset)**2  # Count the number of comparisons made
        return max_set, op
    
    @staticmethod
    # Exhaustive search algorithm to find the maximum weight independent set
    # Introduces backtracking to reduce the number of subsets to check
    def exhaustive_v2(G):
        def backtrack(current_set, remaining_nodes, current_weight, max_set_info):
            op = 0

            # If the current set's weight is higher than our known maximum, update it
            if current_weight > max_set_info[1]:
                max_set_info[0] = current_set.copy()
                max_set_info[1] = current_weight

            # Iterate over remaining nodes to try including them in the independent set
            for i, node in enumerate(remaining_nodes):
                op += 1
                if all(neighbor not in current_set for neighbor in G.neighbors(node)):
                    # Choose node, add it to the current set, and continue
                    current_set.add(node)
                    next_weight = current_weight + G.nodes[node]['weight']
                    # Recursive backtracking with remaining nodes
                    op += backtrack(current_set, remaining_nodes[i + 1:], next_weight, max_set_info)
                    # Backtrack: remove the node from the current set
                    current_set.remove(node)

            return op

        max_set_info = [set(), 0]  # Track max_set and max_weight
        op_count = backtrack(set(), list(G.nodes), 0, max_set_info)
        return max_set_info[0], op_count

    
    @staticmethod
    # Greedy algorithm to find the maximum weight independent set
    # Chooses the node with the highest weight that is not adjacent to any selected nodes
    def biggest_weight_first_v1(G):
        op = 0
        max_set = set()
        nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['weight'], reverse=True)
        for node in nodes:
            op += 1
            if utils.is_independent_set(G, max_set.union({node})):
                max_set.add(node)
        return max_set, op
    
    @staticmethod
    # Greedy algorithm to find the maximum weight independent set
    # Introduced a set to track excluded nodes, reducing the number of checks
    def biggest_weight_first_v2(G):
        op = 0
        max_set = set()
        nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['weight'], reverse=True)
        # Count the number of operations done sorting the nodes
        op += math.ceil(len(G.nodes) * math.log2(len(G.nodes)))
        excluded_nodes = set()  # Set to track nodes that are no longer eligible

        for node in nodes:
            if node in excluded_nodes:
                continue  # Skip nodes that are already adjacent to selected nodes

            op += 2 # Loop + exclusion check
            max_set.add(node)
            
            # Exclude the current node and its neighbors from future consideration
            excluded_nodes.add(node)
            excluded_nodes.update(G.neighbors(node))
            
        return max_set, op
    
    @staticmethod
    # Greedy algorithm to find the maximum weight independent set
    # Chooses the node with the lowest degree that is not adjacent to any selected nodes
    def smallest_degree_first_v1(G):
        op = 0
        max_set = set()
        nodes = sorted(G.nodes, key=lambda x: G.degree(x))
        # Count the number of operations done sorting the nodes
        op += math.ceil(len(G.nodes) * math.log2(len(G.nodes)))
        excluded_nodes = set()  # Set to track nodes that are no longer eligible

        for node in nodes:
            if node in excluded_nodes:
                continue  # Skip nodes that are already adjacent to selected nodes

            op += 2 # Loop + exclusion check
            max_set.add(node)
            
            # Exclude the current node and its neighbors from future consideration
            excluded_nodes.add(node)
            excluded_nodes.update(G.neighbors(node))
            
        return max_set, op
    
    @staticmethod
    # Greedy algorithm to find the maximum weight independent set
    # Chooses the node with the highest weight-to-degree ratio
    def weight_to_degree_v1(G):
        op = 0
        max_set = set()
        nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['weight'] / G.degree(x) if G.degree(x) != 0 else float('inf'), reverse=True)
        # Count the number of operations done sorting the nodes
        op += math.ceil(len(G.nodes) * math.log2(len(G.nodes)))
        excluded_nodes = set()  # Set to track nodes that are no longer eligible

        for node in nodes:
            if node in excluded_nodes:
                continue  # Skip nodes that are already adjacent to selected nodes

            op += 2 # Loop + exclusion check
            max_set.add(node)
            
            # Exclude the current node and its neighbors from future consideration
            excluded_nodes.add(node)
            excluded_nodes.update(G.neighbors(node))
            
        return max_set, op
    
    @staticmethod
    def monte_carlo(G, iterations=1000):
        """
        Randomized algorithm to find a near-optimal maximum weight independent set.
        Combines randomness with weighted heuristics, ensuring unique solutions based on node order.

        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """

        best_set = set()
        best_weight = 0
        ops = 0  # Initialize operation count
        
        isolated_nodes = {node for node in G.nodes if G.degree(node) == 0}
        ops += len(G.nodes)  # Increment for isolated nodes
        
        initial_weight = sum(G.nodes[node]['weight'] for node in isolated_nodes)
        ops += len(isolated_nodes)  # Increment for initial weight calculation
        
        nodes = list(set(G.nodes) - isolated_nodes)

        for _ in range(iterations):
            random.shuffle(nodes)  # Randomize node order
            ops += len(nodes)  # Increment for shuffling nodes

            # Deterministically solve for the current order
            current_set = set(isolated_nodes)
            current_weight = initial_weight
            
            excluded_nodes = set()

            for node in nodes:
                ops += 1  # Increment for checking if node is excluded
                if node in excluded_nodes:
                    continue

                current_set.add(node)
                ops += 1  # Increment for adding a node to the independent set

                current_weight += G.nodes[node]['weight']
                ops += 1  # Increment for updating current weight

                excluded_nodes.add(node)
                ops += 1  # Increment for adding a node to the excluded set

                excluded_nodes.update(G.neighbors(node))
                ops += len(list(G.neighbors(node)))  # Increment for updating neighbors

            # Update the best set if the current set is better
            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight
                ops += 2  # Increment for updating the best solution

        return best_set, ops
    
    @staticmethod
    def simulated_annealing(G, iterations=1000, cooling_rate=0.99):
        """
        Optimized Simulated Annealing algorithm to solve the MWIS problem.
        
        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.
            cooling_rate: Rate at which the temperature cools down.
        
        Returns:
            A tuple of the best independent set found and the operation count.
        """
        ops = 0  # Operation count
        initial_temp = iterations  # Initial temperature based on the number of iterations

        def is_independent(solution, node):
            """Check if adding a node keeps the solution independent."""
            return all(neighbor not in solution for neighbor in G.neighbors(node))

        def calculate_weight(solution):
            """Calculate the total weight of a solution."""
            nonlocal ops
            ops += len(solution)  # Increment for weight calculation
            return sum(G.nodes[node]['weight'] for node in solution)

        def get_random_neighbor(solution):
            """Generate a random neighbor by adding or removing a node."""
            nonlocal ops

            if random.random() < 0.5:  # Try adding a random valid node
                candidates = [node for node in G.nodes if node not in solution and is_independent(solution, node)]
                if candidates:
                    node_to_add = random.choice(candidates)
                    solution.add(node_to_add)
                    ops += 1  # Increment for adding a node
            else:  # Try removing a random node
                if solution:
                    node_to_remove = random.choice(list(solution))
                    solution.remove(node_to_remove)
                    ops += 1  # Increment for removing a node
            
            return solution

        # Initialize with a greedy independent set
        current_set, init_ops = algorithms.monte_carlo(G, 1)  # Replace this with your greedy algorithm
        ops += init_ops
        current_weight = calculate_weight(current_set)

        # Initialize best solution
        best_set = set(current_set)
        best_weight = current_weight

        # Set the initial temperature
        temp = initial_temp

        # Simulated Annealing loop
        for _ in range(iterations):
            ops += 1  # Increment for each iteration

            # Generate a random neighbor
            new_set = get_random_neighbor(set(current_set))  # Create a copy of the current set
            new_weight = calculate_weight(new_set)

            # Decide whether to accept the new solution
            if new_weight > current_weight or random.random() < math.exp((new_weight - current_weight) / temp):
                current_set = new_set
                current_weight = new_weight
                ops += 2  # Increment for accepting a new solution

            # Update the best solution if needed
            if current_weight > best_weight:
                best_set = set(current_set)
                best_weight = current_weight
                ops += 2  # Increment for updating the best solution

            # Cool down the temperature
            temp *= cooling_rate
            ops += 1  # Increment for cooling down the temperature

        return best_set, ops   
 
    @staticmethod
    def heuristic_monte_carlo_worker_thread(G, isolated_nodes, weight_degree_ratio, probabilities, iterations):
        """
        Worker function for thread-based Heuristic Monte Carlo with batched iterations.
        """
        best_set = set()
        best_weight = 0
        ops = 0
        
        initial_weight = sum(G.nodes[node]['weight'] for node in isolated_nodes)
        ops += len(isolated_nodes) # Increment for isolated nodes

        for _ in range(iterations):
            current_set = set(isolated_nodes)
            current_weight = initial_weight
            
            shuffled_nodes = random.choices(
                [node for node, _ in weight_degree_ratio],
                weights=probabilities,
                k=len(weight_degree_ratio)
            )
            ops += len(shuffled_nodes) # Increment for shuffling nodes

            excluded_nodes = set()
            for node in shuffled_nodes:
                ops += 1 # Increment for checking if node is excluded
                if node in excluded_nodes:
                    continue

                current_set.add(node)
                ops += 1 # Increment for adding node to current set

                current_weight += G.nodes[node]['weight']
                ops += 1 # Increment for adding node weight to current weight

                excluded_nodes.add(node)
                ops += 1 # Increment for adding node to excluded nodes
                
                excluded_nodes.update(G.neighbors(node))
                ops += len(list(G.neighbors(node))) # Increment for adding neighbors to excluded nodes

            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight
                ops += 2 # Increment for updating best set and best weight

        return best_set, best_weight, ops

    @staticmethod
    def parallel_heuristic_monte_carlo(G, iterations=1000):
        """
        Thread-based parallel version of Heuristic Monte Carlo to find a near-optimal MWIS.
        
        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """
        isolated_nodes = {node for node in G.nodes if G.degree(node) == 0}
        weight_degree_ratio = [
            (node, G.nodes[node]['weight'] / G.degree(node))
            for node in G.nodes if G.degree(node) > 0
        ]
        
        if not weight_degree_ratio:
            return isolated_nodes, 1

        total_ratio = sum(ratio for _, ratio in weight_degree_ratio)
        probabilities = [ratio / total_ratio for _, ratio in weight_degree_ratio]

        # Determine thread pool size and iterations per thread
        max_threads = min(8, iterations)  # Limit to 8 threads or fewer
        iterations_per_thread = max(1, iterations // max_threads)

        # Prepare worker arguments
        worker_args = [
            (G, isolated_nodes, weight_degree_ratio, probabilities, iterations_per_thread)
            for _ in range(max_threads)
        ]

        best_set = set()
        best_weight = 0
        total_ops = 0

        # Run workers in a thread pool
        with ThreadPoolExecutor(max_threads) as executor:
            results = executor.map(
                lambda args: algorithms.heuristic_monte_carlo_worker_thread(*args),
                worker_args
            )

        # Combine results
        for current_set, current_weight, ops in results:
            total_ops += ops
            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight

        return best_set, total_ops
    
    @staticmethod
    def compare_precision(func, p, n, print_results=False, iterations=1, func_iterations=1000, initial_temp=100, cooling_rate=0.99):
        """
        Compares the precision of a given function against the exhaustive (v2) algorithm.
        If the function is randomized, it will run multiple iterations to calculate the average precision.

        Args:
            func: The algorithm to test.
            p: Probability for edge creation in graph generation.
            n: Maximum number of nodes for testing graphs.
            print_results: Whether to print detailed results for each graph.
            iterations: Number of runs per graph for randomized algorithms.
            func_iterations: Iterations for the algorithm being tested.
            initial_temp: Initial temperature for simulated annealing (if applicable).
            cooling_rate: Cooling rate for simulated annealing (if applicable).

        Returns:
            The precision percentage.
        """
        matches = 0
        total_tests = 0
        start_time = time.time()  # Start timing the process

        for i in tqdm(range(1, n + 1), desc="Testing graphs", unit="graph"):
            # Generate graph with i nodes and edge probability p
            G = utils.create_graph_v4(i, p)

            # Run the exhaustive (v2) algorithm to get the ground truth
            result_exhaustive, _ = algorithms.exhaustive_v2(G)

            for _ in range(iterations):
                total_tests += 1

                # Run the function to test
                if func == algorithms.simulated_annealing:
                    result_func, _ = func(G, func_iterations, initial_temp, cooling_rate)
                elif func == algorithms.monte_carlo or func == algorithms.heuristic_monte_carlo or func == algorithms.threaded_heuristic_monte_carlo:
                    result_func, _ = func(G, func_iterations)
                else:
                    result_func, _ = func(G)

                # Optionally print results
                if print_results:
                    print(f"Graph with {i} nodes (Iteration {_+1}/{iterations}):")
                    print("Exhaustive algorithm:", result_exhaustive)
                    print("Function result:", result_func)
                    print()

                # Check if the results match
                if result_exhaustive == result_func:
                    matches += 1

        # Calculate the precision as a percentage
        precision = (matches / total_tests) * 100
        elapsed_time = time.time() - start_time  # Calculate total time taken

        print(f"Total time taken: {elapsed_time:.2f} seconds")
        return precision
    
    @staticmethod
    def tune_parameters(func, p, n, trials=10, func_iterations=1000):
        """
        Tune the initial temperature and cooling rate for the given function.

        Args:
            func: The algorithm to test (e.g., simulated_annealing).
            p: Probability for edge creation in graph generation.
            n: Maximum number of nodes for testing graphs.
            trials: Number of graphs to test for each configuration.
            func_iterations: Number of iterations for the algorithm.

        Returns:
            A dictionary with the best configuration and its precision.
        """
        best_precision = 0
        best_params = None

        # Parameter ranges for tuning
        initial_temps = [10, 50, 100, 500, 1000]
        cooling_rates = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]

        # Create the directory if it doesn't exist
        results_dir = "../data/sa_tests"
        os.makedirs(results_dir, exist_ok=True)

        # Prepare CSV file
        file_path = os.path.join(results_dir, f"sa_results_n{n}_k{p}.csv")

        # Initialize CSV file with headers
        if not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['initial_temp', 'cooling_rate', 'precision'])

        # Run parameter tuning
        for initial_temp in initial_temps:
            for cooling_rate in cooling_rates:
                print(f"Testing configuration: initial_temp={initial_temp}, cooling_rate={cooling_rate}")
                
                # Run precision tests for the current configuration
                precision = algorithms.compare_precision(
                    func=func,
                    p=p,
                    n=n,
                    print_results=False,
                    iterations=trials,
                    func_iterations=func_iterations,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate
                )

                print(f"Precision: {precision:.2f}% for initial_temp={initial_temp}, cooling_rate={cooling_rate}")

                # Update the best configuration if needed
                if precision > best_precision:
                    best_precision = precision
                    best_params = {'initial_temp': initial_temp, 'cooling_rate': cooling_rate}

                # Save the current configuration and precision to CSV
                with open(file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([initial_temp, cooling_rate, precision])

        return {'best_params': best_params, 'best_precision': best_precision}