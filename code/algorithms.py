from utils import utils
import networkx as nx
from tqdm import tqdm
import time
import math
import random
import hashlib
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
        Combines randomness with weighted heuristics.
        
        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """

        best_set = set()
        best_weight = 0
        ops = 0  # Initialize operation count

        for _ in range(iterations):
            current_set = set()
            current_weight = 0

            nodes = list(G.nodes)
            random.shuffle(nodes)  # Randomize node order

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
                ops += 1  # Increment for updating the best solution

        return best_set, ops

    @staticmethod
    def monte_carlo_with_filter(G, iterations=1000, hash_size=None):
        """
        Monte Carlo algorithm with a mechanism to avoid repeating the same node orders.

        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.
            hash_size: Maximum size of the Bloom-like hash storage.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """
        best_set = set()
        best_weight = 0
        
        ops = 0  # Initialize operation count
        
        if hash_size is None:
            hash_size = iterations

        # Initialize a set to store hashes of processed node orders
        processed_hashes = set()

        for _ in range(iterations):
            # Generate a random order of nodes
            nodes = list(G.nodes)
            random.shuffle(nodes)

            # Create a hash of the sorted node order
            nodes_hash = hashlib.md5(str(nodes).encode()).hexdigest()

            # Check if this order has already been processed
            ops += 1  # Increment for checking if the node order has been processed
            if nodes_hash in processed_hashes:
                continue  # Skip this iteration to avoid duplication

            # Add the hash to the processed set
            processed_hashes.add(nodes_hash)
            if len(processed_hashes) > hash_size:  # Optionally cap the size
                processed_hashes.pop()  # Remove an arbitrary element to maintain size
            
            # Initialize current independent set and weight
            current_set = set()
            current_weight = 0
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
                ops += 1  # Increment for updating the best solution

        return best_set, ops
    
    def heuristic_monte_carlo(G, iterations=1000):
        """
        Randomized algorithm to find a near-optimal maximum weight independent set.
        Combines randomness with the weighted-to-degree heuristic.

        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """
        best_set = set()
        best_weight = 0
        ops = 0

        nodes = list(G.nodes)
        # Separate isolated nodes
        isolated_nodes = {node for node in nodes if G.degree(node) == 0}
        weight_degree_ratio = [
            (node, G.nodes[node]['weight'] / G.degree(node))
            for node in nodes if G.degree(node) > 0
        ]
        
        if not weight_degree_ratio:  # Handle edge case where all nodes are isolated
            return isolated_nodes, 1

        total_ratio = sum(ratio for _, ratio in weight_degree_ratio)
        if total_ratio <= 0 or not math.isfinite(total_ratio):
            raise ValueError("Total ratio must be positive and finite")

        probabilities = [ratio / total_ratio for _, ratio in weight_degree_ratio]

        for _ in range(iterations):
            current_set = set(isolated_nodes)  # Start with isolated nodes
            current_weight = sum(G.nodes[node]['weight'] for node in isolated_nodes)
            
            # Shuffle nodes based on the calculated probabilities
            shuffled_nodes = random.choices([node for node, _ in weight_degree_ratio], weights=probabilities, k=len(weight_degree_ratio))
            excluded_nodes = set()

            for node in shuffled_nodes:
                ops += 1
                if node in excluded_nodes:
                    continue

                current_set.add(node)
                current_weight += G.nodes[node]['weight']
                excluded_nodes.add(node)
                excluded_nodes.update(G.neighbors(node))
                ops += len(list(G.neighbors(node)))

            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight
                ops += 1

        return best_set, ops
    
    @staticmethod
    # Greedy algorithm with random selection factor
    # Chooses a node based on a weighted heuristic with a random factor
    def weight_to_degree_with_randomness(G, coin_prob=0.5):
        op = 0
        max_set = set()
        nodes = list(G.nodes)
        
        # Count the number of operations for sorting the nodes (if it's necessary)
        op += math.ceil(len(nodes) * math.log2(len(nodes))) if len(nodes) > 1 else 0

        excluded_nodes = set()  # Set to track nodes that are no longer eligible

        while nodes:
            # Create the list of eligible nodes
            eligible_nodes = [n for n in nodes if n not in excluded_nodes]
            if not eligible_nodes:
                break  # Exit the loop if there are no eligible nodes left

            # Flip the coin with the given probability
            if random.random() < coin_prob:
                # Select a node randomly from the eligible nodes
                node = random.choice(eligible_nodes)
                op += 1  # Operation for selecting a random node
            else:
                # Heuristic-based selection: node with the highest weight-to-degree ratio
                node = max(
                    eligible_nodes,
                    key=lambda x: G.nodes[x]['weight'] / G.degree(x) if G.degree(x) != 0 else float('inf')
                )
                op += 2  # Operation for selecting using the heuristic
            
            # Add the selected node to the result set
            max_set.add(node)
            
            # Exclude the current node and its neighbors from future consideration
            excluded_nodes.add(node)
            excluded_nodes.update(G.neighbors(node))
            
            # Remove the selected node from the list of available nodes
            nodes.remove(node)

        return max_set, op

    @staticmethod
    def compare_precision(func, p, n, print_results=False, iterations=1, func_iterations=1000, restart_prob=0.1):
        """Compares the precision of a given function against the exhaustive (v2) algorithm."""
        """If the function is randomized, it will run multiple iterations to calculate the average precision."""
        
        matches = 0
        start_time = time.time()  # Start timing the process
        
        if iterations > 1:
            # Run multiple iterations of the randomized algorithm
            for _ in tqdm(range(iterations), desc="Testing graphs", unit="iteration"):
                for i in range(1, n + 1):
                    # Generate graph with i nodes and edge probability p
                    G = utils.create_graph_v4(i, p)
                    
                    # Run the exhaustive (v2) algorithm
                    result_exhaustive, _ = algorithms.exhaustive_v2(G)
                    
                    # Run the function to test
                    result_func, _ = func(G, func_iterations, restart_prob)
                    
                    if print_results:
                        print(f"Graph with {i} nodes:")
                        print("Exhaustive algorithm:", result_exhaustive)
                        print("Function result:", result_func)
                        print()
                    
                    # Check if the results match
                    if result_exhaustive == result_func:
                        matches += 1
        else:
            for i in tqdm(range(1, n + 1), desc="Testing graphs", unit="graph"):
                # Generate graph with i nodes and edge probability p
                G = utils.create_graph_v4(i, p)
                
                # Run the exhaustive (v2) algorithm
                result_exhaustive, _ = algorithms.exhaustive_v2(G)
                
                # Run the function to test
                result_func, _ = func(G)

                if print_results:
                    print(f"Graph with {i} nodes:")
                    print("Exhaustive algorithm:", result_exhaustive)
                    print("Function result:", result_func)
                    print()
                
                # Check if the results match
                if result_exhaustive == result_func:
                    matches += 1
        
        # Calculate the precision as a percentage
        precision = (matches / (n * iterations)) * 100
        elapsed_time = time.time() - start_time  # Calculate total time taken
        
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        return precision