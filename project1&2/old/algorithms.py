from article_mwis import MWIS
from utils import utils
import networkx as nx
from tqdm import tqdm
import time
import math
import random

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
    # Improved greedy algorithm to find the maximum weight independent set
    # From: https://drops.dagstuhl.de/storage/00lipics/lipics-vol244-esa2022/LIPIcs.ESA.2022.45/LIPIcs.ESA.2022.45.pdf
    def article_greedy(G):
        # Get the adjacency list of the graph
        graph = {node: list(neighbors) for node, neighbors in G.adj.items()}

        # Get the weight of each node
        weights = nx.get_node_attributes(G, 'weight')

        # Maximum time allowed for the algorithm
        max_time = 1

        # Create an instance of the MWIS class
        mwis = MWIS(graph, weights, max_time)

        # Run the algorithm
        mwis.run()

        # Return the best solution found
        return mwis.best_solution, mwis.best_weight
    
    @staticmethod
    # Completely random algorithm to find the maximum weight independent set
    # Randomly selects nodes until no more nodes can be added
    def completely_random(G):
        op = 0
        max_set = set()
        nodes = list(G.nodes)
        while nodes:
            node = random.choice(nodes)
            op += 1
        
            if utils.is_independent_set(G, max_set.union({node})):
                max_set.add(node)
            nodes.remove(node)
        return max_set, op
    
    @staticmethod
    def randomized_maximum_weight_independent_set(G, iterations=1000, restart_prob=0.1):
        """
        Randomized algorithm to find a near-optimal maximum weight independent set.
        Combines randomness with weighted heuristics.
        
        Args:
            G: A graph with weights on nodes.
            iterations: Number of iterations for refinement.
            restart_prob: Probability of restarting to introduce randomness (0-1).

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """
        op = 0
        best_set = set()
        best_weight = 0

        for _ in range(iterations):
            op += 1  # Increment for each iteration
            current_set = set()
            current_weight = 0

            nodes = list(G.nodes)
            random.shuffle(nodes)  # Randomize node order

            excluded_nodes = set()

            for node in nodes:
                op += 1  # Increment for loop operations
                if node in excluded_nodes:
                    continue

                # Add the node probabilistically or based on its weight
                if random.random() > restart_prob:  # Favor deterministic selection
                    current_set.add(node)
                    current_weight += G.nodes[node]['weight']
                    excluded_nodes.add(node)
                    excluded_nodes.update(G.neighbors(node))
                else:  # Add some randomness
                    random_node = random.choice([n for n in nodes if n not in excluded_nodes])
                    current_set.add(random_node)
                    current_weight += G.nodes[random_node]['weight']
                    excluded_nodes.add(random_node)
                    excluded_nodes.update(G.neighbors(random_node))

            # Update the best set if the current set is better
            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight

        return best_set, op

    
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
    def compare_precision(func, p, n, print_results=False):
        """Compares the precision of a given function against the exhaustive (v2) algorithm."""
        
        matches = 0
        start_time = time.time()  # Start timing the process

        # Use tqdm to create a progress bar
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
        precision = (matches / n) * 100
        elapsed_time = time.time() - start_time  # Calculate total time taken
        
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        return precision