from utils import utils
import networkx as nx
from tqdm import tqdm
import time
import math

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
            op += 1 # Count iteration
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
                op += 1 # Count iteration
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
    # Version used in the article
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
    # Used to compare precisions, not used in the article
    # It's located in this file and not in utils because of circular imports
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