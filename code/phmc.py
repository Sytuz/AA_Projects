import random
import networkx as nx
from multiprocessing import Pool

class PHMC:
    def __init__(self, G):
        """
        Initialize the class with the graph and precompute shared data.
        """
        self.G = G
        self.isolated_nodes = {node for node in G.nodes if G.degree(node) == 0}
        self.weight_degree_ratio = [
            (node, G.nodes[node]['weight'] / G.degree(node))
            for node in G.nodes if G.degree(node) > 0
        ]

        # Precompute probabilities for weighted random selection
        if self.weight_degree_ratio:
            total_ratio = sum(ratio for _, ratio in self.weight_degree_ratio)
            self.probabilities = [ratio / total_ratio for _, ratio in self.weight_degree_ratio]
        else:
            self.probabilities = []

    def heuristic_monte_carlo_worker_process(self, data):
        """
        Worker function to minimize pickling overhead.
        Args:
            data: A tuple containing (isolated_nodes, weight_degree_ratio, probabilities, iterations).
        """
        iterations = data

        best_set = set()
        best_weight = 0
        ops = 0

        for _ in range(iterations):
            current_set = set(self.isolated_nodes)
            current_weight = sum(self.G.nodes[node]['weight'] for node in self.isolated_nodes)
            ops += len(self.isolated_nodes) # Increment for isolated nodes
            
            if not self.weight_degree_ratio:
                continue

            shuffled_nodes = random.choices(
                [node for node, _ in self.weight_degree_ratio],
                weights=self.probabilities,
                k=len(self.weight_degree_ratio)
            )
            ops += len(shuffled_nodes) # Increment for shuffling nodes

            excluded_nodes = set()
            for node in shuffled_nodes:
                ops += 1 # Increment for checking if node is excluded
                if node in excluded_nodes:
                    continue

                current_set.add(node)
                ops += 1 # Increment for adding node to current set
                
                current_weight += self.G.nodes[node]['weight']
                ops += 1 # Increment for adding node weight to current weight
                
                excluded_nodes.add(node)
                ops += 1 # Increment for adding node to excluded nodes
                
                excluded_nodes.update(self.G.neighbors(node))
                ops += len(list(self.G.neighbors(node))) # Increment for adding neighbors to excluded nodes

            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight
                ops += 2 # Increment for updating best set and best weight

        return best_set, best_weight, ops

    def parallel_heuristic_monte_carlo(self, iterations=1000):
        """
        Process-based parallel version of Heuristic Monte Carlo to find a near-optimal MWIS.

        Args:
            iterations: Number of iterations for refinement.

        Returns:
            A tuple of the maximum independent set found and the operation count.
        """
        if not self.weight_degree_ratio:
            # If no nodes with positive degree exist, return isolated nodes
            return self.isolated_nodes, 1

        # Determine process pool size and iterations per process
        max_processes = min(8, iterations)  # Limit to 8 processes or fewer
        iterations_per_process = max(1, iterations // max_processes)

        # Prepare data for workers
        data = (
            iterations_per_process
        )

        # Run workers in a process pool
        with Pool(max_processes) as pool:
            results = pool.map(self.heuristic_monte_carlo_worker_process, [data] * max_processes)

        # Combine results
        best_set = set()
        best_weight = 0
        total_ops = 0
        for current_set, current_weight, ops in results:
            total_ops += ops
            if current_weight > best_weight:
                best_set = current_set
                best_weight = current_weight
                ops += 2 # Increment for updating best set and best weight

        return best_set, total_ops

    def run(self, iterations=1000):
        """
        A public method to run the parallel Heuristic Monte Carlo algorithm.
        """
        return self.parallel_heuristic_monte_carlo(iterations)
