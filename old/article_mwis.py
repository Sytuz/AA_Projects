import random

class MWIS:
    def __init__(self, graph, weights, max_time):
        self.graph = graph  # adjacency list of the graph
        self.weights = weights  # weight of each node
        self.max_time = max_time
        self.best_solution = set()
        self.best_weight = 0
        self.elite_solutions = []

    def greedy_random_solution(self):
        # Create an initial solution using a randomized greedy approach
        nodes = sorted(self.graph.keys(), key=lambda n: self.weights[n], reverse=True)  # sort by weight
        solution = set()
        for node in nodes:
            # Add node if it doesn't conflict with the current solution
            if all(neigh not in solution for neigh in self.graph[node]):
                solution.add(node)
        return solution

    def calculate_weight(self, solution):
        return sum(self.weights[n] for n in solution)

    def local_search(self, solution):
        # Apply local search moves to improve the solution while ensuring independence
        improved = True
        while improved:
            improved = False
            for node in list(solution):
                # Try to remove 'node' and replace it with a better choice among its neighbors
                new_solution = solution - {node}
                candidate = None
                max_gain = 0
                for neighbor in self.graph[node]:
                    if neighbor not in new_solution and all(n not in new_solution for n in self.graph[neighbor]):
                        gain = self.weights[neighbor] - self.weights[node]
                        if gain > max_gain:
                            max_gain = gain
                            candidate = neighbor
                if candidate:
                    new_solution.add(candidate)
                    solution = new_solution
                    improved = True
                    break
        return solution

    def path_relinking(self, current_solution, target_solution):
        # Combine current solution with an elite solution while preserving independence
        combined_solution = current_solution.copy()
        for node in target_solution:
            if node not in combined_solution and all(neigh not in combined_solution for neigh in self.graph[node]):
                combined_solution.add(node)
        return self.local_search(combined_solution)

    def run(self):
        start_solution = self.greedy_random_solution()
        start_solution = self.local_search(start_solution)
        self.elite_solutions.append(start_solution)
        self.best_solution = start_solution
        self.best_weight = self.calculate_weight(start_solution)

        for _ in range(self.max_time):
            current_solution = self.greedy_random_solution()
            if random.random() < 0.5:
                current_solution = self.local_search(current_solution)

            elite_solution = random.choice(self.elite_solutions)
            new_solution = self.path_relinking(current_solution, elite_solution)
            new_weight = self.calculate_weight(new_solution)

            # Only add new solution if it is an independent set
            if new_weight > self.best_weight and all(all(neigh not in new_solution for neigh in self.graph[node]) for node in new_solution):
                self.best_solution = new_solution
                self.best_weight = new_weight
                self.elite_solutions.append(new_solution)

        return self.best_solution, self.best_weight
