from stresstester import stressTester
from algorithms import algorithms
from constants import k_full
from utils import utils
import time

# The tests were organized and executed in a Jupyter Notebook, however the notebook quickly became desorganized and hard to read.
# Because of that, the tests were moved to this file so that the professor can easily read and understand them.

def graph_creation():
    """ Pre-generated graphs """
    """ ATTENTION: This function will take a long time to run, and occupy approximately 1.5GB of disk space """
    
    for k in k_full:
        # Create small graphs for comparing with exhaustive
        utils.graph_creation(n=500, k=k, step=1, save_data={'output_dir':'../graphs/small_graphs', 'file_name':'small_graphs'})
        
        # Create big graphs for greedy and randomized algorithms
        utils.graph_creation(n=3500, k=k, step=100, save_data={'output_dir':'../graphs/big_graphs', 'file_name':'big_graphs'})
        
def quick_test():
    stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl")
    stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)
    stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)
    stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)

def full_test(iterations=2500):
    """ Full test of all algorithms """
    """ ATTENTION: This function will take a long time to run, and it will overwrite the data that is already stored"""
    
    for k in k_full:
        # Exhaustive
        stressTester.stress_test(func=algorithms.exhaustive_v2, dirname="exhaustive", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.exhaustive_v1, dirname="exhaustive_v1", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", filename=f"results_{k}")
        
        # Greedy
        # Smaller Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", filename=f"results_{k}")
        
        # Bigger Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", filename=f"results_{k}")
        
        # Randomized
        
        # Smaller Graphs
        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        
        # Bigger Graphs

        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
    
    
    # Pre-Generated Graphs - For Non-Exhaustive Algorithms
    stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree/pregen")
    stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/pregen", iterations=iterations)
    stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/spregen", iterations=iterations)
    stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/pregen", iterations=iterations)

def largest_graph_test():
    G = utils.create_graph_v4(25000, 0.25)
    print(f"Graph generated with {len(G.nodes)} nodes and {len(G.edges)} edges (k=0.25)\n")

    for algorithm in [algorithms.weight_to_degree_v1, algorithms.monte_carlo, algorithms.parallel_heuristic_monte_carlo]:
        start_time = time.time()
        if algorithm == algorithms.weight_to_degree_v1:
            solution, ops = algorithm(G)
        else:
            solution, ops = algorithm(G, 1000)
        end_time = time.time()

        print(f"Algorithm: {algorithm.__name__}")
        print(f"Number of Operations: {ops}")
        print(f"Total Weight: {sum(G.nodes[node]['weight'] for node in solution)}")
        print(f"Time taken: {end_time - start_time} seconds\n")
    
        
def quick_precision_test(name, algorithm, k, n, iterations=1000, initial_temp=1000, cooling_rate=0.99, trials=1):
    """ Quick test to check the precision of an algorithm """
    print(f"Testing {name}'s precision")
    precision = algorithms.compare_precision(algorithm, k, n, func_iterations=iterations, initial_temp=initial_temp, cooling_rate=cooling_rate, iterations=trials)
    print(f"Precision: {precision}")
    
def main():
    """ Main function (uncomment to run the tests) """
    
    # Graph Creation
    #graph_creation()
    
    # Quick Tests (to make sure the algorithms are working)
    #quick_test()
    
    # Stress Tests
    #full_test()        
    
    # Largest Graph Test
    #largest_graph_test()    
if __name__ == "__main__":
    main()