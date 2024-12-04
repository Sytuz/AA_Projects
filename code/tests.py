from stresstester import stressTester
from algorithms import algorithms
from constants import k_full
from utils import utils
from visualgraphs import quick_algo_graph, quick_algo_compare_graphs, single_metric_comparison, exhaustive_comparison_time_operations

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
    stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)
    stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)
    stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing", dataset_file="../graphs/small_graphs/small_graphs_0.125.jsonl", iterations=1000)

def full_test(iterations=2500):
    """ Full test of all algorithms """
    """ ATTENTION: This function will take a long time to run, and it will overwrite the data that is already stored"""
    for k in k_full:
        # Exhaustive
        stressTester.stress_test(func=algorithms.exhaustive_v2, dirname="exhaustive", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.exhaustive_v1, dirname="exhaustive_v1", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        
        # Greedy
        # Smaller Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first_compare/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first_compare/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree_compare/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        
        # Bigger Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        
        # Randomized
        
        # Smaller Graphs
        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        
        # Bigger Graphs

        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/big", ataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        
    # Pre-Generated Graphs - For Non-Exhaustive Algorithms
    stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree", iterations=iterations)
    stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo/small", iterations=iterations)
    stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo/small", iterations=iterations)
    stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing/small", iterations=iterations)
    
        
def quick_precision_test(name, algorithm, k, n, iterations=1000, initial_temp=1000, cooling_rate=0.99, trials=1):
    """ Quick test to check the precision of an algorithm """
    print(f"Testing {name}'s precision")
    precision = algorithms.compare_precision(algorithm, k, n, func_iterations=iterations, initial_temp=initial_temp, cooling_rate=cooling_rate, iterations=trials)
    print(f"Precision: {precision}")
    
def main():
    """ Main function """
    
    # Graph Creation
    #graph_creation()
    
    # Quick Tests (to make sure the algorithms are working)
    quick_test()
    
    # Stress Tests
    #full_test()        
    
    # Comparisons
    #quick_algo_compare_graphs(["Monte Carlo", "Parallel Heuristic Monte Carlo", "Simulated Annealing"], ["monte_carlo_test/small/results_0.125.csv", "parallel_heuristic_monte_carlo_test/small/results_0.125.csv", "simulated_annealing_test/small/results_0.125.csv"])
    #quick_algo_graph("Monte Carlo", "monte_carlo_test/small/results_0.125.csv")
    #quick_algo_graph("Monte Carlo", "monte_carlo_test/results_0.125.csv")
    #quick_algo_graph("Simulated Annealing", "simulated_annealing_test/results_0.125.csv")
    
    #quick_algo_compare_graphs([f"Monte Carlo - k={k}" for k in k_full], [f"monte_carlo_test/small/results_{k}.csv" for k in k_full], output_filename="../data/monte_carlo_test/monte_carlo_small_comparison")
    #quick_algo_compare_graphs([f"Parallel Heuristic Monte Carlo - k={k}" for k in k_full], [f"parallel_heuristic_monte_carlo_test/small/results_{k}.csv" for k in k_full], output_filename="../data/parallel_heuristic_monte_test/parallel_heuristic_monte_carlo_small_comparison")
    #quick_algo_compare_graphs([f"Simulated Annealing - k={k}" for k in k_full], [f"simulated_annealing_test/small/results_{k}.csv" for k in k_full], output_filename="../data/simulated_annealing_test/simulated_annealing_small_comparison")
    
if __name__ == "__main__":
    main()