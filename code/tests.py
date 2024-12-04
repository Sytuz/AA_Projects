from stresstester import stressTester
from algorithms import algorithms
from constants import *
from utils import utils
from visualgraphs import quick_algo_graph, quick_algo_compare_graphs, single_metric_comparison, exhaustive_comparison_time_operations
import networkx as nx

# The tests were organized and executed in a Jupyter Notebook, however the notebook quickly became desorganized and hard to read.
# Because of that, the tests were moved to this file so that the professor can easily read and understand them.

def graph_creation():
    """ Pre-generated graphs """
    """ ATTENTION: This function will take a long time to run, and occupy approximately 1.5GB of disk space """
    
    # Big Graphs for Greedy Algorithms
    # utils.graph_creation_and_save(3500, 0.125, 100)
    # utils.graph_creation_and_save(3500, 0.25, 100)
    # utils.graph_creation_and_save(3500, 0.50, 100)
    # utils.graph_creation_and_save(3500, 0.75, 100)
    
    # Smaller Graphs for Comparing With Exhaustive
    # Deprecated: utils.graph_creation_and_save()
    utils.graph_creation_and_save(500, 0.125, 1, file_name="small_graphs")
    utils.graph_creation_and_save(500, 0.25, 1, file_name="small_graphs")
    utils.graph_creation_and_save(500, 0.50, 1, file_name="small_graphs")
    utils.graph_creation_and_save(500, 0.75, 1, file_name="small_graphs")

def full_test():
    """ Full test of all algorithms """
    """ ATTENTION: This function will take a long time to run, and it will overwrite the data that is already stored"""
    
    # Deprecated: utils.full_stress_test()
    # Exhaustive
    utils.full_stress_test(algorithms.exhaustive_v2, base_filename="exhaustive", max_time_minutes=1, stored_graphs=False, sample_size=1)
    utils.full_stress_test(algorithms.exhaustive_v1, base_filename="exhaustive_v1", max_time_minutes=2, stored_graphs=False, sample_size=1)

    # Smaller Greedy Tests, For Comparing With Exhaustive
    utils.full_stress_test(algorithms.biggest_weight_first_v2, base_filename="biggest_weight_first_compare", n_max=800, stored_graphs=False, sample_size=1)
    utils.full_stress_test(algorithms.smallest_degree_first_v1, base_filename="smallest_degree_first_compare", n_max=800, stored_graphs=False, sample_size=1)
    utils.full_stress_test(algorithms.weight_to_degree_v1, base_filename="weight_to_degree_compare", n_max=800, stored_graphs=False, sample_size=1)

    # Normal Greedy Tests
    utils.full_stress_test(algorithms.biggest_weight_first_v2, base_filename="biggest_weight_first", stored_graphs=True, sample_size=100)
    utils.full_stress_test(algorithms.smallest_degree_first_v1, base_filename="smallest_degree_first", stored_graphs=True, sample_size=100)
    utils.full_stress_test(algorithms.weight_to_degree_v1, base_filename="weight_to_degree", stored_graphs=True, sample_size=100)
    
def full_test_with_iterations():
    """ Full test for algorithms that require iterations (randomized algorithms) """
    
    # Standard Randomized, Smaller For Comparing With Exhaustive
    #utils.full_stress_test(algorithms.monte_carlo, base_filename="monte_carlo_compare", n_max=800, stored_graphs=False, sample_size=1, iterations=[25, 50, 100, 250, 500, 750, 1000])
    #utils.full_stress_test(algorithms.monte_carlo_with_filter, base_filename="monte_carlo_with_filter_compare", n_max=800, stored_graphs=False, sample_size=1, iterations=[25, 50, 100, 250, 500, 750, 1000])
    utils.full_stress_test(algorithms.heuristic_monte_carlo, base_filename="heuristic_monte_carlo_compare", stored_graphs=True, sample_size=1, iterations=[25, 50, 100, 250, 500, 750, 1000])
    
def quick_precision_test(name, algorithm, k, n, iterations=1000, initial_temp=1000, cooling_rate=0.99, trials=1):
    """ Quick test to check the precision of an algorithm """
    print(f"Testing {name}'s precision")
    precision = algorithms.compare_precision(algorithm, k, n, func_iterations=iterations, initial_temp=initial_temp, cooling_rate=cooling_rate, iterations=trials)
    print(f"Precision: {precision}")
    
def main():
    """ Main function """
    #quick_precision_test("Monte Carlo", algorithms.monte_carlo, 0.75, 175, 250)
    #quick_precision_test("Heuristic Monte Carlo", algorithms.heuristic_monte_carlo, 0.75, 175, 250)
    #quick_precision_test("Simulated Annealing", algorithms.simulated_annealing, 0.75, 175, 1000, 1000, 0.99)
    #quick_precision_test("Threaded Heuristic Monte Carlo", algorithms.threaded_heuristic_monte_carlo, 0.75, 175, 250)

    # Graph Creation
    """
    for k in k_full:
        # Create small graphs for comparing with exhaustive
        utils.graph_creation(n=500, k=k, step=1, save_data={'output_dir':'../graphs/small_graphs', 'file_name':f'small_graphs'})
        
        # Create big graphs for greedy and randomized algorithms
        utils.graph_creation(n=3500, k=k, step=100, save_data={'output_dir':'../graphs/big_graphs', 'file_name':f'big_graphs'})
    """
    
    # Stress Tests
    for k in k_full:
        # Exhaustive
        """
        stressTester.stress_test(func=algorithms.exhaustive_v2, dirname="exhaustive_test", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.exhaustive_v1, dirname="exhaustive_v1_test", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        """
        
        # Greedy
        """
        # Smaller Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first_compare_test", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first_compare_test", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree_compare_test", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.csv")
        
        # Bigger Graphs
        stressTester.stress_test(func=algorithms.biggest_weight_first_v2, dirname="biggest_weight_first_test", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.smallest_degree_first_v1, dirname="smallest_degree_first_test", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        stressTester.stress_test(func=algorithms.weight_to_degree_v1, dirname="weight_to_degree_test", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.csv")
        """
        
        # Randomized
        
        # Number of iterations
        iterations = 2500
        
        # Smaller Graphs

        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo_test/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo_test/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing_test/small", dataset_file=f"../graphs/small_graphs/small_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")

        
        # Bigger Graphs

        """
        stressTester.stress_test(func=algorithms.monte_carlo, dirname="monte_carlo_test/big", ataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.parallel_heuristic_monte_carlo, dirname="parallel_heuristic_monte_carlo_test/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        stressTester.stress_test(func=algorithms.simulated_annealing, dirname="simulated_annealing_test/big", dataset_file=f"../graphs/big_graphs/big_graphs_{k}.jsonl", iterations=iterations, filename=f"results_{k}")
        """
    
if __name__ == "__main__":
    main()