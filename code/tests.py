from utils import utils
from algorithms import algorithms

# The tests were organized and executed in a Jupyter Notebook, however the notebook quickly became desorganized and hard to read.
# Because of that, the tests were moved to this file so that the professor can easily read and understand them.

def graph_creation():
    """ Pre-generated graphs """
    """ ATTENTION: This function will take a long time to run, and occupy approximately 1.5GB of disk space """
    utils.graph_creation_and_save(3500, 0.125, 100)
    utils.graph_creation_and_save(3500, 0.25, 100)
    utils.graph_creation_and_save(3500, 0.50, 100)
    utils.graph_creation_and_save(3500, 0.75, 100)

def full_test():
    """ Full test of all algorithms """
    """ ATTENTION: This function will take a long time to run, and it will overwrite the data that is already stored"""
    
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
    
def main():
    graph_creation()
    full_test()
    
if __name__ == "__main__":
    main()