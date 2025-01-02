from datasketches import frequent_strings_sketch, frequent_items_error_type
from count_min_sketch import CountMinSketch
from utils import Utils
import random
from typing import Tuple

class Counters:

    @staticmethod
    # Count the exact frequency of each word in the data
    def exact_counter(data):
        """
        Exact counter for finding the frequency of items in a data stream.
        
        Args:
            data (list): The data stream.
            
        Returns:
            dict: A dictionary of the counts for each data item.
        """
        counters = {}
        for word in data:
            if word in counters:
                counters[word] += 1
            else:
                counters[word] = 1
        return counters
    
    @staticmethod
    # Count the approximate frequency of each word in the data
    # Fixed probability counter : 1/2 (1/2^k, k=1)
    def approx_counter(data: list, p: float = 0.5) -> dict:
        """
        Approximate counter with fixed probability of p, default is 0.5.
        
        Args:
            data (list): The data stream.
            p (float): The probability of counting each data item.
            
        Returns:
            dict: A dictionary of the approximate counts for each data item.
        """
        counters = {}
        for word in data:
            if random.random() < p:
                if word in counters:
                    counters[word] += 1
                else:
                    counters[word] = 1
        return counters
    
    # --- Frequent-Count datastream algorithms ---

    @staticmethod
    # Misra-Gries Algorithm
    # https://www.sciencedirect.com/science/article/pii/0167642382900120

    def misra_gries(data: list, k: int, min_frequency=2) -> dict:
        """
        Misra-Gries Algorithm for finding the top k frequent items in a data stream.

        Args:
            data (list): The data stream.
            k (int): The number of top items to find.

        Returns:
            dict: A dictionary of the top k items and their counts.
        """
        data = Utils.filter_rare_words(data, min_frequency)
        counters = {}
        for word in data:
            if word in counters:
                counters[word] += 1
            elif len(counters) < k - 1:
                counters[word] = 1
            else:
                for key in list(counters.keys()):
                    counters[key] -= 1
                    if counters[key] == 0:
                        del counters[key]
        return counters
    
    @staticmethod
    # Count-Min Sketch Algorithm
    # https://www.sciencedirect.com/science/article/abs/pii/S0196677403001913
    def count_min_sketch(data: list, k: int, h: int, n: int) -> Tuple[dict, dict]:
        """
        Count-Min Sketch Algorithm for estimating the frequency of items in a data stream.
        
        Args:
            data (list): The data stream.
            k (int): The number of counters per hash table.
            h (int): The number of hash tables.
            n (int): The wanted number of top items.
            
        Returns:
            dict: A dictionary of the estimated frequencies for each data item.
        """
        cms = CountMinSketch(k, h, n)
        for word in data:
            cms.add(word)
        
        return cms.top_n(), cms.memory_details()
    
    #@staticmethod
    # Data Sketches Algorithm
    # https://arxiv.org/abs/1705.07001
    # def data_sketches(data: list, k: int) -> dict:
    #     """
    #     Data Sketches Algorithm, based on Apache DataSketches library, for estimating the frequency of items in a data stream.
    #     
    #     Args:
    #         data (list): The data stream.
    #         k (int): The number of counters.
    #         
    #     Returns:
    #         frequent_strings_sketch: A sketch of the estimated frequencies for each data item.
    #     """
    #     sketch = frequent_strings_sketch(k)
    #     for word in data:
    #         sketch.update(word)
    # 
    #     # Check sketch integrity
    #     print(sketch.get_frequent_items(frequent_items_error_type.NO_FALSE_NEGATIVES))
    #     return sketch.get_frequent_items(frequent_items_error_type.NO_FALSE_NEGATIVES)