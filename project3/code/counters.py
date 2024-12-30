import random

class Counters:

    @staticmethod
    # Count the exact frequency of each word in the data
    def exact_counter(data):
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
    def approx_counter(data):
        counters = {}
        for word in data:
            if word in counters:
                if random.random() < 0.5:
                    counters[word] += 1
            else:
                counters[word] = 1
        return counters
    
    @staticmethod
    # Count the frequency of each word using a data stream algorithm, with a fixed memory size
    # Frequent-Count algorithm
    def stream_counter(data, memory_size):
        counters = {}
        for word in data:
            if word in counters:
                counters[word] += 1
            elif len(counters) < memory_size:
                counters[word] = 1
            else:
                for key in list(counters.keys()):
                    counters[key] -= 1
                    if counters[key] == 0:
                        del counters[key]
        return counters