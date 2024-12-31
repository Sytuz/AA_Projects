import heapq

class CountMinSketch:
    def __init__(self, k, h, n):
        """
        Initialize the Count-Min Sketch.

        Args:
            k (int): The number of counters per hash table.
            h (int): The number of hash tables.
            n (int): The number of top elements to track.
        """
        self.k = k
        self.h = h
        self.n = n
        self.counters = [[0] * k for _ in range(h)]
        self.hash_seeds = [i for i in range(h)]  # Using seeds to simulate multiple hash functions
        self.heap = []  # Min-heap to track the top n elements
        self.item_map = {}  # Map to track items in the heap

    def add(self, data):
        """
        Add data to the Count-Min Sketch.

        Args:
            data (str): The data item to add.
        """
        for i in range(self.h):
            hash_val = hash(data + str(self.hash_seeds[i])) % self.k
            self.counters[i][hash_val] += 1

        # Update the heap
        estimated_count = self.estimate(data)
        if data in self.item_map:
            # Update the heap if the item is already present
            self.item_map[data][0] = estimated_count
            heapq.heapify(self.heap)
        else:
            # Add new item to the heap
            if len(self.heap) < self.n:
                heapq.heappush(self.heap, [estimated_count, data])
                self.item_map[data] = self.heap[-1]
            else:
                # Replace the smallest if the current count is larger
                if estimated_count > self.heap[0][0]:
                    removed = heapq.heappushpop(self.heap, [estimated_count, data])
                    del self.item_map[removed[1]]
                    self.item_map[data] = self.heap[-1]

    def estimate(self, data):
        """
        Estimate the frequency of a data item.

        Args:
            data (str): The data item to query.

        Returns:
            int: The estimated frequency of the data item.
        """
        min_count = float('inf')
        for i in range(self.h):
            hash_val = hash(data + str(self.hash_seeds[i])) % self.k
            min_count = min(min_count, self.counters[i][hash_val])
        return min_count

    def all_estimates(self, data):
        """
        Estimate the frequency of all data items.

        Args:
            data (list): The list of data items to query.

        Returns:
            dict: A dictionary of the estimated frequencies for each data item.
        """
        return {word: self.estimate(word) for word in data}

    def top_n(self):
        """
        Retrieve the top n most frequent elements.

        Returns:
            dict: A dictionary of the top n elements and their counts.
        """
        return {item[1]: item[0] for item in self.heap if item[0] > 0}
    
    def total_memory(self):
        """
        Calculate the total memory used by the Count-Min Sketch.

        Returns:
            int: The total memory used by the Count-Min Sketch.
        """
        return self.k * self.h