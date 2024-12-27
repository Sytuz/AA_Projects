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