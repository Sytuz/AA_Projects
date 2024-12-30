import math

class Utils:

    @staticmethod
    def file_to_text(filename: str) -> str:
        """
        Read the text from a file.

        Args:
            filename (str): The path to the file.

        Returns:
            str: The text content of the file.
        """
        with open(filename, 'r', encoding="UTF-8") as file:
            return file.read()
        
    @staticmethod
    def memory_used(counters: dict) -> list:
        """
        Calculate the memory used by the counters.

        Args:
            counters (dict): The dictionary.

        Returns:
            dict: Memory usage details based on the values of the counters.
        """

        memory = [max(1, math.ceil(math.log2(counter))) for counter in counters.values()]
        total = sum(memory)
        average_memory = sum(memory) / len(memory)
        median = memory[len(memory) // 2]
        biggest_counter = max(memory)

        return {
            "memory": memory,
            "total": total,
            "average_memory": average_memory,
            "median": median,
            "biggest_counter": biggest_counter
        }