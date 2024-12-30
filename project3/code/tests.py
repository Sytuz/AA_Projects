from counters import Counters
from utils import Utils

def main():
    filenames = ["../books/clean/don_quixote_english.txt", "../books/clean/don_quixote_spanish.txt", "../books/clean/don_quixote_hungarian.txt"]
    languages = ["English", "Spanish", "Hungarian"]

    # Read the text from the files
    texts = [Utils.file_to_text(filename) for filename in filenames]

    # Count the frequency of each word in the texts
    exact_counters = [Counters.exact_counter(text.split()) for text in texts]
    approx_counters = [Counters.approx_counter(text.split()) for text in texts]

    # Fixed probability k (p=1/2^k)
    approx_counter_k = 2

    max_table_width = 97

    # Print the most common words for each language, including memory usage details
    for language, exact_counter, approx_counter in zip(languages, exact_counters, approx_counters):
        print(f"\n{'=' * max_table_width}")
        print(f"Most Common Words in {language}")
        print(f"{'=' * max_table_width}")
        print(f"      | {'Exact Counter':<24} | {'Approx Counter':<60} |")
        print(f"{'-' * 5} | {'-' * 24} | {'-' * 60} |")
        print(f"{'Rank':<5} | {'Word':<15} {'Count':<8} | {'Word':<15} {f'Count':<8} {f'Absolute Error':<15} {f'Relative Error':<15} {' ' * 3} |")
        print(f"{'-' * 5} | {'-' * 24} | {'-' * 60} |")

        # Get top 10 words for both exact and approximate counters
        top_exact = sorted(exact_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        top_approx = sorted(approx_counter.items(), key=lambda x: x[1], reverse=True)[:10]

        # Print side-by-side comparison with better alignment
        for i in range(10):
            exact_word, exact_count = top_exact[i] if i < len(top_exact) else ("-", 0)
            approx_word, approx_count = top_approx[i] if i < len(top_approx) else ("-", 0)
            print(f"{i + 1:<5} | {exact_word:<15} {exact_count:<8} | {approx_word:<15} {approx_count * approx_counter_k:<8} {abs(exact_count - approx_count * approx_counter_k):<15} {abs(exact_count - approx_count * approx_counter_k) / exact_count:<15.07} {' ' * 3} |")
        
        # Add a separator line after the loop to finish this section
        print(f"{'-' * max_table_width}")

        # Calculate memory usage for exact and approximate counters
        exact_memory = Utils.memory_used(exact_counter)
        approx_memory = Utils.memory_used(approx_counter)
        num_exact_counters = len(exact_counter)
        num_approx_counters = len(approx_counter)

        print(f"\n{'Memory Usage Details':<5}")
        print(f"{'-' * max_table_width}")
        print(f"{'Counter Type':<15} {'Num Counters':<15} {'Total Memory':<15} {'Average Memory':<15} {'Median':<15} {'Biggest Counter':<15} |")
        print(f"{'-' * max_table_width}")
        print(
            f"{'Exact':<15} {num_exact_counters:<15} {exact_memory['total']:<15} "
            f"{exact_memory['average_memory']:<15.7} {exact_memory['median']:<15} {exact_memory['biggest_counter']:<15} |"
        )
        print(
            f"{'Approx':<15} {num_approx_counters:<15} {approx_memory['total']:<15} "
            f"{approx_memory['average_memory']:<15.7} {approx_memory['median']:<15} {approx_memory['biggest_counter']:<15} |"
        )

        print(f"{'=' * max_table_width}")


if __name__ == "__main__":
    main()