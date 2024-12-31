from counters import Counters
from utils import Utils
import json

def main():
    filenames = ["../books/clean/don_quixote_english.txt", "../books/clean/don_quixote_spanish.txt", "../books/clean/don_quixote_french.txt"]
    languages = ["English", "Spanish", "French"]

    # Read the text from the files
    texts = [Utils.file_to_text(filename) for filename in filenames]

    # Display information about the texts
    for language, text in zip(languages, texts):
        print(f"\n{language} Text:")
        print(f"> Lenght - {len(text.split())} words ({len(text)} characters)")
        print(f"> Unique Words - {len(set(text.split()))}")

    # Count the frequency of each word in the texts
    exact_counters = [Counters.exact_counter(text.split()) for text in texts]
    approx_counters = [Counters.approx_counter(text.split()) for text in texts]
    misra_gries_counters = [Counters.misra_gries(text.split(), 1000) for text in texts]
    count_min_sketch_counters = [Counters.count_min_sketch(text.split(), 1000, 10, 10) for text in texts]
    # data_sketch_counters = [Counters.data_sketches(text.split(), 5) for text in texts]

    # Fixed probability k (p=1/2^k)
    approx_counter_k = 2

    max_table_width = 97

    # Print the most common words for each language, including memory usage details
    for language, exact_counter, approx_counter, misra_gries_counter, count_min_sketch_counter in zip(languages, exact_counters, approx_counters, misra_gries_counters, count_min_sketch_counters):
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
        top_misra_gries = sorted(misra_gries_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        top_count_min_sketch = sorted(count_min_sketch_counter.items(), key=lambda x: x[1], reverse=True)[:10]

        print(top_misra_gries)
        print(top_count_min_sketch)

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

        misra_gries_memory = Utils.memory_used(misra_gries_counter)
        count_min_sketch_memory = Utils.memory_used(count_min_sketch_counter)

        print(f"\n{'Memory Usage Details':<5}")
        print(f"{'-' * max_table_width}")
        print(f"{'Counter Type':<16} {'Num Counters':<15} {'Total Memory':<15} {'Average Memory':<15} {'Median':<15} {'Biggest Counter':<15} |")
        print(f"{'-' * max_table_width}")
        print(
            f"{'Exact':<16} {len(exact_counter):<15} {exact_memory['total']:<15} "
            f"{exact_memory['average_memory']:<15.7} {exact_memory['median']:<15} {exact_memory['biggest_counter']:<15} |"
        )
        print(
            f"{'Approx':<16} {len(approx_counter):<15} {approx_memory['total']:<15} "
            f"{approx_memory['average_memory']:<15.7} {approx_memory['median']:<15} {approx_memory['biggest_counter']:<15} |"
        )
        print(
            f"{'Misra-Gries':<16} {len(misra_gries_counter):<15} {misra_gries_memory['total']:<15} "
            f"{misra_gries_memory['average_memory']:<15.7} {misra_gries_memory['median']:<15} {misra_gries_memory['biggest_counter']:<15} |"
        )
        print(
            f"{'Count-Min Sketch':<16} {len(count_min_sketch_counter):<15} {count_min_sketch_memory['total']:<15} "
            f"{count_min_sketch_memory['average_memory']:<15.7} {count_min_sketch_memory['median']:<15} {count_min_sketch_memory['biggest_counter']:<15} |"
        )

        print(f"{'=' * max_table_width}")

    # Save all the results to files
    # '../data/counters/exact_counters_{language}.json' - exact counters
    # '../data/counters/approx_counters_{language}.json' - approximate counters
    # '../data/memory/memory_usage_{language}.json' - memory usage details

    memory_usage = {
        "exact": exact_memory,
        "approx": approx_memory,
        "misra_gries": misra_gries_memory,
        "count_min_sketch": count_min_sketch_memory
    }

    for language, exact_counter, approx_counter, misra_gries_counter, count_min_sketch_counter in zip(languages, exact_counters, approx_counters, misra_gries_counters, count_min_sketch_counters):
        with open(f"../data/counters/exact_counters_{language}.json", "w", encoding="UTF-8") as file:
            json.dump(exact_counter, file, ensure_ascii=False, indent=4)

        with open(f"../data/counters/approx_counters_{language}.json", "w", encoding="UTF-8") as file:
            json.dump(approx_counter, file, ensure_ascii=False, indent=4)

        with open(f"../data/counters/misra_gries_counters_{language}.json", "w", encoding="UTF-8") as file:
            json.dump(misra_gries_counter, file, ensure_ascii=False, indent=4)
        
        with open(f"../data/counters/count_min_sketch_counters_{language}.json", "w", encoding="UTF-8") as file:
            json.dump(count_min_sketch_counter, file, ensure_ascii=False, indent=4)

        with open(f"../data/memory/memory_usage_{language}.json", "w", encoding="UTF-8") as file:
            json.dump(memory_usage, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()