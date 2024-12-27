from counters import Counters
from utils import Utils

def main():
    filenames = ["../books/clean/don_quixote_english.txt", "../books/clean/don_quixote_spanish.txt", "../books/clean/don_quixote_hungarian.txt"]
    languages = ["english", "spanish", "hungarian"]

    # Read the text from the files
    texts = [Utils.file_to_text(filename) for filename in filenames]

    # Count the frequency of each word in the texts
    exact_counters = [Counters.exact_counter(text.split()) for text in texts]

    # Print the most common words for each language
    for language, counter in zip(languages, exact_counters):
        print(f"Most common words in {language}:")
        for word, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{word}: {count}")
        print()

if __name__ == "__main__":
    main()