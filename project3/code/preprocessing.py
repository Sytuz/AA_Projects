import nltk
from nltk.corpus import stopwords
from typing import List

# Ensure the stopwords are downloaded
nltk.download('stopwords')
nltk.download('punkt_tab')

CUSTOM_STOPWORDS = {
    "english": [],
    "spanish": ['—'],
    "french": ['–']
}

def remove_gutenberg_header_footer(text: str) -> str:
    """
    Removes the Project Gutenberg header and footer from the text.

    Args:
        text (str): The full text of the book as a string.

    Returns:
        str: The text with the header and footer removed.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    # Find the start of the actual content
    start_index = text.find(start_marker)
    if start_index != -1:
        start_index = text.find("\n", start_index) + 1  # Move to the next line after the marker
    
    # Find the end of the actual content
    end_index = text.find(end_marker)
    if end_index != -1:
        end_index = text.rfind("\n", 0, end_index)  # Move to the line before the marker
    
    # Return the cleaned text
    if start_index != -1 and end_index != -1:
        return text[start_index:end_index].strip()
    else:
        raise ValueError("Markers for header and footer not found in the text.")

def remove_stopwords(text: str, language: str) -> str:
    """
    Remove stopwords from a given text based on the specified language.

    Args:
        text (str): The input text.
        language (str): The language of the stopwords. Must be 'english', 'spanish', or 'french'.

    Returns:
        str: The text with stopwords removed.
    """
    if language not in ['english', 'spanish', 'french']:
        raise ValueError("Language must be one of: 'english', 'spanish', 'french'.")
    
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove punctuation
    words = [word.strip(".,;:!?()[]{}\"”“‘’—–/") for word in words if word.strip(".,;:!?()[]{}\"”“‘’—–/")]

    # Remove unwanted .jpg
    words = [word for word in words if ".jpg" not in word]
    
    # Get the stopwords for the specified language
    stop_words = set(stopwords.words(language))
    
    # Filter out the stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words and word not in CUSTOM_STOPWORDS[language]]
    
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

def main():
    # Read the text from the file
    books = ["../books/raw/don_quixote_english.txt", "../books/raw/don_quixote_spanish.txt", "../books/raw/don_quixote_french.txt"]
    output_filenames = ["../books/clean/don_quixote_english.txt", "../books/clean/don_quixote_spanish.txt", "../books/clean/don_quixote_french.txt"]
    languages = ["english", "spanish", "french"]
    
    for book, output_filename, language in zip(books, output_filenames, languages):
        with open(book, "r", encoding="UTF-8") as file:
            text = file.read()
            
        # Remove the header and footer
        text_no_header_footer = remove_gutenberg_header_footer(text)

        # All text to lowercase
        text_no_header_footer = text_no_header_footer.lower()

        # Remove stopwords
        text_clean = remove_stopwords(text_no_header_footer, language=language)
        
        # Write the clean text to a new file
        with open(output_filename, "w", encoding="UTF-8") as file:
            file.write(text_clean)
        
if __name__ == "__main__":
    main()