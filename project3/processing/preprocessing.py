
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

def main():
    # Read the text from the file
    books = ["../books/raw/don_quixote_english.txt", "../books/raw/don_quixote_spanish.txt", "../books/raw/don_quixote_hungarian.txt"]
    output_filenames = ["../books/clean/don_quixote_english.txt", "../books/clean/don_quixote_spanish.txt", "../books/clean/don_quixote_hungarian.txt"]
    
    for book, output_filename in zip(books, output_filenames):
        with open(book, "r", encoding="UTF-8") as file:
            text = file.read()
            
        # Remove the header and footer
        text_clean = remove_gutenberg_header_footer(text)
        
        # Write the clean text to a new file
        with open(output_filename, "w", encoding="UTF-8") as file:
            file.write(text_clean)
        
if __name__ == "__main__":
    main()