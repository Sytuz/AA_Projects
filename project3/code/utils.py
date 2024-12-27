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