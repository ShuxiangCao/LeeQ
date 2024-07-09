import os


def load_document_file(file_name: str):
    """
    Load the contents of a Markdown file.

    Parameters:
        file_name (str): The name of the Markdown file.

    Returns:
        str: The contents of the Markdown
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)

    # Construct the full path to the Markdown file
    md_file_path = os.path.join(current_dir, 'documents')
    md_file_path = os.path.join(md_file_path, file_name)

    # Read the contents of the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    return md_content
