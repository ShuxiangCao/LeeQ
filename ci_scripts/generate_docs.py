import os
import shutil


def generate_docs(package_path, docs_path):

    shutil.rmtree(docs_path, ignore_errors=False, onerror=None)
    os.mkdir(docs_path)

    for root, dirs, files in os.walk(package_path):
        # Filter out __init__.py if you don't want to document package initialization
        files = [f for f in files if f.endswith('.py') and f != '__init__.py']
        for file in files:
            module_path = os.path.join(root, file)

            relative_path = os.path.relpath(module_path, package_path)
            module_name = relative_path.replace(os.path.sep, '.')[
                          :-3]  # Remove '.py' and replace os separators with dots

            if 'test_' in module_name:
                continue

            doc_file_path = os.path.join(docs_path, relative_path.replace('.py', '.md'))
            os.makedirs(os.path.dirname(doc_file_path), exist_ok=True)

            with open(doc_file_path, 'w') as md_file:
                md_file.write(f"# leeq.{module_name}\n"
                              f"::: leeq.{module_name}\n")


if __name__ == "__main__":
    package_path = './leeq'
    docs_path = './docs/code_reference'
    generate_docs(package_path, docs_path)
