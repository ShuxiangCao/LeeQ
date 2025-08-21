#!/usr/bin/env python3
"""Replace print statements with logging in key files."""

import re
from pathlib import Path
from typing import List, Tuple

# Priority files to update first (from the PRP)
PRIORITY_FILES = [
    "leeq/theory/tomography/state_tomography.py",
    "leeq/theory/fits/multilevel_decay.py", 
    "leeq/theory/fits/fit_exp.py",
    "leeq/core/elements/elements.py",
    "leeq/utils/utils.py",
]

def setup_logging_import(content: str) -> str:
    """Add logging import if not present."""
    if "from leeq.utils.utils import setup_logging" in content:
        return content
    
    if "import logging" in content or "from logging import" in content:
        return content
        
    # Find where to insert the import
    lines = content.split('\n')
    import_index = 0
    
    # Find last import line
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_index = i + 1
    
    # Insert logging setup
    lines.insert(import_index, "from leeq.utils.utils import setup_logging")
    lines.insert(import_index + 1, "logger = setup_logging(__name__)")
    lines.insert(import_index + 2, "")
    
    return '\n'.join(lines)

def replace_print_statements(content: str) -> str:
    """Replace print statements with logger calls."""
    # Handle print() calls
    content = re.sub(r'print\((.*?)\)', r'logger.info(\1)', content)
    
    # Handle pprint() calls  
    content = re.sub(r'pprint\((.*?)\)', r'logger.info(\1)', content)
    
    return content

def process_file(file_path: Path) -> bool:
    """Process a single file to replace print statements."""
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    try:
        content = file_path.read_text()
        original_content = content
        
        # Add logging setup
        content = setup_logging_import(content)
        
        # Replace print statements
        content = replace_print_statements(content)
        
        if content != original_content:
            # Backup original file
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            backup_path.write_text(original_content)
            
            # Write updated content
            file_path.write_text(content)
            print(f"Updated: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function."""
    updated_count = 0
    
    for file_path_str in PRIORITY_FILES:
        file_path = Path(file_path_str)
        if process_file(file_path):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} priority files.")
    print("Run 'ruff check --select T20 leeq/' to check remaining print statements.")

if __name__ == "__main__":
    main()