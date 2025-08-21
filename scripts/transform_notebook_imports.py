#!/usr/bin/env python3
import json
import re
import shutil
from pathlib import Path
import argparse

def transform_notebook_imports(file_path, dry_run=False):
    """Transform labchronicle imports to leeq.chronicle in Jupyter notebooks"""
    
    with open(file_path, 'r') as f:
        notebook = json.load(f)
    
    original = json.dumps(notebook, sort_keys=True)
    
    # Transform patterns
    patterns = [
        (r'from labchronicle import', 'from leeq.chronicle import'),
        (r'from labchronicle\.', 'from leeq.chronicle.'),
        (r'import labchronicle\b', 'import leeq.chronicle as labchronicle'),
        (r'"labchronicle\.', '"leeq.chronicle.'),  # String references
        (r"'labchronicle\.", "'leeq.chronicle."),  # String references
    ]
    
    changed = False
    
    # Process each cell
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                # Join source lines for processing
                source_text = ''.join(source)
                original_source = source_text
                
                # Apply transformations
                for pattern, replacement in patterns:
                    source_text = re.sub(pattern, replacement, source_text)
                
                if source_text != original_source:
                    # Split back into lines
                    cell['source'] = [line for line in source_text.splitlines(keepends=True)]
                    changed = True
    
    if changed:
        if not dry_run:
            # Backup original
            shutil.copy2(file_path, f"{file_path}.bak")
            with open(file_path, 'w') as f:
                json.dump(notebook, f, indent=1)
        print(f"{'[DRY-RUN] ' if dry_run else ''}Updated: {file_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    dirs = ['notebooks/', 'benchmark/']
    updated = 0
    
    for dir_path in dirs:
        if Path(dir_path).exists():
            for ipynb_file in Path(dir_path).rglob('*.ipynb'):
                if transform_notebook_imports(ipynb_file, args.dry_run):
                    updated += 1
    
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Total notebook files updated: {updated}")

if __name__ == '__main__':
    main()