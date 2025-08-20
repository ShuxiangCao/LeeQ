#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path
import argparse

def transform_imports(file_path, dry_run=False):
    """Transform labchronicle imports to leeq.chronicle"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    
    # Transform patterns
    patterns = [
        (r'from labchronicle import', 'from leeq.chronicle import'),
        (r'from labchronicle\.', 'from leeq.chronicle.'),
        (r'import labchronicle\b', 'import leeq.chronicle as labchronicle'),
        (r'"labchronicle\.', '"leeq.chronicle.'),  # String references
        (r"'labchronicle\.", "'leeq.chronicle."),  # String references
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        if not dry_run:
            # Backup original
            shutil.copy2(file_path, f"{file_path}.bak")
            with open(file_path, 'w') as f:
                f.write(content)
        print(f"{'[DRY-RUN] ' if dry_run else ''}Updated: {file_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    dirs = ['leeq/', 'tests/', 'notebooks/', 'benchmark/']
    updated = 0
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            for py_file in Path(dir_path).rglob('*.py'):
                if 'chronicle' not in str(py_file):
                    if transform_imports(py_file, args.dry_run):
                        updated += 1
    
    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Total files updated: {updated}")

if __name__ == '__main__':
    main()