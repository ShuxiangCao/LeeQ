#!/usr/bin/env python3
"""Validate all quality improvements."""

import subprocess
import sys
import os


def run_check(name: str, cmd: list) -> bool:
    """Run a validation check."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        status = "✓" if result.returncode == 0 else "✗"
        print(f"{status} {name}")
        if result.returncode != 0:
            print(f"  Error: {result.stderr[:200]}")
        return result.returncode == 0
    except Exception as e:
        print(f"✗ {name} - Exception: {e}")
        return False


def main():
    """Run all validation checks."""
    checks = [
        ("Target files clean of print statements", ["ruff", "check", "--select", "T20", "src/leeq/theory/tomography/state_tomography.py", "src/leeq/theory/fits/multilevel_decay.py", "src/leeq/theory/fits/fit_exp.py"]),
        ("Linting passes", ["ruff", "check", "src/leeq/", "--statistics"]),
        ("Type checking", ["mypy", "src/leeq/", "--ignore-missing-imports"]),
        ("Basic test execution", ["pytest", "tests/experiments/", "-v", "--ignore=tests/utils/ai/", "--maxfail=3"]),
        ("Coverage measurement", ["pytest", "tests/experiments/", "--cov=leeq", "--cov-fail-under=20", "--ignore=tests/utils/ai/"]),
    ]
    
    print("=== LeeQ Repository Quality Validation ===")
    print()
    
    all_passed = True
    for name, cmd in checks:
        passed = run_check(name, cmd)
        all_passed = all_passed and passed
    
    print()
    print("=== File Structure Validation ===")
    
    required_files = [
        "requirements-dev.txt",
        ".env.example", 
        "src/leeq/config.py",
        ".pre-commit-config.yaml",
        ".coveragerc",
        ".github/workflows/test.yml",
        ".github/workflows/docs.yml",
        ".github/dependabot.yml"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All validation checks passed!")
        return 0
    else:
        print("❌ Some validation checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
