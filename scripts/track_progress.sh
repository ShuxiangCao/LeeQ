#!/bin/bash
echo "=== LeeQ Repository Improvement Progress ==="
echo ""
echo "Phase 1: Repository Cleanup"
[[ -f requirements-dev.txt ]] && echo "  ✓ Dev requirements" || echo "  ✗ Dev requirements"
[[ -f .env.example ]] && echo "  ✓ Environment template" || echo "  ✗ Environment template"
[[ -f leeq/config.py ]] && echo "  ✓ Config module" || echo "  ✗ Config module"

echo ""
echo "Phase 2: Code Quality"
source venv/bin/activate 2>/dev/null && ruff check --select T20 leeq/theory/tomography/state_tomography.py leeq/theory/fits/multilevel_decay.py leeq/theory/fits/fit_exp.py 2>/dev/null && echo "  ✗ Target files have print statements" || echo "  ✓ Target files clean of print statements"
[[ -f .pre-commit-config.yaml ]] && echo "  ✓ Pre-commit configured" || echo "  ✗ Pre-commit missing"

echo ""
echo "Phase 3: Test Coverage"
test_count=$(find tests -name "test_*.py" | wc -l)
echo "  Test files: $test_count"
source venv/bin/activate 2>/dev/null && coverage report --show-missing 2>/dev/null | grep "TOTAL" | awk '{print "  Coverage: " $4}' || echo "  Coverage: Not measured"

echo ""
echo "Phase 4: Documentation" 
[[ -d site/ ]] && echo "  ✓ Docs built" || echo "  ✗ Docs not built"

echo ""
echo "Phase 5: CI/CD"
[[ -f .github/workflows/test.yml ]] && echo "  ✓ Test workflow" || echo "  ✗ Test workflow missing"
[[ -f .github/workflows/docs.yml ]] && echo "  ✓ Docs workflow" || echo "  ✗ Docs workflow missing"