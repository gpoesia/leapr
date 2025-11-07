#!/bin/bash

echo "Migrating existing results to v1 folders..."

# Create v1 directories
mkdir -p results/features/v1
mkdir -p results/evals/v1
mkdir -p results/models/v1

# Create v2 directories
mkdir -p results/features/v2
mkdir -p results/evals/v2
mkdir -p results/models/v2

# Move feature, evals, model files 
if ls results/features/*.json 1> /dev/null 2>&1; then
    echo "Moving feature files..."
    mv results/features/*.json results/features/v1/ 2>/dev/null || true
fi

if ls results/evals/*.json 1> /dev/null 2>&1; then
    echo "Moving eval files..."
    mv results/evals/*.json results/evals/v1/ 2>/dev/null || true
fi

if ls results/models/*.pkl 1> /dev/null 2>&1; then
    echo "Moving model files..."
    mv results/models/*.pkl results/models/v1/ 2>/dev/null || true
fi

echo ""
echo "✓ Migration complete!"
echo ""
echo "V1 files are now in:"
echo "  - results/features/v1/"
echo "  - results/evals/v1/"
echo "  - results/models/v1/"
