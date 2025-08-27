#!/bin/bash
# Setup script for DAFT Orchestration System

set -e

echo "🚀 Setting up DAFT Orchestration System"
echo "======================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p scripts logs outputs

# Make Python scripts executable
echo "🔧 Making scripts executable..."
chmod +x orchestrate_experiment.py manage_experiments.py

# Check if required conda environments exist
echo "🐍 Checking conda environments..."

if ! conda env list | grep -q "^s1 "; then
    echo "⚠️  Warning: 's1' conda environment not found"
    echo "   Please create it with your training dependencies"
fi

if ! conda env list | grep -q "^sal "; then
    echo "⚠️  Warning: 'sal' conda environment not found"
    echo "   Please create it with your evaluation dependencies"
fi

# Check if HuggingFace CLI is configured
echo "🤗 Checking HuggingFace configuration..."
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  Warning: huggingface-cli not found"
    echo "   Install with: pip install huggingface_hub"
else
    if ! huggingface-cli whoami &> /dev/null; then
        echo "⚠️  Warning: HuggingFace not logged in"
        echo "   Login with: huggingface-cli login"
    else
        echo "✅ HuggingFace CLI configured"
    fi
fi

# Check if SLURM is available
echo "⚡ Checking SLURM availability..."
if ! command -v sbatch &> /dev/null; then
    echo "❌ Error: SLURM not found. This system requires SLURM."
    exit 1
else
    echo "✅ SLURM found"
fi

# Validate configuration files
echo "📋 Validating configuration files..."
for config_file in configs/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "  ✓ $(basename "$config_file")"
    fi
done

# Create a simple test
echo "🧪 Running basic validation..."
if python -c "import yaml; import os; import subprocess" 2>/dev/null; then
    echo "✅ Python dependencies available"
else
    echo "❌ Error: Missing Python dependencies (yaml, os, subprocess)"
    exit 1
fi

# Set up environment
echo "🌍 Environment setup..."
echo "export DAFT_ORCHESTRATION_DIR=\"$SCRIPT_DIR\"" > .env
echo "export PATH=\"\$PATH:$SCRIPT_DIR\"" >> .env

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📖 Next steps:"
echo "   1. Source the environment: source .env"
echo "   2. Run a test: ./orchestrate_experiment.py configs/quick_test_config.yaml --dry-run"
echo "   3. Check the README.md for detailed usage instructions"
echo ""
echo "🔍 Useful commands:"
echo "   ./orchestrate_experiment.py --help"
echo "   ./manage_experiments.py --help"
echo "   ./manage_experiments.py list"
echo ""
