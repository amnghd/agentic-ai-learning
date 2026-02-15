#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==> Setting up Week 1 Agentic AI project..."

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    local version_output=$($python_cmd --version 2>&1 || echo "")
    if [ -n "$version_output" ]; then
        local version=$(echo "$version_output" | awk '{print $2}')
        local major=$(echo "$version" | cut -d. -f1)
        local minor=$(echo "$version" | cut -d. -f2)
        
        # Check if major and minor are numeric
        if [[ "$major" =~ ^[0-9]+$ ]] && [[ "$minor" =~ ^[0-9]+$ ]]; then
            if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
                echo "$python_cmd"
                return 0
            fi
        fi
    fi
    return 1
}

# Try to find a suitable Python version
PYTHON_CMD=""
REQUIRED_MAJOR=3
REQUIRED_MINOR=11

# Check common Python command names
for cmd in python3.12 python3.11 python3 python; do
    if check_python_version "$cmd"; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python $REQUIRED_MAJOR.$REQUIRED_MINOR+ is required"
    echo ""
    echo "Current Python version: $(python3 --version 2>&1)"
    echo ""
    echo "To install Python 3.11+ on macOS:"
    echo ""
    echo "Option 1: Using Homebrew (recommended):"
    echo "  brew install python@3.11"
    echo "  # Then use: python3.11"
    echo ""
    echo "Option 2: Using pyenv (for managing multiple versions):"
    echo "  brew install pyenv"
    echo "  pyenv install 3.11.0"
    echo "  pyenv local 3.11.0"
    echo ""
    echo "Option 3: Download from python.org:"
    echo "  Visit: https://www.python.org/downloads/"
    echo ""
    echo "After installing, re-run this script."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "  Using Python $PYTHON_VERSION ($PYTHON_CMD)"

# Create virtual environment if not exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "==> Creating virtual environment..."
    $PYTHON_CMD -m venv "$PROJECT_ROOT/.venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Install dependencies
echo "==> Installing dependencies..."
bash "$SCRIPT_DIR/install_dependencies.sh"

# Initialize git repo if not already one
if ! git -C "$PROJECT_ROOT" rev-parse --git-dir > /dev/null 2>&1; then
    echo "==> Initializing git repository..."
    git -C "$PROJECT_ROOT" init
    git -C "$PROJECT_ROOT" add .
    git -C "$PROJECT_ROOT" commit -m "Initial project structure"
fi

# Install pre-commit hooks
echo "==> Installing pre-commit hooks..."
pre-commit install

# Create .env file from template if not present
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "==> Creating .env from template..."
    cat > "$PROJECT_ROOT/.env" <<EOF
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentic_db

# Redis
REDIS_URL=redis://localhost:6379/0
EOF
    echo "  .env created. Please update with your actual API keys."
fi

echo ""
echo "==> Setup complete! Activate your environment with:"
echo "    source .venv/bin/activate"
