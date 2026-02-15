# Installing Python 3.11+ on macOS

## Quick Installation Options

### Option 1: Install Homebrew (Recommended)

If you don't have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Python 3.11:

```bash
brew install python@3.11
```

After installation, you can use `python3.11` command. The setup script will automatically detect it.

### Option 2: Direct Download from python.org

1. Visit: https://www.python.org/downloads/
2. Download Python 3.11 or later for macOS
3. Run the installer
4. Make sure to check "Add Python to PATH" during installation

### Option 3: Using pyenv (For Managing Multiple Versions)

Install pyenv:

```bash
# Install pyenv via Homebrew
brew install pyenv

# Or install manually
curl https://pyenv.run | bash

# Add to your shell profile (~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

Then install Python 3.11:

```bash
pyenv install 3.11.0
pyenv local 3.11.0  # For this project only
# OR
pyenv global 3.11.0  # For all projects
```

## Verify Installation

After installing, verify it works:

```bash
python3.11 --version
# Should show: Python 3.11.x
```

## Re-run Setup

Once Python 3.11+ is installed, re-run the setup script:

```bash
cd /Users/aminghd/Agentic\ AI\ project/agentic_learning/Projects/week1
bash scripts/setup.sh
```

The updated script will automatically detect Python 3.11+ if it's available.

## Troubleshooting

### If python3.11 command not found after installation:

1. **Homebrew**: Make sure Homebrew's bin directory is in your PATH:
   ```bash
   echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **pyenv**: Make sure pyenv is initialized in your shell:
   ```bash
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **Check all Python versions**:
   ```bash
   ls -la /usr/local/bin/python*  # Homebrew location
   ls -la ~/.pyenv/versions/       # pyenv location
   which python3.11
   ```

### If you need to use a specific Python version:

You can modify the setup script to use a specific Python version by editing `scripts/setup.sh` and changing the `PYTHON_CMD` variable, or set an environment variable:

```bash
export PYTHON_CMD=python3.11
bash scripts/setup.sh
```
