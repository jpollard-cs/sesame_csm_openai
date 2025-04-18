#!/usr/bin/env bash
# setup_macos.sh: Install system tools, Rust, Python venv, and dependencies for MPS on macOS
set -euo pipefail

# 1. Requires Homebrew
if ! command -v brew &>/dev/null; then
  echo "Homebrew not found, but required for setup. Please install Homebrew or complete the setup manually."
  exit 1
else
  echo "Homebrew is already installed."
fi

# 2. Install system dependencies via Homebrew
echo "Installing ffmpeg and libsndfile..."
brew update
brew install ffmpeg libsndfile

# 3. Install Rust toolchain if missing (needed for sphn, moshi)
if ! command -v rustup &>/dev/null; then
  echo "Rust toolchain not found. Installing rustup..."
  curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
else
  echo "Rust toolchain already installed."
fi
# Activate Cargo environment
source "$HOME/.cargo/env"

# 4. Ensure Python 3.10 is installed via Homebrew
echo "Checking for python3.10..."
if ! command -v python3.10 &>/dev/null; then
  echo "python3.10 not found. Installing python@3.10 via Homebrew..."
  brew install python@3.10
  # Add python3.10 to PATH
  export PATH="$(brew --prefix python@3.10)/bin:$PATH"
else
  echo "python3.10 is already installed"
fi

# Use python3.10 for venv
echo "Setting up Python 3.10 virtual environment..."
python3.10 -m venv .venv
source .venv/bin/activate

# 5. Install Python dependencies
echo "Updating pip and installing Python requirements..."
pip install --upgrade pip

# Install PyTorch, TorchAudio and TorchVision with Mac MPS support
echo "Installing PyTorch, TorchAudio and TorchVision with Mac MPS support..."
# Use PyTorch index for CPU wheels that include MPS backend
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

pip install -r requirements_macos.txt

# 6. Confirm MPS availability
python - << 'EOF'
import torch
print("MPS available:", torch.backends.mps.is_available(), "| Built:", torch.backends.mps.is_built())
EOF

echo "Setup complete. To start the API server, run:"
echo "  source .venv/bin/activate"
echo "  python -m app.main" 
