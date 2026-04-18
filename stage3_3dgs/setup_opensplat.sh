#!/bin/bash
# Setup OpenSplat on macOS with Metal/MPS support
# Prerequisites: Xcode, CMake, OpenCV

set -e

echo "=== OpenSplat Setup for macOS (Metal/MPS) ==="

# Check prerequisites
if ! command -v cmake &> /dev/null; then
    echo "Installing CMake via Homebrew..."
    brew install cmake
fi

if ! command -v brew &> /dev/null; then
    echo "ERROR: Homebrew is required. Install from https://brew.sh"
    exit 1
fi

# Install OpenCV if needed
if ! brew list opencv &> /dev/null 2>&1; then
    echo "Installing OpenCV..."
    brew install opencv
fi

# Download LibTorch (PyTorch C++ distribution)
LIBTORCH_DIR="$HOME/.local/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading LibTorch for macOS (ARM64)..."
    mkdir -p "$HOME/.local"
    cd /tmp
    # Use the CPU version for Mac - MPS is handled by OpenSplat's build system
    curl -L -o libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip"
    unzip -q -o libtorch.zip -d "$HOME/.local/"
    rm libtorch.zip
    echo "LibTorch installed to $LIBTORCH_DIR"
else
    echo "LibTorch already installed at $LIBTORCH_DIR"
fi

# Clone and build OpenSplat
OPENSPLAT_DIR="$(dirname "$0")/OpenSplat"
if [ ! -d "$OPENSPLAT_DIR" ]; then
    echo "Cloning OpenSplat..."
    git clone https://github.com/pierotofy/OpenSplat.git "$OPENSPLAT_DIR"
fi

cd "$OPENSPLAT_DIR"
echo "Building OpenSplat with Metal/MPS support..."
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" -DGPU_RUNTIME=MPS ..
make -j$(sysctl -n hw.logicalcpu)

OPENSPLAT_BIN="$(pwd)/opensplat"
echo ""
echo "=== Build complete! ==="
echo "OpenSplat binary: $OPENSPLAT_BIN"
echo ""
echo "Add to PATH (optional):"
echo "  export PATH=\"$(pwd):\$PATH\""
echo ""
echo "Test with:"
echo "  $OPENSPLAT_BIN --help"
