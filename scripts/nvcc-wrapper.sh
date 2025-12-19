#!/bin/bash
# Wrapper script to ensure nvcc is in PATH before compilation

# Add CUDA to PATH if not already present
if ! command -v nvcc &> /dev/null; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi

# Execute nvcc with all provided arguments
exec nvcc "$@"
