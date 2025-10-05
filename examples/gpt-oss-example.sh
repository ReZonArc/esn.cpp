#!/bin/bash
# Example script for running gpt-oss models with llama.cpp
# 
# This script provides simple examples of how to run gpt-oss models
# with different hardware configurations.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if llama-server exists
if ! command -v ./build/bin/llama-server &> /dev/null; then
    echo -e "${RED}Error: llama-server not found. Please build llama.cpp first.${NC}"
    echo "Run: cmake -B build && cmake --build build --config Release -j \$(nproc)"
    exit 1
fi

print_usage() {
    echo "Usage: $0 [OPTIONS] MODEL_SIZE"
    echo ""
    echo "MODEL_SIZE:"
    echo "  20b     - Run gpt-oss-20b (requires ~15GB RAM/VRAM)"
    echo "  120b    - Run gpt-oss-120b (requires ~64GB RAM/VRAM)"
    echo ""
    echo "OPTIONS:"
    echo "  --cpu-layers N  - Keep N MoE layers on CPU (adjust for your VRAM)"
    echo "  --context N     - Set context size (default: 0 = model default)"
    echo "  --cpu-only      - Run entirely on CPU"
    echo "  --help          - Show this help"
    echo ""
    echo "Hardware configuration examples:"
    echo "  RTX 4090 (24GB):  $0 20b"
    echo "  RTX 4090 (24GB):  $0 120b --cpu-layers 8"
    echo "  RTX 4080 (16GB):  $0 20b --cpu-layers 8"
    echo "  RTX 2060 (8GB):   $0 20b --cpu-layers 22"
    echo "  CPU only:         $0 20b --cpu-only"
}

# Default values
MODEL_SIZE=""
CPU_LAYERS=""
CONTEXT_SIZE="0"
CPU_ONLY=false
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-layers)
            CPU_LAYERS="$2"
            shift 2
            ;;
        --context)
            CONTEXT_SIZE="$2"
            shift 2
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        20b|120b)
            MODEL_SIZE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check if model size is specified
if [[ -z "$MODEL_SIZE" ]]; then
    echo -e "${RED}Error: Please specify model size (20b or 120b)${NC}"
    print_usage
    exit 1
fi

# Set model repository based on size
if [[ "$MODEL_SIZE" == "20b" ]]; then
    MODEL_REPO="ggml-org/gpt-oss-20b-GGUF"
    echo -e "${GREEN}Running gpt-oss 20B model${NC}"
elif [[ "$MODEL_SIZE" == "120b" ]]; then
    MODEL_REPO="ggml-org/gpt-oss-120b-GGUF"
    echo -e "${GREEN}Running gpt-oss 120B model${NC}"
fi

# Build command arguments
CMD_ARGS="-hf $MODEL_REPO --ctx-size $CONTEXT_SIZE --jinja -ub 2048 -b 2048"

# Add CPU-specific arguments
if [[ "$CPU_ONLY" == true ]]; then
    CMD_ARGS="$CMD_ARGS --cpu-moe"
    echo -e "${YELLOW}Running in CPU-only mode${NC}"
elif [[ -n "$CPU_LAYERS" ]]; then
    CMD_ARGS="$CMD_ARGS --n-cpu-moe $CPU_LAYERS"
    echo -e "${YELLOW}Keeping $CPU_LAYERS MoE layers on CPU${NC}"
fi

# Show the command that will be executed
echo -e "${GREEN}Command:${NC} ./build/bin/llama-server $CMD_ARGS"
echo ""
echo -e "${YELLOW}Starting llama-server...${NC}"
echo "Once loaded, you can:"
echo "  - Open http://localhost:8080 in your browser for the web UI"
echo "  - Use the OpenAI-compatible API at http://localhost:8080/v1/chat/completions"
echo "  - Press Ctrl+C to stop the server"
echo ""

# Execute the command
exec ./build/bin/llama-server $CMD_ARGS