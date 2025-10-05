# gpt-oss Examples

This directory contains examples for running gpt-oss models with llama.cpp.

## Quick Start Script

The `gpt-oss-example.sh` script provides an easy way to start gpt-oss models with hardware-appropriate settings:

```bash
# For RTX 4090 (24GB VRAM)
./examples/gpt-oss-example.sh 20b

# For RTX 4080 (16GB VRAM) 
./examples/gpt-oss-example.sh 20b --cpu-layers 8

# For RTX 2060 (8GB VRAM)
./examples/gpt-oss-example.sh 20b --cpu-layers 22

# CPU-only inference
./examples/gpt-oss-example.sh 20b --cpu-only
```

## Manual Commands

### Basic gpt-oss-20b with automatic model download:

```bash
./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048
```

### gpt-oss-120b with CPU layers (for limited VRAM):

```bash
./build/bin/llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 8
```

### CPU-only inference:

```bash
./build/bin/llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --jinja -ub 512 -b 512 --cpu-moe
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-hf ggml-org/gpt-oss-20b-GGUF` | Downloads gpt-oss-20b model from Hugging Face |
| `-hf ggml-org/gpt-oss-120b-GGUF` | Downloads gpt-oss-120b model from Hugging Face |
| `--ctx-size 0` | Use model's default context size (128k tokens) |
| `--ctx-size 32768` | Limit context to 32k tokens (saves memory) |
| `--jinja` | Use the model's built-in chat template |
| `-ub 2048 -b 2048` | Set batch sizes (adjust based on available memory) |
| `--n-cpu-moe N` | Keep N MoE layers on CPU (for limited VRAM) |
| `--cpu-moe` | Keep all MoE layers on CPU (CPU-only inference) |

## Using the Server

Once llama-server is running, you can:

1. **Web UI**: Open http://localhost:8080 in your browser
2. **API**: Use the OpenAI-compatible API at `http://localhost:8080/v1/chat/completions`
3. **Health Check**: `curl http://localhost:8080/health`

## Example API Usage

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss", 
    "messages": [
      {"role": "user", "content": "Hello! Can you help me write a Python script?"}
    ],
    "temperature": 1.0,
    "top_p": 1.0
  }'
```

## Recommended Settings

- **Temperature**: 1.0
- **Top-p**: 1.0
- **Repetition penalty**: Disabled (gpt-oss models don't need repetition penalties)

## For More Information

See the comprehensive [gpt-oss guide](../docs/gpt-oss.md) for detailed hardware configurations, troubleshooting, and advanced usage.