# Running gpt-oss Models with llama.cpp

> [!NOTE]
> This guide is a live document. Feedback and benchmark numbers are welcome - the guide will be updated accordingly.

---

## Overview

This is a detailed guide for running the new [gpt-oss](https://github.com/ggml-org/llama.cpp/discussions/15095) models locally with the best performance using `llama.cpp`. The guide covers a very wide range of hardware configurations. The `gpt-oss` models are very lightweight so you can run them efficiently in surprisingly low-end configurations.

## Prerequisites

### `llama.cpp` binaries for your system

Make sure you are running the latest release of `llama.cpp`: https://github.com/ggml-org/llama.cpp/releases

### `gpt-oss` model data (optional)

The commands used below in the guide will automatically download the model data and store it locally on your device. So this step is completely optional and provided for completeness.

The original models provided by OpenAI are here:

- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/openai/gpt-oss-120b

First, you need to manually convert them to [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) format. For convenience, we host pre-converted models here in [ggml-org](https://huggingface.co/ggml-org).

Pre-converted GGUF models:

- https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
- https://huggingface.co/ggml-org/gpt-oss-120b-GGUF

> [!TIP]
> Running the commands below will automatically download the latest version of the model and store it locally on your device for later usage. A WebUI chat and an OAI-compatible API will become available on localhost.

<table>
<tbody><tr>
<td>
<img src="https://github.com/user-attachments/assets/66984701-c49d-460d-92a0-909b71f810fb">

</td>
<td>
<img width="2546" height="1296" alt="image" src="https://github.com/user-attachments/assets/d1c04564-10d9-430d-982e-83ad06d56b5b">

<div align="center"><i>Using <b>llama-server</b> with <a href="https://github.com/charmbracelet/crush">crush</a> coding agent (<b>gpt-oss-20b</b>)</i></div>
</td>
</tr>
</tbody></table>

## Minimum requirements

Here are some hard memory requirements for the 2 models. These numbers could vary a little bit by adjusting the CLI arguments, but should give a good reference point.

| Model            | Model data (GB) | Compute buffers (GB) | KV cache per 8,192 tokens (GB) | Total @ 8,192 tokens (GB) | Total @ 32,768 tokens (GB) | Total @ 131,072 tokens (GB) |
|------------------|-----------------|----------------------|--------------------------------|---------------------------|----------------------------|-----------------------------|
| gpt‑oss 20B      | 12.0            | 2.7                  | 0.2                            | **14.9**                  | **15.5**                   | **17.9**                    |
| gpt‑oss 120B     | 61.0            | 2.7                  | 0.3                            | **64.0**                  | **64.9**                   | **68.5**                    |

> [!NOTE]
> It is not necessary to fit the entire model in VRAM to get good performance. Offloading just the attention tensors and the KV cache in VRAM and keeping the rest of the model in the CPU RAM can provide decent performance as well. This is taken into account in the rest of the guide.

## Relevant CLI arguments

Using the correct CLI arguments in your commands is crucial for getting the best performance for your hardware. Here is a summary of the important flags and their meaning:

| Argument | Purpose |
| ---      | ---     |
| `-hf`   | Specify the Hugging Face model ID to use. The model will be downloaded using `curl` from the respective model repository |
| `-c`   | Specify the context size to use. More context requires more memory. Both `gpt-oss` models have a maximum context of 128k tokens. Use `-c 0` to set to the model's default |
| `-ub N -b N` | Specify the max batch size `N` during processing. Larger size increases the size of compute buffers, but can improve the performance in some cases |
| `-fa` | Enable Flash Attention kernels. This improves the performance on backends that support the operator |
| `--n-cpu-moe N` | Number of MoE layers `N` to keep on the CPU. This is used in hardware configs that cannot fit the models fully on the GPU. The specific value depends on your memory resources and finding the optimal value requires some experimentation |
| `--jinja` | Tell `llama.cpp` to use the Jinja chat-template embedded in the GGUF model file |

## Hardware-specific configurations

### NVIDIA

#### ✅ RTX 4090 / RTX 5090 (24GB / 32GB VRAM)

Perfect for running both models with excellent performance:

```bash
# gpt-oss-20b, full context
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048

# gpt-oss-120b, 32k context, 8 layers on the CPU
llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 8
```

#### ✅ RTX 4080 / RTX 4080 Super (16GB VRAM)

Good for both models with some CPU layers:

```bash
# gpt-oss-20b, full context, 8 layers on the CPU
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048 --n-cpu-moe 8

# gpt-oss-120b, 32k context, 20 layers on the CPU
llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 20
```

#### ✅ Devices with less than 16GB VRAM

For this config, it is recommended to tell `llama.cpp` to run the entire model on the GPU while keeping enough layers on the CPU. Here is a specific example with an **RTX 2060 8GB** machine:

```bash
# gpt-oss-20b, full context, 22 layers on the CPU
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048 --n-cpu-moe 22

# gpt-oss-20b, 32k context, 16 layers on the CPU (faster, but has less total context)
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 16
```

Note that even with just 8GB of VRAM, we can adjust the CPU layers so that we can run the large 120B model too:

```bash
# gpt-oss-120b, 32k context, 35 layers on the CPU
llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 35
```

> [!TIP]
> For more information about how to adjust the CPU layers, see the "Tips" section at the end of this guide.

### AMD

> [!NOTE]
> If you have AMD hardware, please provide feedback about running the `gpt-oss` models on it and the performance that you observe. See the sections above for what kind of commands to try and try to adjust respectively.

With AMD devices, you can use either the ROCm or the Vulkan backends. Depending on your specific hardware, the results can vary.

#### ✅ RX 7900 XT (20GB VRAM) using ROCm backend

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048
```

<details>

<summary> 🟢 Benchmarks for `gpt-oss-20b`</summary>

```bash
llama-bench -m gpt-oss-20b-mxfp4.gguf -t 1 -fa 1 -b 2048 -ub 2048 -p 2048,8192,16384,32768
```

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XT, gfx1100 (0x1100), VMM: no, Wave Size: 32

| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --------------: | -------------------: |
| gpt-oss 20B BF16                |  12.83 GiB |    20.91 B | ROCm,RPC   |  99 |       1 |    4096 |     2048 |  1 |          pp2048 |      4251.56 ± 21.68 |
| gpt-oss 20B BF16                |  12.83 GiB |    20.91 B | ROCm,RPC   |  99 |       1 |    4096 |     2048 |  1 |          pp8192 |      3567.45 ± 11.84 |
| gpt-oss 20B BF16                |  12.83 GiB |    20.91 B | ROCm,RPC   |  99 |       1 |    4096 |     2048 |  1 |         pp16384 |      2948.39 ± 10.34 |
| gpt-oss 20B BF16                |  12.83 GiB |    20.91 B | ROCm,RPC   |  99 |       1 |    4096 |     2048 |  1 |         pp32768 |      2099.25 ± 13.17 |
| gpt-oss 20B BF16                |  12.83 GiB |    20.91 B | ROCm,RPC   |  99 |       1 |    4096 |     2048 |  1 |           tg128 |        101.92 ± 0.27 |

build: 3007baf2 (6194)

</details>

More information: https://github.com/ggml-org/llama.cpp/discussions/15396#discussioncomment-14143427

#### ✅ Few more low-end configurations

- [AMD Radeon 890M using Vulkan](https://github.com/ggml-org/llama.cpp/discussions/15396#discussioncomment-14157403)
- [AMD Radeon FirePro W8100 + AMD Radeon RX 470 using Vulkan](https://github.com/ggml-org/llama.cpp/discussions/15396#discussioncomment-14158676)

### Apple Silicon (Metal)

#### ✅ M1/M2/M3/M4 series with 16GB+ unified memory

Apple Silicon devices with Metal backend work excellently:

```bash
# gpt-oss-20b, full context
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048

# gpt-oss-120b, 32k context, adjust layers based on available unified memory
llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 32768 --jinja -ub 2048 -b 2048 --n-cpu-moe 10
```

### CPU-only configurations

For CPU-only inference, the performance will be slower but still usable:

```bash
# gpt-oss-20b, CPU-only
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --jinja -ub 512 -b 512 --cpu-moe

# gpt-oss-120b, CPU-only with reduced context
llama-server -hf ggml-org/gpt-oss-120b-GGUF --ctx-size 8192 --jinja -ub 512 -b 512 --cpu-moe
```

## Tips

<details>
<summary>Determining the optimal number of layers to keep on the CPU</summary>

Good general advice for most MoE models would be to offload the entire model, and use `--n-cpu-moe` to keep as many MoE layers as necessary on the CPU. The minimum amount of VRAM to do this with the 120B model is about 8GB, below that you will need to start reducing context size and the number of layers offloaded. You can get for example about 30 t/s at zero context on a 5090 with `--n-cpu-moe 21`.

Caveat: on Windows it is possible to allocate more VRAM than available, and the result will be slow swapping to RAM and very bad performance. Just because the model loads without errors, it doesn't mean you have enough VRAM for the settings that you are using. A good way to avoid this is to look at the "GPU Memory" in Task Manager and check that it does not exceed the GPU VRAM.

Example on 5090 (32GB):
good, `--n-cpu-moe 21`, GPU Memory < 32:
<img width="396" height="258" alt="image" src="https://github.com/user-attachments/assets/2ebef66d-613c-4bc5-9105-41e3b187912f" />

bad, `--n-cpu-moe 20`, GPU Memory > 32:
<img width="390" height="249" alt="image" src="https://github.com/user-attachments/assets/bdc2ad35-72ca-429a-ab71-5c03296cd7dd" />

</details>

<details>
<summary>Using `gpt-oss` + `llama.cpp` with coding agents (such as Claude Code, crush, etc.)</summary>

- Setup the coding agent of your choice to look for a localhost OAI endpoint (see https://github.com/ggml-org/llama.cpp/discussions/14758)
- Start `llama-server` like this:

  ```bash
  # adjust this command for your hardware
  llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048

  # some agents such as Claude Code can benefit from multiple parallel server slots
  # note: currently this requires extra memory!
  llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 524288 -np 4 --jinja -ub 2048 -b 2048
  ```
- Sample usage with `crush`: https://github.com/ggml-org/llama.cpp/discussions/15396#discussioncomment-14144716
- Some agents such as Cline and Roo Code do not support native tool calls. A workaround is to use a custom grammar: https://github.com/ggml-org/llama.cpp/discussions/15396#discussioncomment-14145537

</details>

<details>
<summary>Configure the default sampling and reasoning settings</summary>

When starting a `llama-server` command, you can change the default sampling and reasoning settings like so:

```bash
# use recommended gpt-oss sampling params
llama-server ... --temp 1.0 --top-p 1.0

# set default reasoning effort
llama-server ... --chat-template-kwargs '{"reasoning_effort": "high"}'
```

Note that these are just the default settings and they could be overridden by the client connecting to the `llama-server`.

</details>

## Frequently asked questions

### Do I need to quantize gpt-oss models?

`gpt-oss` models are natively "quantized". I.e. they are trained in the MXFP4 format which is roughly equivalent to `ggml`'s `Q4_0`. The main difference with `Q4_0` is that the MXFP4 models get to keep their full quality. This means that no quantization in the usual sense is necessary.

### What are the recommended sampling settings?

`temperature=1.0 and top_p=1.0`.

**Do not use repetition penalties!** Some clients tend to enable repetition penalties by default - make sure to disable those.

<img src="https://github.com/user-attachments/assets/e7873fe9-bd34-4967-808a-adba3c8189d0">

### Can I change the chat template?

`ggml-org/gpt-oss` models have a built-in chat template that is used by default. The only reasons to ever want to change the chat template manually are:

- If there is a bug in the built-in chat template
- If you have a very specific use case and you know very well what you are doing

## Known issues

Some rough edges in the implementation are still being polished. Here is a list of issues to keep track of:

- https://github.com/ggml-org/llama.cpp/issues/15274
- https://github.com/ggml-org/llama.cpp/discussions/15362

## Building from source

If you need to build `llama.cpp` from source to run `gpt-oss` models, follow the standard build instructions:

```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build with CMake
cmake -B build
cmake --build build --config Release -j $(nproc)

# The binaries will be in build/bin/
./build/bin/llama-server --help
```

For specific backend support (CUDA, Metal, etc.), refer to the [build documentation](build.md).

## Performance optimization

### Memory optimization

- Use `--ctx-size` to limit context length if you don't need the full 128k tokens
- Adjust batch sizes (`-b`, `-ub`) based on available memory
- Use `--n-cpu-moe` to balance GPU/CPU memory usage

### Compute optimization

- Enable Flash Attention with `-fa` for supported backends
- Use appropriate thread counts (`-t`, `-tb`) for your CPU
- Consider CPU affinity settings for multi-socket systems

### Monitoring

Use the monitoring endpoints provided by `llama-server`:

- Health check: `http://localhost:8080/health`
- Metrics: `http://localhost:8080/metrics` 
- Model info: `http://localhost:8080/v1/models`

## Contributing

This guide is a living document. If you have:
- Performance benchmarks on different hardware
- Hardware configurations not covered here
- Optimizations or tips

Please contribute to the discussion at: https://github.com/ggml-org/llama.cpp/discussions/15396