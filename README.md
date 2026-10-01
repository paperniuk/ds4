# ds4 for M1/M2 Macs: Qwen3.8-Flash-Next at 262K context

A fork of [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) tuned for
Apple7/Apple8 GPUs, that is M1 and M2 Macs, which do not get the Metal 4
tensor API that stock ds4 uses on M5.

It runs **Qwen3.8-Flash-Next** (the ISTA-DASLab GSQ-RCO Q2_0 GGUF) on a
**64 GB M1 Max** with the **full 262K context**, MTP speculative decoding and
vision, as a local OpenAI/Anthropic compatible server for OpenCode, Claude
Code, Pi or any other agent.

## Numbers

M1 Max, 32-core GPU, 64 GB. Greedy decoding, `--prefill-chunk 2048`.

| Context | Decode | Decode with MTP | Prefill |
|---|---|---|---|
| 4K | 35.5 tok/s | 39 to 47 tok/s | ~290 tok/s |
| 128K | 33.8 tok/s | 39 tok/s | ~270 tok/s |
| 262K | 29.7 tok/s | 36.1 tok/s | ~269 tok/s |

MTP gains depend on the text: about +30% on code, about +10% on prose.

Same machine, plain decode, before this fork:

| Engine | Short context | Long context | 262K |
|---|---|---|---|
| llama.cpp 0.5.0 | 27 tok/s | 13 tok/s at 64K | not measured |
| stock ds4 (*) | 21 to 23 tok/s | 22.7 tok/s at 128K | 17.7 tok/s |
| **this fork** | **35.5 tok/s** | **33.8 tok/s at 128K** | **29.7 tok/s** |

(*) Stock ds4 plus the n-gram sidecar patch, which it needs to load this
model. The short context figure was measured with the DS4-IQ2 weights.

Quality is unchanged: every kernel is checked against the reference path,
and `score_official` on the Alibaba fixtures gives the same average NLL and
top-1 counts before and after the kernel work (0.45044 vs 0.45045 short,
0.17410 vs 0.17410 long).

## What changed

- **Apple7 matvec kernels** for the mixed precision ISTA GGUF: Q2_0, Q5_0,
  IQ4_NL/XS, Q3_K, Q4_K, Q5_K, Q6_K and split-K BF16, plus variants that read
  the weights once for 2 or 3 tokens, used by MTP verification.
- **Indexer and attention on long context**: the vector block scorer runs on
  every Apple GPU (it was M5 only), the decode attention folds four keys per
  round, and the top-k selection gathers with coalesced simdgroup segments.
  At 128K this took decode from 23.8 to 33.8 tok/s with byte-identical output.
- **n-gram table from a sidecar**: the 26.8 GiB IQ4_NL n-gram table is read
  from the second GGUF shard on disk, one small read per token, and the
  IQ4_NL block size is fixed.
- **MTP block from another GGUF**, since the ISTA release has none.
- **Prompt anchor for agents**: Qwen3.8's recurrent layers cannot be rewound,
  so a retried or edited agent turn used to prefill the whole transcript
  again. The server now keeps a copy of the state at the last turn marker. A
  retried turn on a 31K prompt went from 109 s to 0.3 s.
- **Cold cache anchor for ChatML**: the disk KV checkpoint now ends right
  before the first user message, so a new agent session with the same system
  prompt and tools loads it instead of prefilling the shared prefix again.
- **Remainder tiles in prefill**: the expert GEMMs use 8 and 16 token tiles
  for the leftover tokens of each expert. Small prefills, which is what an
  agent sends after every tool call, gain the most: 182 to 223 tok/s at 300
  tokens, 231 to 268 at 600, with identical output.
- **Vision in agent sessions**: up to 64 images per request, older ones are
  replaced by a short note instead of failing the request.

Each change is a separate commit with a test.

## Requirements

- An M1/M2 Mac (Max or Ultra) with 64 GB of RAM. Only the M1 Max has been
  measured so far; reports from other chips are welcome.
- About 67 GB of disk for the model (both shards) and 45 GB more for the
  GGUF that provides the MTP block. A fast internal SSD, since the n-gram
  table is read from disk on every token.
- 262K context needs a higher GPU wired memory limit, reset at every reboot:

```sh
sudo sysctl iogpu.wired_limit_mb=57344
```

Without it, use `--ctx 131072` or less.

## Quick start

```sh
git clone -b m1-flash-next https://github.com/paperniuk/ds4.git
cd ds4 && make

# Model: both shards of the Q2_0 release
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --include "Q2_0/*" --local-dir ~/models/flash-next
# MTP block (optional, for --mtp)
hf download ivanfioravanti/Qwen3.8-Flash-Next-DS4-IQ2 --local-dir ~/models/flash-next/ds4-iq2
# Vision encoder (optional, for --vision)
./download_model.sh qwen38-vision

M=~/models/flash-next/Q2_0
DS4_QWEN_NGRAM_GGUF=$M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
DS4_QWEN_MTP_GGUF=~/models/flash-next/ds4-iq2/Qwen3.8-Flash-Next-IQ2XXSImatrix-Q2KDownPad768-MTP.gguf \
./ds4-server -m $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ctx 262144 --prefill-chunk 2048 --mtp \
    --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
    --host 127.0.0.1 --port 8010
```

Then point your agent at `http://127.0.0.1:8010/v1`. See
[docs/CLIENTS.md](docs/CLIENTS.md) for OpenCode and Claude Code setup.

## Tips for agents

- Pick the reasoning effort once per session. Qwen3.8 writes it at the top of
  the system prompt, so switching it mid-session prefills the whole
  conversation again.
- The server keeps one live conversation. Switching to another one saves the
  current state to the disk cache (`--kv-disk-dir`), and switching back
  restores it in about a second.

## Status

Experimental, one developer, one machine. Parts of this fork that help every
ds4 user are being proposed upstream. Everything else in ds4 (other models,
CUDA, distributed inference, SSD streaming) is inherited from upstream and
should keep working, but is not the focus here.

Model weights are under the Qwen community license; read it before any
commercial use. The code is MIT, like upstream.

---

The upstream README follows.

<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming. SSD streaming is also needed in order to run very
  large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.
