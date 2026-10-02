# ds4 Flash-Next: Qwen3.8-Flash-Next on Apple Silicon

A fork of [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar), tuned
mostly for M1/M2, that runs **Qwen3.8-Flash-Next** on Apple Silicon Macs with
64 GB of memory or more:
the **full 262K context**, MTP speculative decoding and vision, as a local
OpenAI/Anthropic compatible server for OpenCode, Claude Code, Pi or any
other agent. The same launcher also runs **DeepSeek V4 Flash** and the other
ds4 models, see [Other models](#other-models).

What it adds to stock ds4:

- **The small ISTA-DASLab GSQ-RCO files.** Their weights take 35 to 51 GiB,
  so the model fits a 64 GB Mac with room for the whole context window.
  Stock ds4 does not open these files; its own take 42 and 70 GiB.
- **One launcher.** `dstar serve` picks the context that fits the memory of
  the machine, turns on MTP and vision, and prints the one `sysctl` line it
  may need.
- **Agent sessions that do not replay.** A retried or edited turn on a 31K
  prompt takes 0.3 s instead of 109 s, and a new session with the same
  system prompt and tools starts from a checkpoint on disk.
- **Metal kernels for these quants**, written and measured on an M1 Max,
  where stock ds4 decoded at 21 to 23 tokens per second and this fork
  decodes at 35.
- **DeepSeek V4 Flash on a 64 GB Mac.** The 81 GiB Q2 file is larger than
  the memory, so `dstar serve deepseek` streams it from the SSD: 10.35
  tokens per second on an M1 Max, against 9.4 in stock ds4. See
  [the numbers](#deepseek-v4-flash-on-a-64-gb-mac).

Nearly all of it is the same code on every Apple Silicon chip. What is and
what is not specific to M1 is listed in [Which Macs gain](docs/FLASH_NEXT_FORK.md#which-macs-gain).

## Qwen3.8-Flash-Next: which quant for your Mac

Qwen3.8-Flash-Next is the main model here and what `dstar pull` downloads.
ISTA-DASLab publishes it in three sizes (quants); a larger one answers
better and is slower. `--quant` picks one, in `dstar pull` and in
`dstar serve`:

| Your Mac | Quant | `--quant` | Weights | Context it runs with |
|---|---|---|---|---|
| 64 GB | Q2_0, the fastest | `q2` (default) | 35 GiB | 262K; up to 524K |
| 64 GB | IQ3_XXS, better answers | `iq3` | 44 GiB | 131K; 262K with `--ctx 262k` |
| 96 GB or more | IQ3_S, the best file | `iq3s` | 51 GiB | 262K; up to 524K from 128 GB |

The 64 GB rows are measured on an M1 Max. The IQ3_S row comes from the
memory plan: on 64 GB that file runs, but only with a 32K context.
ISTA's LiveCodeBench scores are 81.1, 86.3 and 86.9; speeds are in
[The larger quants](docs/FLASH_NEXT_FORK.md#the-larger-quants).
`dstar doctor` prints the largest context each quant can hold on your Mac.

**On 64 GB the GPU memory limit has to be raised for the long contexts**
(above 131K with Q2_0, 262K with IQ3_XXS). macOS lets the GPU use about
three quarters of the memory by default, 48 GiB of 64. When a context does
not fit, `dstar serve` takes a smaller one and prints the line to run:

```sh
sudo sysctl iogpu.wired_limit_mb=57344    # 61440 for iq3 at 262K or q2 at 524K
```

The setting is lost at every reboot. With IQ3_XXS at 262K the server holds
56.6 GiB of the 64, so close the browser first.

Below 64 GB nothing has been tried. Qwen3.8 has no SSD streaming in ds4, so
the weights must fit in memory: Q2_0 with a 32K context needs 42 GiB.

## Flash-Next speed on an M1 Max

M1 Max, 32-core GPU, 64 GB, the Q2_0 file. Greedy decoding,
`--prefill-chunk 2048`.

| Context | Decode | Decode with MTP | Prefill |
|---|---|---|---|
| 4K | 35.5 tok/s | 39 to 47 tok/s | ~290 tok/s |
| 128K | 33.8 tok/s | 39 tok/s | ~270 tok/s |
| 262K | 29.7 tok/s | 36.1 tok/s | ~269 tok/s |

MTP gains depend on the text: about +30% on code, about +10% on prose.
IQ3_XXS runs at about three quarters of this speed and IQ3_S a little below
that.

These are the only measured numbers so far. Newer chips have faster GPUs and
run the same kernels, but nobody has timed them; if you do, the two commands
at the end of [Which Macs gain](docs/FLASH_NEXT_FORK.md#which-macs-gain) make a useful report.

## DeepSeek V4 Flash on a 64 GB Mac

The other model measured here, `dstar pull ds4f-q2` and `dstar serve deepseek`.
The Q2 file is 81 GiB, so on 64 GB ds4 keeps the dense weights and a cache
of routed experts in memory and reads the rest from the SSD as tokens need
them. Measured on an M1 Max, 32K context, greedy decoding:

| | Before the changes | This fork |
|---|---|---|
| Decode, 400 token answer | 9.4 tok/s | 10.35 tok/s |
| Prefill, 5K prompt | 94 tok/s | 101 tok/s |

Decode reaches 14.4 tokens per second while every expert it needs is in the
cache, and falls as the answer wanders: each of the 15 or so experts read
from the SSD per token costs about 2 ms. The memory plan is 48.9 GiB, of
which 36.4 GiB is the cache, 5526 of the 11008 experts.

Two things to know. DSpark speculative decoding is slower here (6.3 to 7.4
tokens per second), because every drafted token pulls more experts from the
disk, so the launcher leaves it off. And a busy performance core slows the
GPU by up to 45%: run it on a quiet machine. With 32 GB it is not usable,
3.6 tokens per second in a simulation.

## Requirements

- An Apple Silicon Mac with 64 GB of RAM or more. The fork targets M1 and M2
  (Max or Ultra); M3 and later run the same code. Only the M1 Max has been
  measured so far; reports from other chips are welcome.
- For Flash-Next, about 67 GB of disk for the model (both shards) and 1.5 GB
  for the MTP block. A fast internal SSD, since the n-gram
  table is read from disk on every token.
- For Flash-Next contexts above 131K on 64 GB, a higher GPU memory limit,
  see [above](#qwen38-flash-next-which-quant-for-your-mac).

## Quick start

Prebuilt binaries, macOS 15 or newer, nothing to compile:

```sh
curl -fsSL https://raw.githubusercontent.com/paperniuk/ds4/m1-flash-next/install.sh | bash
cd ~/ds4-flash-next
```

Or from source:

```sh
git clone https://github.com/paperniuk/ds4.git
cd ds4 && make
```

Then, from any directory:

```sh
dstar pull       # 69 GB: the model, the MTP block, the vision encoder
dstar serve      # server on http://127.0.0.1:8010/v1
dstar opencode   # provider block for OpenCode
```

The installer links the launcher into `~/.local/bin`. After a source build
run `./dstar link` once to do the same, or keep calling it as `./dstar`. If
`~/.local/bin` is not on your `PATH`, `dstar link` prints the line to add.

`dstar serve` picks the context for you: the full 262K when the memory of
the machine and the GPU memory limit allow it, a smaller one otherwise, and
it prints the `sudo sysctl` line that unlocks the larger one. MTP and vision are on when their files are present.

| Command | What it does |
|---|---|
| `dstar pull` | download what is missing into `~/models/flash-next`, resumable; `--quant q2/iq3/iq3s` picks the quant, `--no-vision` skips the encoder |
| `dstar serve` | start the server; `--ctx 131k/262k/400k/524k`, `--quant q2/iq3/iq3s`, `--port N`, `--lan`, `--no-mtp`, `--no-vision`, `--power N` |
| `dstar serve --dry-run` | print the `ds4-server` command and environment instead of running it |
| `dstar chat` | talk to the model in the terminal |
| `dstar doctor` | check the machine, the files, the memory limit and which contexts fit |
| `dstar opencode` | print the provider block for `~/.config/opencode/opencode.json` |
| `dstar link` | link the launcher into `~/.local/bin`, so `dstar` works from any directory |
| `dstar models` | list the other models ds4 runs |
| `dstar pull NAME` | download one of them into `./gguf` |
| `dstar serve NAME` | serve another model by a word from its file name (`dstar serve deepseek`); also `dstar chat NAME` |

### Other models

ds4 is not a Flash-Next engine, and neither is the launcher. It runs
DeepSeek V4 and V4.1 Flash, GLM 5.2 and 5.3 and the upstream Qwen3.8 files
too:

```sh
dstar models                  # names and sizes
dstar pull ds4f-q2            # DeepSeek V4 Flash Q2, 81 GiB
dstar serve deepseek          # or a path to the GGUF
dstar chat deepseek
```

The name is looked up in `./gguf` and next to `~/models/flash-next`.
Without a name, `dstar serve` and `dstar chat` run Flash-Next when it is on
disk and otherwise the one other model that is; `DSTAR_MODEL=deepseek`
makes another model the default. `dstar doctor` lists what it found.

Only Flash-Next gets its context sized to the machine. For the others the
context defaults to 32768 (`--ctx N` changes it) and is not checked against
the memory. A model larger than the RAM is started with `--ssd-streaming`.
Everything after `--` goes to `ds4-server` unchanged. See
[docs/MODELS.md](docs/MODELS.md) for what fits where.

`DSTAR_MODELS` changes the Flash-Next directory, `DSTAR_QUANT` the default
quant, `PORT` and `HOST` the address.
See [docs/CLIENTS.md](docs/CLIENTS.md) for Claude Code and other clients;
use port 8010 and the context the server was started with.

`--power N` keeps the GPU busy N percent of the time, for a cooler and
quieter Mac: the engine sleeps after every decoded token and every prefill
chunk. Speed drops a little more than in proportion, `--power 50` gives 16
tokens per second where the full speed is 35, and the output does not
change. In `dstar chat` the `/power N` command changes it on the fly.

After a `git pull`, run `make` and restart the server. There is nothing to
switch on: the speedups are the default path.

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

## More

- [The fork in detail](docs/FLASH_NEXT_FORK.md): contexts beyond 262K, what changed, which Macs gain, the larger quants, running `ds4-server` by hand, sampling and the switches for measuring.
- [The upstream README](https://github.com/antirez/ds4#readme): what DwarfStar is, its other models and hardware.
- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.
