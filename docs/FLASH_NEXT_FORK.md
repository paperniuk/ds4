# The fork in detail

What this fork of ds4 changes, which Macs gain from it, and the options of
the `dstar` launcher that the [README](../README.md) leaves out.

## Beyond 262K

The model was trained on a 262,144 token window, but the same build runs
longer ones. 524K is the ceiling on 64 GB; one million tokens would need
about 72 GiB.

| Context | GPU memory plan | Decode | Prefill | Needles found |
|---|---|---|---|---|
| 262K | 45.8 GiB | 31.5 tok/s | 271 tok/s | 20 of 20 |
| 400K | 50.4 GiB | 29.2 tok/s | 258 tok/s | 20 of 20 |
| 524K | 54.5 GiB | 27.5 tok/s | 248 tok/s | 18 of 20 |

Decode and prefill are the speed at that depth of one pass over a 524K token
document (`ds4-bench --teacher-forced-decode`, no MTP). The needles are 20
one-line facts spread over the document, each asked for by name after a
single prefill, with YaRN factor 2 (see `speed-bench/long_context_probe.c`).
MTP and vision add about 2 GiB.

Prediction quality does not drop past the trained window: the tokens of the
second half of that document score an average NLL of 0.265 at depths 262K to
524K and 0.276 when the same text is fed from position zero.

Set `DS4_QWEN4_YARN_FACTOR=2` for a context above 262144 and raise the wired
limit to 61440 for 524K. YaRN changes nothing below 262K and is slightly
better above it (20 against 19 needles at 400K, 18 against 17 at 524K).
KV checkpoints record the factor; keep a separate `--kv-disk-dir` for each.
With prompts this long, pass `--kv-cache-continued-interval-tokens 32768`:
the default writes a multi-gigabyte snapshot every 10K tokens of prefill.

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
- **MTP block from another GGUF**, since the ISTA release has none. A
  [1.5 GB file](https://huggingface.co/paperniuk/Qwen3.8-Flash-Next-MTP-block)
  with only that block is enough; the `mtp-*-Q8_0.gguf` from ggml-org works
  as well.
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
- **Smaller draft head for MTP**: `DS4_QWEN4_MTP_DRAFT_VOCAB` points at a
  list of token ids and the drafter scores only those rows of the output
  matrix. `gguf-tools/qwen4_mtp_draft_vocab.py` writes a Latin, code and
  Cyrillic list of 116K ids out of 248K. The draft step drops from 3.3 to
  2.7 ms with the same acceptance and the same output. A draft is only 8% of
  an MTP cycle, so this is worth about 1% in tokens per second, which is
  inside the run to run noise. It is off by default.
- **Vision in agent sessions**: up to 64 images per request, older ones are
  replaced by a short note instead of failing the request.
- **n-gram read behind the first layer**: the sixteen table rows of a token
  are uncached disk reads, 0.45 ms during which the GPU had nothing to do.
  Only layer 1 needs them, so layer 0 is submitted first and the read runs
  while it executes. MTP on code went from 46.8 to 47.7 tok/s, MTP on Russian
  text from 43.5 to 44.2, plain decode from 36.5 to 36.6, with byte-identical
  output.
- **Faster sampler**: with the server defaults (temperature above 0, top-p 1,
  min-p 0.05) picking a token took 1.34 ms of host time on the 248K
  vocabulary, again with the GPU idle. It now takes 0.13 ms and returns the
  same token for the same seed. That is 27.9 to 26.9 ms per token in plain
  decode and 40.1 to 39.1 ms per two-token MTP cycle. Greedy decoding does
  not use the sampler and does not change.
- **Faster prefill of the expert layers**: three changes to the prompt
  tiles, measured on a 16K token prompt. The Q6_K decoder reads four weights
  at a time instead of branching per weight (276 to 291 tok/s). The gate/up
  tile reads the next expert weights before the multiply, so the wait for
  cold memory overlaps with it (291 to 303). And the Q2_0 expert tiles keep
  the weights and accumulators in registers, the simdgroup matrix layout of
  the Splash M1 port, with fp32 activations staged once per threadgroup (303
  to 327). The first two leave the output byte for byte the same; the last
  one is more accurate than the tile it replaces and `DS4_QWEN4_MOE_REG=0`
  turns it off.

Each change is a separate commit with a test.

## Which Macs gain

The fork was written and measured on one M1 Max. The code falls into three
groups, and only the last one looks at the chip.

**Any Apple Silicon Mac, nothing to switch on.** These are not chosen by
chip name, so an M3, M4 or M5 runs them too:

- Loading the ISTA-DASLab GGUFs at all: the Q2_0, IQ3_XXS and IQ3_S files, the
  n-gram table from the second shard, the MTP block from another file.
  Stock ds4 does not open these files.
- The decode kernels for the mixed precision weights, and the long context
  work in the indexer and the attention. The vector block scorer was M5
  only upstream and is now the path on every chip.
- Everything on the host side: the prompt anchor, the cold cache anchor,
  the n-gram read behind the first layer, the sampler, images in agent
  sessions, contexts beyond 262K with YaRN, `--power` for Qwen3.8.
- The Q6_K decoder and the read ahead in the expert prompt tile.
- For DeepSeek V4 with `--ssd-streaming`: the early commit of the routed
  experts and the single simdgroup rows of the narrow Q8_0 projection.

**On by default on M1 only, one variable elsewhere.** These were kept behind
a chip name because a tile that wins on one GPU can lose on another, and
nothing but the M1 Max was measured:

| Change | Default | To try it on another chip |
|---|---|---|
| Remainder tiles for Q2_0 experts in prefill | M1 | `DS4_QWEN4_MOE_TAILS=1` (`dstar serve` sets it on every Mac) |
| Register tiles for Q2_0 experts in prefill | M1 | `DS4_QWEN4_MOE_REG=1` |
| Packed group rows in the DeepSeek V4 attention output | M1 | `DS4_METAL_ATTN_OUT_DENSE=1` |

An M2 reports itself as M2, so it starts with the last two off. It has the
same GPU family as the M1 and they are expected to help there; set the
variables and compare.

**Where the gain is smaller or unknown.** The numbers on this page are M1 Max
numbers. On M5 stock ds4 already uses Metal 4 tensor kernels for part of the
work, and upstream has its own tuning for M3, so the distance to stock ds4
there is not known. On a Mac with more memory the context table does not
apply as written: 262K needs no wired limit change on 96 GB or more.

If you run it on anything else, the two commands below are enough for a
useful report: the first prints the machine and what fits, the second a
prefill and decode speed.

```sh
dstar doctor
dstar chat -- -p "Write a C function that reverses a linked list." -n 300 --temp 0
```

## The larger quants

```sh
dstar pull --quant iq3     # IQ3_XXS: 76 GB, or 47 GB next to another quant
dstar serve --quant iq3

dstar pull --quant iq3s    # IQ3_S: 83 GB, or 55 GB next to another quant
dstar serve --quant iq3s
```

`dstar pull --quant` downloads that quant only, there is no need to fetch
Q2_0 first. The n-gram shard, the MTP block and the vision encoder are the
same files for every quant, so a second quant downloads only its own
weights. Without `--quant`, `dstar serve` takes the quant that is on disk,
Q2_0 when there are several.

| `--quant` | ISTA file | Weights | ISTA task average | LiveCodeBench | Plain | With MTP | Default context on 64 GB |
|---|---|---|---|---|---|---|---|
| `q2` | Q2_0 | 35.0 GiB | 89.07 | 81.14 | 35 tok/s | 39 to 47 | 262K |
| `iq3` | IQ3_XXS | 43.8 GiB | 92.57 | 86.29 | 29 tok/s | 35 | 131K |
| `iq3s` | IQ3_S | 51.0 GiB | 93.26 | 86.86 | 26 tok/s | 32 to 34 | 32K |

The scores are ISTA's own, from their model card. The speeds are M1 Max
numbers at a short context, on one code prompt for IQ3_S.

These files keep the experts in IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS and
many dense projections in IQ3_S, quants that stock ds4 does not load. This
fork dequantizes them on the CPU bit for bit as llama.cpp does and has Metal
kernels for them, checked against that reference by `make test-quant-types`
and `make test-qwen4-kernels`.

**IQ3_XXS on 64 GB.** `dstar serve` starts it with a 131K context, which
plans 52 GiB of GPU memory. `--ctx 262k` works and plans 56.6 GiB, but leaves
the rest of the system about 5 GB, so close the browser first. Longer
contexts do not fit.

**IQ3_S is for 96 GB and more.** With MTP and vision it plans about 56 GiB
at 32K and 60 GiB at 131K. On a 64 GB Mac it therefore runs with a 32K
context and little room for anything else; it was loaded and measured that
way here. On 96 GB or more `dstar serve` picks the full 262K window, and the
default GPU memory limit is already high enough. That choice comes from the
memory plan, it has not been run on such a machine.

`dstar doctor` lists the largest context every quant can hold on the
machine. `dstar doctor --quant NAME` and `dstar chat --quant NAME` take
the same option, and `DSTAR_QUANT=iq3` makes one the default.

Above that, upstream's Q4 file (`dstar pull qwen38-q4k`, 165 GiB on disk,
70 GiB of weights in memory) runs through the launcher as one of the other
models: no automatic context, and the kernels of this fork for the ISTA
quants do not take part.

## By hand

`dstar serve` runs this, with the paths filled in:

```sh
M=~/models/flash-next/Q2_0
DS4_QWEN4_MOE_TAILS=1 \
DS4_QWEN_NGRAM_GGUF=$M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
DS4_QWEN_MTP_GGUF=~/models/flash-next/Qwen3.8-Flash-Next-MTP-block.gguf \
./ds4-server -m $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ctx 262144 --prefill-chunk 2048 --mtp \
    --vision ~/models/flash-next/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
    --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 12288 \
    --host 127.0.0.1 --port 8010
```

Run it from the directory that holds `metal/`: the shaders are compiled from
there at start.

## Longer contexts

`dstar serve --ctx 400k` or `--ctx 524k` does all of the below. By hand:
pick the context, raise the wired limit to match, and add YaRN and a
separate checkpoint directory above 262144:

| `--ctx` | `iogpu.wired_limit_mb` | Extra settings |
|---|---|---|
| 131072 or less | default | none |
| 262144 | 57344 | none |
| 400000 | 57344 | YaRN 2, own KV directory |
| 524288 | 61440 | YaRN 2, own KV directory |

```sh
sudo sysctl iogpu.wired_limit_mb=61440

M=~/models/flash-next/Q2_0
DS4_QWEN4_YARN_FACTOR=2 \
DS4_QWEN_NGRAM_GGUF=$M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
DS4_QWEN_MTP_GGUF=~/models/flash-next/mtp/Qwen3.8-Flash-Next-MTP-block.gguf \
./ds4-server -m $M/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ctx 524288 --prefill-chunk 2048 --mtp \
    --kv-disk-dir ~/.ds4/server-kv-yarn2 --kv-disk-space-mb 40960 \
    --kv-cache-continued-interval-tokens 32768 \
    --host 127.0.0.1 --port 8010
```

Close other heavy applications first: at 524K the server holds about 56 GiB
of the 64. Measured at `--ctx 400000`: a cold 346K token document takes
about 22 minutes to read, a follow-up question in the same chat starts
answering in about a second, and a new chat on the same document in about a
minute and a half from the checkpoint.

## Sampling

Clients that send no sampling settings get temperature 1, top-p 1 and min-p
0.05, which is the fast sampler path. A request with top-p below 1 takes
about 0.9 ms of host time per token instead of 0.13. For reproducible output
send `"temperature": 0` or a `seed`.

## Switches for measuring

| Variable | Effect |
|---|---|
| `DS4_QWEN4_NGRAM_FIRST=1` | read the n-gram rows before any GPU work, as before |
| `DS4_QWEN4_MOE_REG=0` | staged expert tiles in prefill instead of the register ones |
| `DS4_METAL_FORCE_METAL4=1` | Metal 4 tensor kernels on chips older than M5 |
| `DS4_QWEN4_TIMING=1` | print host and GPU time per forward call every 50 calls |
| `DS4_QWEN4_SPEC_TIMING=1` | print the draft and verify time of the MTP cycle |
| `DS4_METAL_ENCODER_TIMELINE=file` | write the GPU time of every dispatch (slows the run) |
| `DS4_QWEN4_MTP_DRAFT_VOCAB=file` | draft with a reduced vocabulary head |

Compare variants in alternating runs: on a Mac in use, tokens per second of
a single run moves by 1 to 2 percent.
