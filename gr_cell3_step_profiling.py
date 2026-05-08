#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================
# GR_CELL3 — STEP-BY-STEP PROFILING (bar + pie)
# GNU Radio LoRa Transmitter
#
# Equivalent to cell3_step_profiling.py (Sionna).
# Measures each GNU Radio block in isolation using a minimal
# flowgraph: vector_source → [block under test] → null_sink.
#
# The vector_source feeds pre-computed data matching the expected
# input type and size of each block, so each measurement captures
# only the block's own processing latency — not the file I/O or
# scheduler startup overhead.
#
# IMPORTANT DESIGN NOTES:
# ──────────────────────────────────────────────────────────────────
# 1. GNU Radio blocks are not callable functions — they are streaming
#    processors. To time a single block we must:
#      a) Build a minimal flowgraph around it
#      b) Start the flowgraph, wait for completion, and stop it
#    This means each "step" measurement includes GR scheduler startup
#    overhead (~few ms). This is analogous to the JIT warm-up in the
#    Sionna/TF measurements; we apply the same warm-up + multi-run
#    averaging strategy to make the numbers comparable.
#
# 2. The whitening block in gr-lora_sdr expects a comma-separated
#    ASCII hex string from a file_source. For the isolated step
#    measurement we therefore keep file_source → whitening only,
#    with a vector_sink to drain output.
#
# 3. For all subsequent blocks (header, add_crc, …) we generate a
#    vector_source with the expected byte/bit sequence and connect
#    it directly to the block under test.
#
# 4. The modulate block produces gr_complex samples; null_sink is
#    used as output sink for all blocks that produce stream output.
# ===================================================================

import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────────────────────────────────────────────────
# PARAMETERS
# ───────────────────────────────────────────────────────────────────

BASE_DIR = Path("C:/Adrion/Doutorado/Sionna - LoRa/Simulador LoRa Phy/gr-lora_sdr-master")
DATA_DIR = BASE_DIR / "data" / "GRC_default"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_RUNS      = 20
NUM_WARMUP    = 10
PAYLOAD_BYTES = 10
SF            = 7
BW            = 125000
CR            = 2
SAMP_RATE     = 500000
PREAMB_LEN    = 8

print("=" * 70)
print("GR_CELL3 — STEP-BY-STEP PROFILING (bar + pie)")
print(f"  SF={SF}  BW={BW/1e3:.0f} kHz  CR=4/{4+CR}  payload={PAYLOAD_BYTES} bytes")
print("=" * 70)

# ───────────────────────────────────────────────────────────────────
# GNU RADIO CHECK
# ───────────────────────────────────────────────────────────────────

try:
    from gnuradio import gr, blocks
    import gnuradio.lora_sdr as lora_sdr
    import pmt
    print("  GNU Radio detected\n")
except ImportError:
    print("  ERROR: GNU Radio not found. Exiting.")
    raise SystemExit(1)

# ───────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────

def _run_fg(tb, timeout: float = 30.0) -> float | None:
    """
    Starts a top_block, waits for completion, returns elapsed ms.
    Returns None on timeout or error.
    """
    finished = threading.Event()
    error    = [None]

    def _worker():
        try:
            tb.start()
            tb.wait()
        except Exception as e:
            error[0] = e
        finally:
            finished.set()

    t0 = time.perf_counter()
    threading.Thread(target=_worker, daemon=True).start()

    if finished.wait(timeout=timeout):
        elapsed = (time.perf_counter() - t0) * 1000.0
        if error[0]:
            return None
        return elapsed
    else:
        try:
            tb.stop()
        except Exception:
            pass
        return None


def measure_step_fg(fg_factory, num_runs=NUM_RUNS, num_warmup=NUM_WARMUP) -> dict:
    """
    Measures execution time of a flowgraph built by fg_factory().

    fg_factory is called once per run to get a fresh top_block
    (GR blocks are stateful and cannot be restarted after stop()).

    Returns dict: mean_ms, std_ms, min_ms, max_ms
    """
    # Warm-up runs
    for _ in range(num_warmup):
        _run_fg(fg_factory())

    # Timed runs
    times_ms = []
    for _ in range(num_runs):
        t = _run_fg(fg_factory())
        if t is not None:
            times_ms.append(t)

    if not times_ms:
        return {'mean_ms': float('nan'), 'std_ms': 0.0,
                'min_ms': float('nan'), 'max_ms': float('nan')}

    return {
        'mean_ms': float(np.mean(times_ms)),
        'std_ms':  float(np.std(times_ms)),
        'min_ms':  float(np.min(times_ms)),
        'max_ms':  float(np.max(times_ms)),
    }


# ───────────────────────────────────────────────────────────────────
# PRE-COMPUTE DATA FOR EACH STAGE
# ───────────────────────────────────────────────────────────────────
# We run the full pipeline once to obtain the exact byte sequences
# expected at each block input. This ensures each isolated flowgraph
# feeds realistic data of the correct length and type.

# Random payload (PAYLOAD_BYTES bytes, as ASCII hex + trailing comma)
_raw_payload = np.random.randint(0, 256, size=PAYLOAD_BYTES, dtype=np.uint8)
_hex_payload = ''.join(f'{b:02x}' for b in _raw_payload) + ','

# Write a 1-packet input file
_input_file = DATA_DIR / "step_profile_1pkt.txt"
_input_file.write_text(_hex_payload)

# ───────────────────────────────────────────────────────────────────
# STEP FLOWGRAPH FACTORIES — CHAIN-SUBTRACTION STRATEGY
# ───────────────────────────────────────────────────────────────────
#
# Several gr-lora_sdr blocks consume stream tags produced by upstream
# blocks and crash with "wrong_type" if those tags are absent.
# A plain vector_source does not emit tags, so we cannot isolate
# these blocks directly.
#
# Strategy: build cumulative chains from file_source and derive each
# block's isolated time by subtracting the previous chain's time.
#
#   Chain A  : file_source → whitening
#   Chain B  : file_source → whitening → header
#   Chain B2 : file_source → whitening → header → add_crc → hamming_enc
#   Chain C  : file_source → whitening → header → add_crc → hamming_enc → interleaver
#   Chain D  : Chain C → gray_demap
#   Chain E  : Chain D → modulate
#
#   t_whitening   = t(A)
#   t_header      = t(B)  − t(A)
#   t_add_crc     = measured standalone (no tag dependency)
#   t_hamming     = measured standalone (no tag dependency)
#   t_interleaver = t(C)  − t(B2)   ← B2 is the chain just before interleaver
#   t_gray_demap  = t(D)  − t(C)
#   t_modulate    = t(E)  − t(D)
#
# NOTE: add_crc and hamming_enc are measured standalone because they do
# not consume stream tags and can be isolated directly. They must NOT be
# used in the interleaver subtraction — doing so double-counts the
# per-run GR scheduler overhead and zeroes out the result.
#
# std is propagated in quadrature for each subtraction.
#
# I/O item sizes:
#   whitening / header / add_crc / hamming_enc → sizeof_char in+out
#   interleaver  → sizeof_char in / sizeof_int out
#   gray_demap   → sizeof_int  in / sizeof_int out
#   modulate     → sizeof_int  in / sizeof_gr_complex out

def _sub(m_long, m_short):
    """Subtract two measurement dicts; propagate std in quadrature."""
    return {
        'mean_ms': max(m_long['mean_ms'] - m_short['mean_ms'], 0.0),
        'std_ms':  float(np.sqrt(m_long['std_ms']**2 + m_short['std_ms']**2)),
        'min_ms':  float('nan'),
        'max_ms':  float('nan'),
    }


# ── Chain A: file_source → whitening ────────────────────────────
def fg_chain_A():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.sink = blocks.null_sink(gr.sizeof_char)
            self.connect(self.src, self.wh, self.sink)
    return FG()


# ── Chain B: … → header ─────────────────────────────────────────
def fg_chain_B():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.hdr  = lora_sdr.header(False, True, CR)
            self.sink = blocks.null_sink(gr.sizeof_char)
            self.connect(self.src, self.wh, self.hdr, self.sink)
    return FG()


# ── Chain B2: … → add_crc → hamming_enc ────────────────────────
def fg_chain_B2():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.hdr  = lora_sdr.header(False, True, CR)
            self.crc  = lora_sdr.add_crc(True)
            self.ham  = lora_sdr.hamming_enc(CR, SF)
            self.sink = blocks.null_sink(gr.sizeof_char)
            self.connect(self.src, self.wh, self.hdr, self.crc,
                         self.ham, self.sink)
    return FG()


# ── Chain C: … → add_crc → hamming_enc → interleaver ────────────
def fg_chain_C():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.hdr  = lora_sdr.header(False, True, CR)
            self.crc  = lora_sdr.add_crc(True)
            self.ham  = lora_sdr.hamming_enc(CR, SF)
            self.ilv  = lora_sdr.interleaver(CR, SF, False, BW)
            self.sink = blocks.null_sink(gr.sizeof_int)   # interleaver out = int
            self.connect(self.src, self.wh, self.hdr, self.crc,
                         self.ham, self.ilv, self.sink)
    return FG()


# ── Chain D: … → gray_demap ─────────────────────────────────────
def fg_chain_D():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.hdr  = lora_sdr.header(False, True, CR)
            self.crc  = lora_sdr.add_crc(True)
            self.ham  = lora_sdr.hamming_enc(CR, SF)
            self.ilv  = lora_sdr.interleaver(CR, SF, False, BW)
            self.gd   = lora_sdr.gray_demap(SF)
            self.sink = blocks.null_sink(gr.sizeof_int)
            self.connect(self.src, self.wh, self.hdr, self.crc,
                         self.ham, self.ilv, self.gd, self.sink)
    return FG()


# ── Chain E: … → modulate ───────────────────────────────────────
def fg_chain_E():
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.file_source(
                gr.sizeof_char, str(_input_file), False, 0, 0)
            self.src.set_begin_tag(pmt.PMT_NIL)
            self.wh   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.hdr  = lora_sdr.header(False, True, CR)
            self.crc  = lora_sdr.add_crc(True)
            self.ham  = lora_sdr.hamming_enc(CR, SF)
            self.ilv  = lora_sdr.interleaver(CR, SF, False, BW)
            self.gd   = lora_sdr.gray_demap(SF)
            self.mod  = lora_sdr.modulate(
                SF, SAMP_RATE, BW, [0x12],
                int(20 * 2**SF * SAMP_RATE / BW), PREAMB_LEN)
            self.sink = blocks.null_sink(gr.sizeof_gr_complex)
            self.connect(self.src, self.wh, self.hdr, self.crc,
                         self.ham, self.ilv, self.gd, self.mod, self.sink)
    return FG()


# ── Standalone: add_crc and hamming_enc (no tag dependency) ─────
def fg_add_crc():
    data = _raw_payload.tolist()
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.vector_source_b(data, False)
            self.blk  = lora_sdr.add_crc(True)
            self.sink = blocks.null_sink(gr.sizeof_char)
            self.connect(self.src, self.blk, self.sink)
    return FG()

def fg_hamming_enc():
    nibble_count = (3 + PAYLOAD_BYTES + 2) * 2
    data = np.random.randint(0, 16, size=nibble_count, dtype=np.uint8).tolist()
    class FG(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            self.src  = blocks.vector_source_b(data, False)
            self.blk  = lora_sdr.hamming_enc(CR, SF)
            self.sink = blocks.null_sink(gr.sizeof_char)
            self.connect(self.src, self.blk, self.sink)
    return FG()


# ───────────────────────────────────────────────────────────────────
# MEASURE ALL STEPS
# ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("MEASURING EACH STAGE (chain-subtraction method)")
print("=" * 70)

step_records = []

# ── Measure all chains ──────────────────────────────────────────
print("\n  Measuring cumulative chains...")

print("  Chain A  (whitening)...")
m_A = measure_step_fg(fg_chain_A)
print(f"    {m_A['mean_ms']:.3f} ± {m_A['std_ms']:.3f} ms")

print("  Chain B  (... + header)...")
m_B = measure_step_fg(fg_chain_B)
print(f"    {m_B['mean_ms']:.3f} ± {m_B['std_ms']:.3f} ms")

print("  Chain B2 (... + add_crc + hamming_enc)...")
m_B2 = measure_step_fg(fg_chain_B2)
print(f"    {m_B2['mean_ms']:.3f} ± {m_B2['std_ms']:.3f} ms")

print("  Chain C  (... + interleaver)...")
m_C = measure_step_fg(fg_chain_C)
print(f"    {m_C['mean_ms']:.3f} ± {m_C['std_ms']:.3f} ms")

print("  Chain D  (... + gray_demap)...")
m_D = measure_step_fg(fg_chain_D)
print(f"    {m_D['mean_ms']:.3f} ± {m_D['std_ms']:.3f} ms")

print("  Chain E  (... + modulate)...")
m_E = measure_step_fg(fg_chain_E)
print(f"    {m_E['mean_ms']:.3f} ± {m_E['std_ms']:.3f} ms")

print("\n  Measuring standalone blocks (add_crc, hamming_enc)...")
m_crc = measure_step_fg(fg_add_crc)
print(f"    add_crc     : {m_crc['mean_ms']:.3f} ± {m_crc['std_ms']:.3f} ms")
m_ham = measure_step_fg(fg_hamming_enc)
print(f"    hamming_enc : {m_ham['mean_ms']:.3f} ± {m_ham['std_ms']:.3f} ms")

# ── Derive isolated times ────────────────────────────────────────
print("\n  Deriving isolated stage times...")

m_whitening   = m_A
m_header      = _sub(m_B,  m_A)
# add_crc and hamming_enc: standalone (no tag dependency)
# interleaver:  Chain_C − Chain_B2  (B2 stops just before interleaver)
m_interleaver = _sub(m_C, m_B2)

m_gray_demap = _sub(m_D, m_C)
m_modulate   = _sub(m_E, m_D)

# ── Collect results ─────────────────────────────────────────────
step_records = [
    {'stage': "1. Whitening\n(whitening)",          **m_whitening},
    {'stage': "2. Header\n(header)",                **m_header},
    {'stage': "3. Add CRC\n(add_crc)",              **m_crc},
    {'stage': "4. Hamming Encode\n(hamming_enc)",   **m_ham},
    {'stage': "5. Interleaving\n(interleaver)",     **m_interleaver},
    {'stage': "6. Gray Demap\n(gray_demap)",        **m_gray_demap},
    {'stage': "7. Modulate\n(modulate)",            **m_modulate},
]

print("\n  Isolated times:")
for r in step_records:
    print(f"    {r['stage'].replace(chr(10),' '):<45} "
          f"{r['mean_ms']:>8.3f} ± {r['std_ms']:>6.3f} ms")

# ───────────────────────────────────────────────────────────────────
# TABULAR SUMMARY
# ───────────────────────────────────────────────────────────────────

df = pd.DataFrame(step_records)
total_ms = df['mean_ms'].sum()
df['pct'] = (df['mean_ms'] / total_ms * 100).round(1)

print("\n" + "=" * 70)
print("SUMMARY — TIME PER STAGE (1 payload)")
print("=" * 70)
hdr = f"{'Stage':<45} {'Mean (ms)':>10} {'Std (ms)':>10} {'%':>7}"
print(hdr)
print("-" * len(hdr))
for _, row in df.iterrows():
    nm = row['stage'].replace('\n', ' ')
    print(f"{nm:<45} {row['mean_ms']:>10.3f} {row['std_ms']:>10.3f} {row['pct']:>6.1f}%")
print("-" * len(hdr))
print(f"{'TOTAL (sum of stages)':<45} {total_ms:>10.3f}")

df.to_csv(DATA_DIR / "gr_step_profiling.csv", index=False)
print(f"\n  CSV saved: {DATA_DIR / 'gr_step_profiling.csv'}")

# ───────────────────────────────────────────────────────────────────
# PLOT — Bar chart + Pie chart
# ───────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(df)))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

labels_full  = [r['stage'] for r in step_records]
labels_short = [r['stage'].split('\n')[0] for r in step_records]
means   = df['mean_ms'].values
stds    = df['std_ms'].values
percents = df['pct'].values

# ---- Left: horizontal bar chart ----
ax1 = axes[0]
bars = ax1.barh(
    range(len(means)), means,
    xerr=stds,
    color=COLORS, edgecolor='black', linewidth=0.8,
    error_kw={'elinewidth': 1.5, 'capsize': 4}
)
ax1.set_yticks(range(len(labels_full)))
ax1.set_yticklabels(labels_full, fontsize=9)
ax1.set_xlabel('Mean Time (ms)', fontsize=11)
ax1.set_title(
    f'Time per Pipeline Stage\n'
    f'(1 payload, average of {NUM_RUNS} runs)',
    fontsize=12, fontweight='bold'
)
ax1.grid(True, axis='x', alpha=0.3)
ax1.invert_yaxis()

for bar, val, std, pct in zip(bars, means, stds, percents):
    ax1.text(
        bar.get_width() + std + max(means) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.2f} ms ({pct}%)',
        va='center', fontsize=8.5
    )

# ---- Right: pie chart ----
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    means,
    labels=labels_short,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    colors=COLORS,
    startangle=140,
    pctdistance=0.75,
    textprops={'fontsize': 9}
)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title(
    'Percentage of Total Time\nper Pipeline Stage',
    fontsize=12, fontweight='bold'
)

fig.suptitle(
    f"GNU Radio LoRa — Step Profiling\n"
    f"SF{SF} | BW={BW/1e3:.0f} kHz | CR=4/{4+CR} | "
    f"{PAYLOAD_BYTES} bytes/payload | CPU only",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
out = DATA_DIR / "gr_step_profiling.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Figure saved: {out}")

print("\n" + "=" * 70)
print("✅ Step profiling complete!")
print("=" * 70)