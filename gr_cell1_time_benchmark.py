#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================
# GR_CELL1 — PROCESSING TIME BENCHMARK × BATCH SIZE
# GNU Radio LoRa Transmitter
#
# Equivalent to cell1_time_benchmark.py (Sionna).
# Measures total and per-packet processing time across batch sizes
# by running the complete GNU Radio LoRa TX flowgraph.
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

NUM_RUNS      = 20       # measurement runs per batch size (averaged)
NUM_WARMUP    = 10        # warm-up runs (discarded)
PAYLOAD_BYTES = 10       # bytes per payload
BATCH_SIZES   = [1, 10, 50, 100, 200, 350, 500, 1000]

LORA_CONFIG = dict(sf=7, bw=125000, cr=2)

print("=" * 70)
print("GR_CELL1 — PROCESSING TIME BENCHMARK × BATCH SIZE")
print(f"  SF={LORA_CONFIG['sf']}  BW={LORA_CONFIG['bw']/1e3:.0f} kHz  "
      f"CR=4/{4+LORA_CONFIG['cr']}  payload={PAYLOAD_BYTES} bytes")
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
# INPUT FILE GENERATOR
# ───────────────────────────────────────────────────────────────────

def generate_lora_input_file(num_packets: int, payload_bytes: int = 10) -> Path:
    """
    Generates a gr-lora_sdr input file (hex payloads separated by commas,
    terminated with a trailing comma as required by the whitening block).
    """
    filepath = DATA_DIR / f"test_{num_packets}pkt_{payload_bytes}b.txt"
    payloads = []
    for _ in range(num_packets):
        raw = np.random.randint(0, 256, size=payload_bytes, dtype=np.uint8)
        payloads.append(''.join(f'{b:02x}' for b in raw))
    filepath.write_text(','.join(payloads) + ',')
    return filepath


# ───────────────────────────────────────────────────────────────────
# FLOWGRAPH FACTORY
# ───────────────────────────────────────────────────────────────────

def create_lora_flowgraph(input_file: Path, sf=7, bw=125000, cr=2):
    """Creates a complete LoRa TX flowgraph (file_source → null_sink)."""

    class LoRaTxFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            samp_rate  = 500000
            preamb_len = 8

            self.file_source = blocks.file_source(
                gr.sizeof_char * 1, str(input_file), False, 0, 0
            )
            self.file_source.set_begin_tag(pmt.PMT_NIL)

            self.whitening  = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.header     = lora_sdr.header(False, True, cr)
            self.add_crc    = lora_sdr.add_crc(True)
            self.hamming_enc = lora_sdr.hamming_enc(cr, sf)
            self.interleaver = lora_sdr.interleaver(cr, sf, False, bw)
            self.gray_demap  = lora_sdr.gray_demap(sf)
            self.modulate    = lora_sdr.modulate(
                sf, samp_rate, bw, [0x12],
                int(20 * 2**sf * samp_rate / bw),
                preamb_len
            )
            self.null_sink = blocks.null_sink(gr.sizeof_gr_complex * 1)

            self.connect(self.file_source, self.whitening)
            self.connect(self.whitening,   self.header)
            self.connect(self.header,      self.add_crc)
            self.connect(self.add_crc,     self.hamming_enc)
            self.connect(self.hamming_enc, self.interleaver)
            self.connect(self.interleaver, self.gray_demap)
            self.connect(self.gray_demap,  self.modulate)
            self.connect(self.modulate,    self.null_sink)

    return LoRaTxFlowgraph()


# ───────────────────────────────────────────────────────────────────
# SINGLE RUN: run flowgraph and return elapsed time in ms
# ───────────────────────────────────────────────────────────────────

def _run_flowgraph(input_file: Path, timeout: float = 60.0, **lora_kw) -> float | None:
    """
    Runs one complete flowgraph execution.
    Returns elapsed time in ms, or None on timeout/error.
    """
    tb       = create_lora_flowgraph(input_file, **lora_kw)
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
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if error[0]:
            print(f"    ERROR: {error[0]}")
            return None
        return elapsed_ms
    else:
        tb.stop()
        print("    WARNING: timeout — run discarded")
        return None


# ───────────────────────────────────────────────────────────────────
# TIME MEASUREMENT FUNCTION
# ───────────────────────────────────────────────────────────────────

def measure_time(num_packets: int,
                 payload_bytes: int = PAYLOAD_BYTES,
                 num_runs: int = NUM_RUNS,
                 num_warmup: int = NUM_WARMUP,
                 **lora_kw) -> dict | None:
    """
    Measures processing time (ms) for a given batch size.

    Parameters
    ----------
    num_packets  : int — batch size
    payload_bytes: int — bytes per payload
    num_runs     : int — measurement runs (averaged)
    num_warmup   : int — warm-up runs (discarded)
    **lora_kw    : sf, bw, cr passed to the flowgraph

    Returns
    -------
    dict with: time_mean_ms, time_std_ms, time_min_ms, time_max_ms,
               per_pkt_ms, throughput
    """
    input_file = generate_lora_input_file(num_packets, payload_bytes)

    # Warm-up runs (allow GR scheduler to stabilise)
    print(f"    Warm-up ({num_warmup} runs)...", end='', flush=True)
    for _ in range(num_warmup):
        _run_flowgraph(input_file, **lora_kw)
    print(" done")

    # Timed runs
    times_ms = []
    for i in range(num_runs):
        t = _run_flowgraph(input_file, **lora_kw)
        if t is not None:
            times_ms.append(t)

    if not times_ms:
        return None

    mean_ms    = float(np.mean(times_ms))
    per_pkt_ms = mean_ms / num_packets
    throughput = num_packets / (mean_ms / 1000.0)

    return {
        'time_mean_ms': mean_ms,
        'time_std_ms':  float(np.std(times_ms)),
        'time_min_ms':  float(np.min(times_ms)),
        'time_max_ms':  float(np.max(times_ms)),
        'per_pkt_ms':   per_pkt_ms,
        'throughput':   throughput,
    }


# ───────────────────────────────────────────────────────────────────
# BENCHMARK LOOP
# ───────────────────────────────────────────────────────────────────

benchmark_records = []

for bs in BATCH_SIZES:
    print(f"\n  Batch size = {bs} packets")
    print("  " + "-" * 48)

    m = measure_time(bs, payload_bytes=PAYLOAD_BYTES, **LORA_CONFIG)

    if m is None:
        print("    SKIPPED (no successful runs)")
        continue

    benchmark_records.append({'batch_size': bs, **m})

    print(f"    Total time  : {m['time_mean_ms']:.2f} ± {m['time_std_ms']:.2f} ms")
    print(f"    Per packet  : {m['per_pkt_ms']:.2f} ms  |  "
          f"Throughput: {m['throughput']:.1f} pkt/s")

df_time = pd.DataFrame(benchmark_records)

# Tabular summary
print("\n" + "=" * 70)
print("SUMMARY — TIME BENCHMARK")
print("=" * 70)
cols = ['batch_size', 'time_mean_ms', 'time_std_ms', 'per_pkt_ms', 'throughput']
print(df_time[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

df_time.to_csv(DATA_DIR / "gr_time_benchmark.csv", index=False)
print(f"\n  CSV saved: {DATA_DIR / 'gr_time_benchmark.csv'}")


# ───────────────────────────────────────────────────────────────────
# PLOT — Processing Time vs Batch Size
# ───────────────────────────────────────────────────────────────────

cfg = LORA_CONFIG
bs_vals = df_time['batch_size'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: Total processing time ---
ax = axes[0]
ax.errorbar(
    bs_vals,
    df_time['time_mean_ms'],
    yerr=df_time['time_std_ms'],
    fmt='o-', color='steelblue', linewidth=2, markersize=8,
    capsize=5, elinewidth=1.5, label='Mean ± Std'
)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Total Processing Time (ms)', fontsize=11)
ax.set_title('Total Processing Time vs Batch Size', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
for x, y, s in zip(bs_vals, df_time['time_mean_ms'], df_time['time_std_ms']):
    ax.annotate(f'{y:.1f} ms', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8.5)

# --- Right: Per-packet processing time ---
ax = axes[1]
ax.plot(bs_vals, df_time['per_pkt_ms'], 's--', color='darkorange',
        linewidth=2, markersize=8, label='Per-packet time')
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Time per Packet (ms)', fontsize=11)
ax.set_title('Per-Packet Processing Time vs Batch Size', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
for x, y in zip(bs_vals, df_time['per_pkt_ms']):
    ax.annotate(f'{y:.2f} ms', (x, y),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=8.5)

fig.suptitle(
    f"GNU Radio LoRa — Processing Time Benchmark\n"
    f"SF{cfg['sf']} | BW={cfg['bw']/1e3:.0f} kHz | CR=4/{4+cfg['cr']} | "
    f"{PAYLOAD_BYTES} bytes/payload | {NUM_RUNS} runs",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out = DATA_DIR / "gr_time_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Figure saved: {out}")
