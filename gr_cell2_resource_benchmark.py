#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================
# GR_CELL2 — RESOURCE USAGE BENCHMARK × BATCH SIZE
# GNU Radio LoRa Transmitter
#
# Equivalent to cell2_resource_benchmark.py (Sionna).
# Measures CPU% and RAM while the complete GNU Radio LoRa TX
# flowgraph runs for different batch sizes.
#
# NOTE on CPU measurement:
# ─────────────────────────
# psutil.cpu_percent(interval=None) returns the CPU usage since the
# last call to cpu_percent() on that Process object. The FIRST call
# always returns 0.0 (no reference point yet) — this is a psutil
# design requirement. We prime the counter before starting the
# monitor thread and discard the first sample to avoid this artifact.
#
# CPU% can exceed 100% if the GNU Radio scheduler spawns multiple
# worker threads (which it does by default). This reflects real
# multi-core usage.
#
# NOTE on RAM:
# ─────────────────────────
# We measure RSS (Resident Set Size) of the Python process. This
# includes the GR runtime, all block buffers, and the GNU Radio
# scheduler. It is the most stable and reliable memory metric here.
# ===================================================================

import time
import threading
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import psutil

# ───────────────────────────────────────────────────────────────────
# PARAMETERS
# ───────────────────────────────────────────────────────────────────

BASE_DIR = Path("C:/Adrion/Doutorado/Sionna - LoRa/Simulador LoRa Phy/gr-lora_sdr-master")
DATA_DIR = BASE_DIR / "data" / "GRC_default"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NUM_RUNS      = 20
NUM_WARMUP    = 10
PAYLOAD_BYTES = 10
BATCH_SIZES   = [1, 10, 50, 100, 200, 350, 500, 1000]

LORA_CONFIG = dict(sf=7, bw=125000, cr=2)

_PROCESS = psutil.Process(os.getpid())

print("=" * 70)
print("GR_CELL2 — RESOURCE USAGE BENCHMARK × BATCH SIZE")
print(f"  SF={LORA_CONFIG['sf']}  BW={LORA_CONFIG['bw']/1e3:.0f} kHz  "
      f"CR=4/{4+LORA_CONFIG['cr']}  payload={PAYLOAD_BYTES} bytes")
print(f"  Logical CPUs : {psutil.cpu_count(logical=True)}")
print(f"  Total RAM    : {psutil.virtual_memory().total / 1024**3:.1f} GB")
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

    class LoRaTxFlowgraph(gr.top_block):
        def __init__(self):
            gr.top_block.__init__(self)
            samp_rate  = 500000
            preamb_len = 8

            self.file_source = blocks.file_source(
                gr.sizeof_char * 1, str(input_file), False, 0, 0
            )
            self.file_source.set_begin_tag(pmt.PMT_NIL)

            self.whitening   = lora_sdr.whitening(True, False, ',', 'packet_len')
            self.header      = lora_sdr.header(False, True, cr)
            self.add_crc     = lora_sdr.add_crc(True)
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
# RESOURCE MONITOR (separate thread — same pattern as Cell2 Sionna)
# ───────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """
    Samples CPU% and RAM in a separate thread while the target runs.

    CPU sampling uses cpu_percent(interval=None) which returns usage
    since the last call. The counter MUST be primed once before
    start() to avoid the mandatory 0.0 first-call artifact.

    Usage:
        _PROCESS.cpu_percent(interval=None)   # prime once at startup
        mon = ResourceMonitor(interval_s=0.05)
        mon.start()
        ... run flowgraph ...
        mon.stop()
        stats = mon.stats()
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s   = interval_s
        self._cpu_samples : list[float] = []
        self._ram_samples : list[float] = []
        self._stop_event  = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop_event.clear()
        self._cpu_samples.clear()
        self._ram_samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        # Prime counter at thread start (its first call returns 0.0 by design)
        _PROCESS.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self._cpu_samples.append(_PROCESS.cpu_percent(interval=None))
            self._ram_samples.append(_PROCESS.memory_info().rss / 1024**2)  # MB
            time.sleep(self.interval_s)

    def stats(self) -> dict:
        def _s(lst):
            if not lst:
                return {'mean': 0.0, 'peak': 0.0, 'std': 0.0}
            # Drop first sample (still affected by priming gap)
            a = np.array(lst[1:] if len(lst) > 1 else lst, dtype=float)
            return {'mean': float(a.mean()), 'peak': float(a.max()), 'std': float(a.std())}
        return {
            'cpu': _s(self._cpu_samples),
            'ram': _s(self._ram_samples),
            'n_samples': len(self._cpu_samples),
        }


# Prime the process CPU counter once (mandatory before first measurement)
_PROCESS.cpu_percent(interval=None)


# ───────────────────────────────────────────────────────────────────
# RESOURCE MEASUREMENT FUNCTION
# ───────────────────────────────────────────────────────────────────

def _run_flowgraph_timed(input_file: Path, timeout: float = 60.0, **lora_kw) -> bool:
    """Runs the flowgraph; returns True on success, False on timeout/error."""
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

    threading.Thread(target=_worker, daemon=True).start()
    ok = finished.wait(timeout=timeout)
    if not ok:
        tb.stop()
    return ok and error[0] is None


def measure_resources(num_packets: int,
                      payload_bytes: int = PAYLOAD_BYTES,
                      num_runs: int = NUM_RUNS,
                      num_warmup: int = NUM_WARMUP,
                      **lora_kw) -> dict | None:
    """
    Measures CPU% and RAM while the GNU Radio LoRa TX flowgraph runs.

    Parameters
    ----------
    num_packets  : int — batch size
    payload_bytes: int — bytes per payload
    num_runs     : int — measurement runs (averaged)
    num_warmup   : int — warm-up runs (discarded)
    **lora_kw    : sf, bw, cr

    Returns
    -------
    dict with: cpu_mean_pct, cpu_peak_pct, ram_mean_mb, ram_peak_mb
    """
    input_file = generate_lora_input_file(num_packets, payload_bytes)

    # Warm-up
    print(f"    Warm-up ({num_warmup} runs)...", end='', flush=True)
    for _ in range(num_warmup):
        _run_flowgraph_timed(input_file, **lora_kw)
    print(" done")

    monitor   = ResourceMonitor(interval_s=0.05)
    all_stats = []

    for i in range(num_runs):
        monitor.start()
        ok = _run_flowgraph_timed(input_file, **lora_kw)
        monitor.stop()

        if ok:
            all_stats.append(monitor.stats())

    if not all_stats:
        return None

    cpu_means = [s['cpu']['mean'] for s in all_stats]
    cpu_peaks = [s['cpu']['peak'] for s in all_stats]
    ram_means = [s['ram']['mean'] for s in all_stats]
    ram_peaks = [s['ram']['peak'] for s in all_stats]

    return {
        'cpu_mean_pct': float(np.mean(cpu_means)),
        'cpu_peak_pct': float(np.max(cpu_peaks)),
        'ram_mean_mb':  float(np.mean(ram_means)),
        'ram_peak_mb':  float(np.max(ram_peaks)),
    }


# ───────────────────────────────────────────────────────────────────
# BENCHMARK LOOP
# ───────────────────────────────────────────────────────────────────

benchmark_records = []

for bs in BATCH_SIZES:
    print(f"\n  Batch size = {bs} packets")
    print("  " + "-" * 48)

    m = measure_resources(bs, payload_bytes=PAYLOAD_BYTES, **LORA_CONFIG)

    if m is None:
        print("    SKIPPED (no successful runs)")
        continue

    benchmark_records.append({'batch_size': bs, **m})

    print(f"    CPU : mean={m['cpu_mean_pct']:.1f}%  peak={m['cpu_peak_pct']:.1f}%")
    print(f"    (>100% = multiple GR worker threads on multi-core)")
    print(f"    RAM : mean={m['ram_mean_mb']:.0f} MB  peak={m['ram_peak_mb']:.0f} MB")

df_res = pd.DataFrame(benchmark_records)

print("\n" + "=" * 70)
print("SUMMARY — RESOURCE BENCHMARK")
print("=" * 70)
cols = ['batch_size', 'cpu_mean_pct', 'cpu_peak_pct', 'ram_mean_mb', 'ram_peak_mb']
print(df_res[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

df_res.to_csv(DATA_DIR / "gr_resource_benchmark.csv", index=False)
print(f"\n  CSV saved: {DATA_DIR / 'gr_resource_benchmark.csv'}")


# ───────────────────────────────────────────────────────────────────
# PLOT — Resource Usage vs Batch Size
# ───────────────────────────────────────────────────────────────────

cfg     = LORA_CONFIG
bs_vals = df_res['batch_size'].values

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# --- CPU ---
ax = axes[0]
ax.plot(bs_vals, df_res['cpu_mean_pct'], 's--', color='darkorange',
        linewidth=2, markersize=7, label='Mean')
ax.plot(bs_vals, df_res['cpu_peak_pct'], '^-', color='red',
        linewidth=2, markersize=7, label='Peak')
ax.axhline(100, color='gray', linestyle=':', linewidth=1.2, label='100% = 1 core')
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('CPU Usage (%)', fontsize=11)
ax.set_title('CPU Usage\n(>100% = multi-threaded GR scheduler)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- RAM ---
ax = axes[1]
ax.plot(bs_vals, df_res['ram_mean_mb'], 's--', color='purple',
        linewidth=2, markersize=7, label='Mean')
ax.plot(bs_vals, df_res['ram_peak_mb'], '^-', color='darkviolet',
        linewidth=2, markersize=7, label='Peak')
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Process RAM (MB)', fontsize=11)
ax.set_title('RAM Usage\n(Process RSS)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# # --- Measurement notes panel ---
# ax = axes[2]
# ax.axis('off')
# notes = (
#     "Measurement notes\n"
#     "─────────────────\n\n"
#     "CPU (%):\n"
#     "  psutil measures the Python process.\n"
#     "  GR uses multiple worker threads\n"
#     "  (one per block by default), so >100%\n"
#     "  reflects real multi-core parallelism.\n"
#     "  Counter is primed before each run\n"
#     "  to avoid the 0.0 first-call artifact.\n\n"
#     "RAM (MB):\n"
#     "  RSS (Resident Set Size) of the process.\n"
#     "  Includes GR runtime, all block buffers\n"
#     "  and the scheduler overhead.\n\n"
#     "No GPU metric: GNU Radio runs on CPU only."
# )
# ax.text(0.02, 0.98, notes,
#         transform=ax.transAxes,
#         fontsize=8.5, verticalalignment='top',
#         fontfamily='monospace',
#         bbox=dict(boxstyle='round', facecolor='lightyellow',
#                   edgecolor='gray', alpha=0.7))

fig.suptitle(
    f"GNU Radio LoRa — Resource Usage Benchmark\n"
    f"SF{cfg['sf']} | BW={cfg['bw']/1e3:.0f} kHz | CR=4/{4+cfg['cr']} | "
    f"{PAYLOAD_BYTES} bytes/payload | {NUM_RUNS} runs",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out = DATA_DIR / "gr_resource_benchmark.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Figure saved: {out}")
