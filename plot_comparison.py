#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===================================================================
# COMPARISON PLOTS — Sionna/TF vs GNU Radio LoRa Transmitter
#
# Place all 6 CSVs in the same folder as this script and run it.
#
#   sionna_time_benchmark.csv     → batch_size, time_mean_ms, time_std_ms, per_pkt_ms, throughput
#   sionna_resource_benchmark.csv → batch_size, cpu_mean_pct, cpu_peak_pct, ram_mean_mb, ram_peak_mb, gpu_mean_mb, gpu_peak_mb
#   sionna_step_profiling.csv     → stage, mean_ms, std_ms, pct
#
#   gr_time_benchmark.csv         → batch_size, time_mean_ms, time_std_ms, per_pkt_ms, throughput
#   gr_resource_benchmark.csv     → batch_size, cpu_mean_pct, cpu_peak_pct, ram_mean_mb, ram_peak_mb
#   gr_step_profiling.csv         → stage, mean_ms, std_ms, pct
#
# Output figures (saved in the same folder):
#   Fig 1 — comparison_time_total.png : total processing time only
#   Fig 2 — comparison_time.png       : total + per-packet time
#   Fig 3 — comparison_resources.png  : CPU, RAM, GPU usage
#   Fig 4 — comparison_step_pie.png   : per-stage pie charts
# ===================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE  = Path(__file__).parent   # all CSVs and figures live here

C_SIONNA = '#2196F3'            # blue
C_GR     = '#FF5722'            # orange-red

# ───────────────────────────────────────────────────────────────────
# LOAD CSVs
# ───────────────────────────────────────────────────────────────────

def _load(name: str) -> pd.DataFrame:
    path = HERE / name
    if not path.exists():
        raise FileNotFoundError(
            f"\n  ❌ File not found: {path}\n"
            f"  Generate it by running the corresponding script/notebook."
        )
    return pd.read_csv(path)

df_s_time = _load('sionna_time_benchmark.csv')
df_g_time = _load('gr_time_benchmark.csv')
df_s_res  = _load('sionna_resource_benchmark.csv')
df_g_res  = _load('gr_resource_benchmark.csv')
df_s_step = _load('sionna_step_profiling.csv')
df_g_step = _load('gr_step_profiling.csv')

# Normalise Sionna column names if they differ between notebook versions
for df in (df_s_time, df_g_time):
    if 'time_mean_ms' not in df.columns and 'total_ms' in df.columns:
        df.rename(columns={'total_ms': 'time_mean_ms',
                           'total_std_ms': 'time_std_ms'}, inplace=True)

print("✅ CSVs loaded successfully.")

# ───────────────────────────────────────────────────────────────────
# HELPER: align two DataFrames on common batch sizes
# ───────────────────────────────────────────────────────────────────

def _align(df_s, df_g, key='batch_size'):
    common = sorted(set(df_s[key]) & set(df_g[key]))
    ds = df_s[df_s[key].isin(common)].set_index(key).loc[common]
    dg = df_g[df_g[key].isin(common)].set_index(key).loc[common]
    return np.array(common), ds, dg


bs, ds_t, dg_t = _align(df_s_time, df_g_time)
x     = np.arange(len(bs))
width = 0.35


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — TOTAL PROCESSING TIME ONLY
# ═══════════════════════════════════════════════════════════════════

fig1, ax = plt.subplots(figsize=(9, 5))

bars_s = ax.bar(x - width/2, ds_t['time_mean_ms'], width,
                yerr=ds_t['time_std_ms'],
                label='Sionna/TF', color=C_SIONNA, edgecolor='black',
                linewidth=0.6, capsize=4, error_kw={'elinewidth': 1.3})
bars_g = ax.bar(x + width/2, dg_t['time_mean_ms'], width,
                yerr=dg_t['time_std_ms'],
                label='GNU Radio', color=C_GR, edgecolor='black',
                linewidth=0.6, capsize=4, error_kw={'elinewidth': 1.3})

ax.set_xticks(x)
ax.set_xticklabels(bs, fontsize=9)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Total Processing Time (ms)', fontsize=11)
ax.set_title('Total Processing Time vs Batch Size', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar in bars_s:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color=C_SIONNA)
for bar in bars_g:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color=C_GR)

fig1.suptitle(
    'Sionna/TF vs GNU Radio — Processing Time Benchmark\n'
    'SF7 | BW=125 kHz | CR=4/5 | 10 bytes/payload',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
out1 = HERE / 'comparison_time_total.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Fig 1 saved: {out1}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — TOTAL TIME + PER-PACKET TIME
# ═══════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# ---- Subplot A: total time ----------------------------------------
ax = axes2[0]
bars_s = ax.bar(x - width/2, ds_t['time_mean_ms'], width,
                yerr=ds_t['time_std_ms'],
                label='Sionna/TF', color=C_SIONNA, edgecolor='black',
                linewidth=0.6, capsize=4, error_kw={'elinewidth': 1.3})
bars_g = ax.bar(x + width/2, dg_t['time_mean_ms'], width,
                yerr=dg_t['time_std_ms'],
                label='GNU Radio', color=C_GR, edgecolor='black',
                linewidth=0.6, capsize=4, error_kw={'elinewidth': 1.3})
ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=9)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Total Processing Time (ms)', fontsize=11)
ax.set_title('Total Processing Time vs Batch Size', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
for bar in bars_s:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color=C_SIONNA)
for bar in bars_g:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=7, color=C_GR)

# ---- Subplot B: per-packet time -----------------------------------
ax = axes2[1]
ax.bar(x - width/2, ds_t['per_pkt_ms'], width,
       label='Sionna/TF', color=C_SIONNA, edgecolor='black', linewidth=0.6)
ax.bar(x + width/2, dg_t['per_pkt_ms'], width,
       label='GNU Radio', color=C_GR,     edgecolor='black', linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=9)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('Time per Packet (ms)', fontsize=11)
ax.set_title('Per-Packet Processing Time vs Batch Size', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig2.suptitle(
    'Sionna/TF vs GNU Radio — Processing Time Benchmark\n'
    'SF7 | BW=125 kHz | CR=4/5 | 10 bytes/payload',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
out2 = HERE / 'comparison_time.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Fig 2 saved: {out2}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — RESOURCE USAGE × BATCH SIZE
# ═══════════════════════════════════════════════════════════════════

bs, ds_r, dg_r = _align(df_s_res, df_g_res)
x = np.arange(len(bs))

has_gpu_sionna = 'gpu_mean_mb' in ds_r.columns
has_gpu_gr     = 'gpu_mean_mb' in dg_r.columns
ncols = 3 if (has_gpu_sionna or has_gpu_gr) else 2

fig3, axes3 = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

# ---- CPU ----------------------------------------------------------
ax = axes3[0]
ax.bar(x - width/2, ds_r['cpu_mean_pct'], width,
       label='Sionna/TF (mean)', color=C_SIONNA, edgecolor='black', linewidth=0.6)
ax.bar(x + width/2, dg_r['cpu_mean_pct'], width,
       label='GNU Radio (mean)', color=C_GR,     edgecolor='black', linewidth=0.6)
ax.axhline(100, color='gray', linestyle=':', linewidth=1.2, label='100% = 1 core')
ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=9)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('CPU Usage (%)', fontsize=11)
ax.set_title('CPU Usage\n(>100% = multi-core parallelism)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# ---- RAM ----------------------------------------------------------
ax = axes3[1]
ax.bar(x - width/2, ds_r['ram_mean_mb'], width,
       label='Sionna/TF', color=C_SIONNA, edgecolor='black', linewidth=0.6)
ax.bar(x + width/2, dg_r['ram_mean_mb'], width,
       label='GNU Radio', color=C_GR,     edgecolor='black', linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=9)
ax.set_xlabel('Batch Size (packets)', fontsize=11)
ax.set_ylabel('RAM Usage (MB)', fontsize=11)
ax.set_title('RAM Usage (Mean)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# ---- GPU (Sionna only) --------------------------------------------
if ncols == 3:
    ax = axes3[2]
    s_gpu = ds_r['gpu_mean_mb'] if has_gpu_sionna else np.zeros(len(bs))
    g_gpu = dg_r['gpu_mean_mb'] if has_gpu_gr     else np.zeros(len(bs))
    ax.bar(x - width/2, s_gpu, width,
           label='Sionna/TF', color=C_SIONNA, edgecolor='black', linewidth=0.6)
    ax.bar(x + width/2, g_gpu, width,
           label='GNU Radio (CPU only)', color=C_GR, edgecolor='black',
           linewidth=0.6, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(bs, fontsize=9)
    ax.set_xlabel('Batch Size (packets)', fontsize=11)
    ax.set_ylabel('GPU Memory (MB)', fontsize=11)
    ax.set_title('GPU Memory Usage\n(GNU Radio: CPU only)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig3.suptitle(
    'Sionna/TF vs GNU Radio — Resource Usage Benchmark\n'
    'SF7 | BW=125 kHz | CR=4/5 | 10 bytes/payload',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
out3 = HERE / 'comparison_resources.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Fig 3 saved: {out3}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — PER-STAGE PIE CHARTS
# ═══════════════════════════════════════════════════════════════════

def _short(s: str) -> str:
    return str(s).split('\n')[0].strip()

for df in (df_s_step, df_g_step):
    if 'pct' not in df.columns:
        df['pct'] = df['mean_ms'] / df['mean_ms'].sum() * 100

s_labels = [_short(s) for s in df_s_step['stage']]
g_labels = [_short(s) for s in df_g_step['stage']]
s_vals   = df_s_step['mean_ms'].values
g_vals   = df_g_step['mean_ms'].values

COLORS_S = plt.cm.Blues(np.linspace(0.35, 0.85, len(s_vals)))
COLORS_G = plt.cm.Oranges(np.linspace(0.35, 0.85, len(g_vals)))

fig4, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(16, 7))

def _pie(ax, vals, labels, colors, title):
    _, _, autotexts = ax.pie(
        vals, labels=labels,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=colors, startangle=140, pctdistance=0.75,
        textprops={'fontsize': 9}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(title, fontsize=12, fontweight='bold')

_pie(ax_s, s_vals, s_labels, COLORS_S,
     f'Sionna/TF — Time per Stage\n(total = {s_vals.sum():.1f} ms)')
_pie(ax_g, g_vals, g_labels, COLORS_G,
     f'GNU Radio — Time per Stage\n(total = {g_vals.sum():.1f} ms)')

fig4.suptitle(
    'Sionna/TF vs GNU Radio — Processing Time Distribution per Stage\n'
    'SF7 | BW=125 kHz | CR=4/5 | 1 payload | 10 bytes',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
out4 = HERE / 'comparison_step_pie.png'
plt.savefig(out4, dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Fig 4 saved: {out4}")

print("\n" + "=" * 60)
print("✅ All comparison figures generated!")
print("=" * 60)
