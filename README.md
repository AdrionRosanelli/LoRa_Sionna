# LoRa_Sionna

This repository contains a study and comparison of two LoRa physical-layer transmitter implementations:

- a **Sionna + TensorFlow** implementation, and
- a **GNU Radio** implementation used as a reference baseline.

At the moment, the Sionna-based work includes the **transmitter path only** (the notebook `Lora_Phy_Tx_Batch.ipynb`), while the GNU Radio side is used to evaluate performance and resource usage.

## Project Overview

The goal of this project is to investigate and compare the behavior of a LoRa PHY transmitter implemented with Sionna and TensorFlow against an existing GNU Radio implementation. The comparison focuses on:

- processing time,
- CPU and memory usage,
- step-by-step profiling of the transmitter pipeline,
- and visualization of the resulting benchmark metrics.

## Repository Contents

### Sionna / TensorFlow implementation

- `Lora_Phy_Tx_Batch.ipynb`
  - Main notebook for the LoRa transmitter simulation.
  - Contains the Sionna-based transmitter pipeline and analysis.
  - Recommended execution environment: **WSL**.

### GNU Radio benchmark scripts

The following scripts are intended to reproduce benchmark experiments for the GNU Radio implementation:

- `gr_cell1_time_benchmark.py`
  - Measures processing time for different batch sizes.
- `gr_cell2_resource_benchmark.py`
  - Measures CPU and memory usage during execution.
- `gr_cell3_step_profiling.py`
  - Profiles the cost of each stage of the transmitter pipeline.

These scripts are expected to run in a **Radioconda / GNU Radio** environment and require the `gr-lora_sdr` repository to be available locally.

Example command used to run the GNU Radio benchmark script:

```powershell
& C:/Users/aaros/radioconda/python.exe "c:/Adrion/Doutorado/Sionna - LoRa/Simulador LoRa Phy/gr-lora_sdr-master/gr_cell1_time_benchmark.py"
```

The `gr-lora_sdr` source repository should be cloned from:

https://github.com/tapparelj/gr-lora_sdr.git

## Comparison and Visualization

- `plot_comparison.py`
  - Generates comparison plots between the Sionna and GNU Radio results.
  - Requires the relevant `.csv` files produced by the benchmark scripts/notebooks.

## Additional Reference Material

- `Sionna_tutorial_part1.ipynb`
  - A first tutorial notebook for understanding the Sionna PHY workflow.

## Environment and Tested Versions

The experiments in this repository were run with the following software versions:

- **Python**: 3.12.3
- **TensorFlow**: 2.16.1
- **Sionna**: 1.2.1
- **GNU Radio**: 3.10.12.0
- **GNU Radio Python runtime**: Python 3.12.9

These versions are relevant for reproducing the results and for understanding which environment was used during the benchmarks.

## Notes

- This repository currently focuses on the **transmitter** side of the LoRa PHY chain.
- The comparison is mainly based on benchmark outputs and profiling results.
- The exact execution environment may vary depending on the machine setup (WSL for the notebook flow, GNU Radio/Radioconda for the reference scripts).

## Suggested Workflow

1. Run the Sionna transmitter notebook.
2. Run the GNU Radio benchmark scripts.
3. Generate the comparison plots with `plot_comparison.py`.

