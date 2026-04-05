# WBTC/WETH Ethereum MEV Dataset Pipeline

This project builds a reproducible dataset package for a UNSW FINS3666 assignment on
Ethereum mainnet `WBTC/WETH` DEX-vs-DEX arbitrage. The implementation focuses on the
dataset layer needed by the rest of the team:

- raw event and block collection from public Ethereum JSON-RPC endpoints
- curated swap and pool-state datasets
- executable arbitrage labels using constant-product AMM math
- chronological train/validation/test splits
- QC checks and background-analysis outputs for the report

## Install

```bash
python3 -m pip install -e .
```

## Environment

Set a public or personal Ethereum RPC URL before live collection:

```bash
export ETH_RPC_URL="https://ethereum-rpc.publicnode.com"
```

The commands will also accept `--rpc-url` explicitly.

## Commands

```bash
mev-dataset collect-chain-data
mev-dataset build-curated-dataset
mev-dataset build-arb-labels
mev-dataset make-background-report
mev-dataset run-all
```

## Outputs

- `data/raw/`
  - `pair_metadata.parquet`
  - `event_logs.parquet`
  - `block_headers.parquet`
  - `source_manifest.yaml`
- `data/curated/`
  - `events_curated.parquet`
  - `swaps_raw.parquet`
  - `pool_state_1m.parquet`
  - `arb_labels_1m.parquet`
  - `split_assignments.parquet`
  - `split_manifest.yaml`
  - `qc_report.json`
- `outputs/report/`
  - summary CSV tables and PNG charts
- `outputs/metadata/`
  - `dataset_dictionary.csv`

## Notes

- The code uses Ethereum mainnet `WBTC` and `WETH`, not native BTC.
- The arbitrage labels are realized post-block research signals, not mempool or builder-level MEV predictions.
- The default window is the latest 120 days unless overridden in `config/markets.yaml`.
# FINS3666-Group-Assignment
