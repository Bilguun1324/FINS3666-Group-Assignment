# WBTC/WETH Ethereum Arbitrage Dataset Pipeline

This repository contains the dataset pipeline for a FINS3666 group project on **Ethereum mainnet WBTC/WETH DEX arbitrage**.

The project is not a live trading bot. It is a **historical research pipeline** that collects on-chain data, rebuilds pool state through time, and labels when a cross-DEX arbitrage trade would have been profitable after fees and gas.

## What This Project Studies

We study the same market on two decentralized exchanges:

- `Uniswap V2`
- `SushiSwap V2`

The asset pair is:

- `WBTC` = wrapped Bitcoin on Ethereum
- `WETH` = wrapped Ether on Ethereum

At any point in time, both DEX pools hold reserves of WBTC and WETH. Those reserves imply a price. If the price on one DEX is lower than the price on the other, there may be an arbitrage opportunity:

1. buy WBTC on the cheaper DEX
2. sell WBTC on the more expensive DEX
3. keep the difference if it is still positive after swap fees and Ethereum gas costs

The pipeline turns that idea into a dataset that the rest of the team can use for analysis, backtesting, evaluation, and report writing.

## What Was Run

On **April 6, 2026** the full pipeline was run with a public Ethereum RPC endpoint and the default configuration in [`config/markets.yaml`](config/markets.yaml).

The resulting time window is:

- start: `2025-12-07 09:17:00+00:00`
- end: `2026-04-06 08:36:00+00:00`

## Generated Outputs

### Raw data

- `data/raw/event_logs.parquet`: `68,702` rows
- `data/raw/block_headers.parquet`: `21,148` rows
- `data/raw/pair_metadata.parquet`
- `data/raw/source_manifest.yaml`

### Curated data

- `data/curated/events_curated.parquet`: `68,702` rows
- `data/curated/swaps_raw.parquet`: `29,428` rows
- `data/curated/pool_state_1m.parquet`: `345,130` rows
- `data/curated/arb_labels_1m.parquet`: `172,370` rows
- `data/curated/split_assignments.parquet`: `172,370` rows
- `data/curated/split_manifest.yaml`
- `data/curated/qc_report.json`

### Report outputs

- `outputs/report/price_summary.csv`
- `outputs/report/swap_activity_by_hour.csv`
- `outputs/report/liquidity_depth_summary.csv`
- `outputs/report/opportunity_summary.csv`
- `outputs/report/split_descriptives.csv`
- `outputs/report/pool_price_series.png`
- `outputs/report/cross_dex_spread.png`
- `outputs/metadata/dataset_dictionary.csv`

## Current Dataset Snapshot

From the generated dataset:

- DEX venues covered: `uniswap_v2`, `sushiswap_v2`
- arbitrage label rows: `172,370`
- net-positive opportunity flags: `4`
- mean net edge: `-100.0464 bps`

Chronological split counts:

- train: `120,658`
- validation: `25,856`
- test: `25,856`

This is a useful result, not a bad one. It shows that once fees and gas are included, most apparent price differences do **not** survive as profitable trades. That is exactly the kind of economically meaningful finding the assignment expects.

## Quality Checks

The QC report in `data/curated/qc_report.json` passed all checks:

- no schema errors
- no duplicate swap events
- no timestamp monotonicity violations
- no non-positive reserves
- no 1-minute pool-state gaps
- no null stale-state flags

## Pipeline Logic

The pipeline runs in six stages.

### 1. Collect raw chain data

The command:

```bash
mev-dataset collect-chain-data
```

collects:

- pool event logs from Ethereum JSON-RPC
- block headers for timestamps and base fees
- pair metadata for both DEX pools

The raw event types used are:

- `Swap`
- `Sync`
- `Mint`
- `Burn`

These are enough to reconstruct trading activity and pool reserves through time.

### 2. Normalize and decode logs

The command:

```bash
mev-dataset build-curated-dataset
```

converts raw hexadecimal event data into readable numeric fields.

This stage:

- handles token decimal conversion
- fixes token ordering differences across pools
- reconstructs post-event WBTC and WETH reserves
- computes pool-implied `ETH per BTC` prices

### 3. Build swap-level dataset

The `swaps_raw` table is the trade-level table.

Each row contains:

- event timestamp
- block number
- venue
- transaction hash
- swap size in WBTC and WETH
- post-trade reserves
- trade direction

This is the most useful table for microstructure-style background analysis.

### 4. Build 1-minute pool state

The `pool_state_1m` table is the minute-level research dataset.

For each DEX and each minute, it records:

- latest forward-filled reserve state
- implied price
- swap count
- WBTC and WETH volume
- stale-state flag

This is the main table for charts, descriptive statistics, and intraday analysis.

### 5. Build arbitrage labels

The command:

```bash
mev-dataset build-arb-labels
```

creates `arb_labels_1m`.

For each minute, it checks both directions:

- buy on Uniswap, sell on Sushi
- buy on Sushi, sell on Uniswap

It uses constant-product AMM math, not a naive mid-price difference.

The label includes:

- `gross_edge_bps`
- `fee_cost_bps`
- `gas_cost_weth`
- `net_edge_bps`
- `buy_dex`
- `sell_dex`
- `opportunity_flag`

The default trade size is `0.10 WBTC`.

### 6. Split, validate, and report

The pipeline then:

- builds chronological train/validation/test splits
- runs QC checks
- writes summary CSVs and charts for the report

The command:

```bash
mev-dataset make-background-report
```

creates the final report-ready summary files.

## Why The Output Matters For The Assignment

This dataset supports the assignment requirements directly.

It gives the team:

- **historical intraday market data**
- **background analysis inputs** such as price, volume, and liquidity
- **train / validation / test splits** for out-of-sample evaluation
- **economically meaningful labels** that include costs, not just raw spreads
- **report-ready figures and tables**

In practical terms:

- the dataset role is done here
- strategy design can be built on top of `arb_labels_1m`
- descriptive analysis can be built on top of `swaps_raw` and `pool_state_1m`

## Commands

Install the package:

```bash
python3 -m pip install -e .
```

Set an Ethereum RPC URL:

```bash
export ETH_RPC_URL="https://ethereum-rpc.publicnode.com"
```

Run the whole pipeline from scratch:

```bash
mev-dataset run-all
```

Or run stage by stage:

```bash
mev-dataset collect-chain-data
mev-dataset build-curated-dataset
mev-dataset build-arb-labels
mev-dataset make-background-report
```

## Notebook

Use the notebook below to inspect the final dataset interactively:

- `notebooks/01_dataset_background.ipynb`

It loads the curated tables and reproduces the core charts and summary tables.

## Important Limits

This is a **historical post-block research dataset**, not a full production MEV system.

It does **not** include:

- mempool prediction
- block-builder ordering logic
- frontrunning or sandwich simulation
- live execution infrastructure

That is intentional. For an academic project, this dataset is the correct scope: rigorous, reproducible, and grounded in real market data.
