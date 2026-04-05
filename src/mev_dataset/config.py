"""Configuration models and loaders for the dataset pipeline."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TokenConfig(BaseModel):
    symbol: str
    address: str
    decimals: int

    model_config = ConfigDict(frozen=True)

    @field_validator("address")
    @classmethod
    def normalize_address(cls, value: str) -> str:
        return value.lower()


class DexConfig(BaseModel):
    name: str
    factory_address: str
    fee_bps: float
    pair_address: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @field_validator("factory_address", "pair_address")
    @classmethod
    def normalize_optional_address(cls, value: Optional[str]) -> Optional[str]:
        return value.lower() if value else value

    @property
    def fee_rate(self) -> float:
        return self.fee_bps / 10_000.0


class SplitConfig(BaseModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_total(self) -> "SplitConfig":
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("train + validation + test must equal 1.0")
        return self


class RpcConfig(BaseModel):
    env_var: str = "ETH_RPC_URL"
    timeout_seconds: int = 30
    max_retries: int = 4
    retry_backoff_seconds: float = 1.5
    cache_dir: Path = Path("data/raw/rpc_cache")

    model_config = ConfigDict(frozen=True)


class GasConfig(BaseModel):
    gas_units: int = 220000
    priority_fee_gwei: float = 2.0

    model_config = ConfigDict(frozen=True)


class MarketConfig(BaseModel):
    market: str
    chain_id: int = 1
    sample_window_days: int = 120
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    block_chunk_size: int = 4000
    stale_after_minutes: int = 15
    arbitrage_notional_wbtc: float = 0.10
    raw_data_dir: Path = Path("data/raw")
    curated_data_dir: Path = Path("data/curated")
    report_dir: Path = Path("outputs/report")
    metadata_dir: Path = Path("outputs/metadata")
    splits: SplitConfig = Field(default_factory=SplitConfig)
    rpc: RpcConfig = Field(default_factory=RpcConfig)
    gas: GasConfig = Field(default_factory=GasConfig)
    tokens: dict[str, TokenConfig]
    dexes: list[DexConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_required_tokens(self) -> "MarketConfig":
        required = {"wbtc", "weth"}
        missing = required.difference(self.tokens)
        if missing:
            raise ValueError(f"missing token definitions: {sorted(missing)}")
        return self

    def resolve_window(self, now: Optional[datetime] = None) -> tuple[datetime, datetime]:
        current_time = now or datetime.now(UTC)
        end = self.end_date.astimezone(UTC) if self.end_date else current_time
        start = self.start_date.astimezone(UTC) if self.start_date else end - timedelta(days=self.sample_window_days)
        if start >= end:
            raise ValueError("start_date must be earlier than end_date")
        return start, end

    def ensure_directories(self) -> None:
        for path in [self.raw_data_dir, self.curated_data_dir, self.report_dir, self.metadata_dir, self.rpc.cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def rpc_url(self, explicit: Optional[str] = None) -> str:
        if explicit:
            return explicit
        value = os.getenv(self.rpc.env_var)
        if not value:
            raise ValueError(
                f"Ethereum RPC URL missing. Set {self.rpc.env_var} or pass --rpc-url explicitly."
            )
        return value


def load_market_config(path: str | Path = "config/markets.yaml") -> MarketConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    return MarketConfig.model_validate(raw)
