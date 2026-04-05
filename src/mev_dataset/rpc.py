"""Low-level Ethereum JSON-RPC client with disk caching and retries."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import requests


class JsonRpcError(RuntimeError):
    """Raised when the upstream JSON-RPC server returns an error."""


@dataclass(frozen=True)
class BlockHeader:
    block_number: int
    block_hash: str
    parent_hash: str
    block_timestamp: datetime
    base_fee_per_gas_wei: int
    gas_used: int
    gas_limit: int


class RpcClient:
    def __init__(
        self,
        rpc_url: str,
        cache_dir: str | Path,
        timeout_seconds: int = 30,
        max_retries: int = 4,
        retry_backoff_seconds: float = 1.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.rpc_url = rpc_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

    @staticmethod
    def to_hex_quantity(value: int) -> str:
        return hex(int(value))

    def _cache_path(self, method: str, params: list[Any]) -> Path:
        payload = json.dumps([method, params], sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def call(self, method: str, params: list[Any], use_cache: bool = False) -> Any:
        cache_path = self._cache_path(method, params)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())["result"]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.rpc_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                if "error" in payload:
                    raise JsonRpcError(f"{method} failed: {payload['error']}")
                result = payload["result"]
                if use_cache:
                    cache_path.write_text(json.dumps({"result": result}))
                return result
            except Exception as exc:  # pragma: no cover - retry path is timing dependent
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise JsonRpcError(f"{method} failed after {self.max_retries} attempts: {last_error}")

    def get_latest_block_number(self) -> int:
        return int(self.call("eth_blockNumber", [], use_cache=False), 16)

    def eth_call(self, to: str, data: str, block: str = "latest", use_cache: bool = True) -> str:
        return self.call("eth_call", [{"to": to, "data": data}, block], use_cache=use_cache)

    def eth_get_logs(
        self,
        address: str,
        from_block: int,
        to_block: int,
        topics: Optional[list[Any]] = None,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        params = [{
            "address": address,
            "fromBlock": self.to_hex_quantity(from_block),
            "toBlock": self.to_hex_quantity(to_block),
        }]
        if topics is not None:
            params[0]["topics"] = topics
        return self.call("eth_getLogs", params, use_cache=use_cache)

    def get_block_by_number(self, block_number: int, use_cache: bool = True) -> BlockHeader:
        raw = self.call(
            "eth_getBlockByNumber",
            [self.to_hex_quantity(block_number), False],
            use_cache=use_cache,
        )
        return BlockHeader(
            block_number=int(raw["number"], 16),
            block_hash=raw["hash"],
            parent_hash=raw["parentHash"],
            block_timestamp=datetime.fromtimestamp(int(raw["timestamp"], 16), tz=UTC),
            base_fee_per_gas_wei=int(raw.get("baseFeePerGas", "0x0"), 16),
            gas_used=int(raw["gasUsed"], 16),
            gas_limit=int(raw["gasLimit"], 16),
        )

    def block_timestamp(self, block_number: int) -> datetime:
        return self.get_block_by_number(block_number, use_cache=True).block_timestamp

    def find_block_by_timestamp(self, target: datetime, direction: str = "after") -> int:
        if target.tzinfo is None:
            raise ValueError("target must be timezone-aware")
        latest = self.get_latest_block_number()
        earliest = 1
        earliest_ts = self.block_timestamp(earliest)
        latest_ts = self.block_timestamp(latest)
        if target <= earliest_ts:
            return earliest
        if target >= latest_ts:
            return latest

        lo = earliest
        hi = latest
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_ts = self.block_timestamp(mid)
            if mid_ts < target:
                lo = mid + 1
            elif mid_ts > target:
                hi = mid - 1
            else:
                return mid

        if direction == "after":
            return lo
        if direction == "before":
            return hi
        raise ValueError("direction must be 'after' or 'before'")
