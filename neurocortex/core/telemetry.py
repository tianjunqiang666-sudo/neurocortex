"""
NeuroCortex AI — Telemetry (可观测性)
=======================================
轻量级性能追踪模块。
提供 @trace_latency 装饰器，自动记录每个模块的处理耗时。
追踪数据存入 data/telemetry.jsonl，支持 Dashboard 可视化。
"""

from __future__ import annotations

import json
import time
import functools
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class Telemetry:
    """全局性能追踪器

    Attributes:
        records: 内存中最近 N 条追踪记录
        storage_path: JSONL 持久化路径
    """

    _instance: Telemetry | None = None

    def __new__(cls, *args, **kwargs) -> Telemetry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_records: int = 200) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.storage_path = Path(__file__).parent.parent / "data" / "telemetry.jsonl"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._module_stats: dict[str, dict[str, Any]] = {}

    def record(self, module: str, function: str, latency_ms: float,
               success: bool = True, metadata: dict | None = None) -> None:
        """记录一次追踪事件"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "function": function,
            "latency_ms": round(latency_ms, 2),
            "success": success,
        }
        if metadata:
            entry["metadata"] = metadata

        self.records.append(entry)
        self._update_stats(module, latency_ms, success)

        # 追加写入 JSONL
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 追踪失败不应影响主流程

    def _update_stats(self, module: str, latency_ms: float, success: bool) -> None:
        """更新模块统计信息"""
        if module not in self._module_stats:
            self._module_stats[module] = {
                "call_count": 0, "total_ms": 0.0,
                "min_ms": float("inf"), "max_ms": 0.0,
                "errors": 0,
            }
        stats = self._module_stats[module]
        stats["call_count"] += 1
        stats["total_ms"] += latency_ms
        stats["min_ms"] = min(stats["min_ms"], latency_ms)
        stats["max_ms"] = max(stats["max_ms"], latency_ms)
        if not success:
            stats["errors"] += 1

    def get_module_stats(self) -> dict[str, dict[str, Any]]:
        """获取各模块统计摘要"""
        result = {}
        for module, stats in self._module_stats.items():
            count = stats["call_count"]
            result[module] = {
                "call_count": count,
                "avg_ms": round(stats["total_ms"] / count, 2) if count > 0 else 0,
                "min_ms": round(stats["min_ms"], 2) if stats["min_ms"] != float("inf") else 0,
                "max_ms": round(stats["max_ms"], 2),
                "errors": stats["errors"],
            }
        return result

    def get_recent(self, n: int = 20) -> list[dict[str, Any]]:
        """获取最近 N 条追踪记录"""
        return list(self.records)[-n:]


# === 全局单例 ===
_telemetry = Telemetry()


def trace_latency(module_name: str | None = None) -> Callable:
    """性能追踪装饰器

    用法:
        @trace_latency("FrontalLobe")
        def generate_response(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        mod = module_name or func.__qualname__.split(".")[0]

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                _telemetry.record(mod, func.__name__, elapsed, success)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                _telemetry.record(mod, func.__name__, elapsed, success)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def get_telemetry() -> Telemetry:
    """获取全局 Telemetry 单例"""
    return _telemetry
