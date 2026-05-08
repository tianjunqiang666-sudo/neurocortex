"""
NeuroCortex AI — Adaptive Params (自适应调节)
==============================================
自适应参数管理器，基于滑动窗口和历史数据动态计算系统参数（如预测误差阈值）。
替代了原先的硬编码阈值，使系统具备“自适应环境”的能力。
"""

import json
from collections import deque
from pathlib import Path
from typing import Any

from loguru import logger

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_ADAPTIVE_FILE = _DATA_DIR / "adaptive_state.json"


class AdaptiveThreshold:
    """滑动窗口自适应阈值

    公式: Threshold = mean + k * std_dev
    """

    def __init__(self, name: str, window_size: int = 50, k: float = 1.5, default_val: float = 0.6):
        self.name = name
        self.window_size = window_size
        self.k = k
        self.default_val = default_val
        self.history: deque[float] = deque(maxlen=window_size)

    def record(self, value: float) -> None:
        """记录一个新观测值"""
        self.history.append(value)

    def get_threshold(self) -> float:
        """计算当前自适应阈值"""
        if not self.history:
            return self.default_val

        n = len(self.history)
        mean = sum(self.history) / n

        if n < 2:
            return mean + self.k * 0.1  # 给个基础方差估计

        variance = sum((x - mean) ** 2 for x in self.history) / (n - 1)
        std_dev = variance ** 0.5

        # 限制阈值范围，避免跑飞
        threshold = mean + self.k * std_dev
        return max(0.1, min(0.9, threshold))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "window_size": self.window_size,
            "k": self.k,
            "default_val": self.default_val,
            "history": list(self.history)
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        self.window_size = data.get("window_size", self.window_size)
        self.k = data.get("k", self.k)
        self.default_val = data.get("default_val", self.default_val)
        self.history = deque(data.get("history", []), maxlen=self.window_size)


class AdaptiveManager:
    """自适应参数管理器 (单例)"""
    _instance: 'AdaptiveManager | None' = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.params: dict[str, AdaptiveThreshold] = {
            "prediction_error": AdaptiveThreshold("prediction_error", window_size=50, k=1.5, default_val=0.6),
            "feedback_sensitivity": AdaptiveThreshold("feedback_sensitivity", window_size=30, k=1.0, default_val=0.5)
        }
        self.load()

    def load(self) -> None:
        if not _ADAPTIVE_FILE.exists():
            return
        try:
            with open(_ADAPTIVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, param_data in data.items():
                if name in self.params:
                    self.params[name].from_dict(param_data)
        except Exception as e:
            logger.error(f"加载自适应参数失败: {e}")

    def save(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = {name: param.to_dict() for name, param in self.params.items()}
            with open(_ADAPTIVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"保存自适应参数失败: {e}")

    def record_error(self, error_val: float) -> None:
        self.params["prediction_error"].record(error_val)
        self.save()

    def get_error_threshold(self) -> float:
        return self.params["prediction_error"].get_threshold()


# 全局单例
_adaptive_manager = AdaptiveManager()

def get_adaptive_manager() -> AdaptiveManager:
    return _adaptive_manager
