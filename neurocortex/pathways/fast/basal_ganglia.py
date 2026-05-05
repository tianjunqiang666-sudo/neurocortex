"""
NeuroCortex AI — BasalGanglia (基底节)
========================================
快速通路：习惯学习与执行、奖励预测。
匹配当前上下文与已学得策略，高置信度直接输出动作，绕过皮层。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class Action:
    """自动化动作指令"""

    def __init__(self, action_type: str, content: str, confidence: float) -> None:
        self.action_type = action_type
        self.content = content
        self.confidence = confidence

    def to_dict(self) -> dict[str, Any]:
        return {"action_type": self.action_type, "content": self.content,
                "confidence": self.confidence}

    def __repr__(self) -> str:
        return f"Action({self.action_type}, confidence={self.confidence:.2f})"


class BasalGanglia:
    """基底节 — 习惯匹配与自动化执行

    维护"刺激-反应"习惯库，高置信度匹配时直接输出动作。

    Attributes:
        habits: 习惯库 {pattern_key: Action}
        match_threshold: 匹配置信度阈值
    """

    def __init__(self, habits_path: str | Path | None = None,
                 match_threshold: float = 0.85) -> None:
        self.match_threshold = match_threshold
        self.habits: dict[str, dict[str, Any]] = {}
        self._habits_path = Path(habits_path) if habits_path else None

        if self._habits_path and self._habits_path.exists():
            self._load_habits()

    def habit_lookup(self, state: dict[str, Any]) -> Optional[Action]:
        """查找是否有匹配的习惯

        Args:
            state: 当前状态描述

        Returns:
            匹配的 Action，或 None（放行至皮层）
        """
        if not self.habits:
            return None  # 无习惯，放行

        state_key = self._state_to_key(state)

        for pattern, habit_data in self.habits.items():
            if pattern in state_key:
                confidence = habit_data.get("confidence", 0.0)
                if confidence >= self.match_threshold:
                    action = Action(
                        action_type=habit_data["action_type"],
                        content=habit_data["content"],
                        confidence=confidence,
                    )
                    logger.info(f"基底节习惯匹配: {pattern} → {action}")
                    return action

        logger.debug("基底节未匹配到习惯，放行至皮层")
        return None

    def add_habit(self, pattern: str, action_type: str, content: str,
                  confidence: float = 0.9) -> None:
        """添加或更新习惯

        Args:
            pattern: 匹配模式 (关键词或状态描述)
            action_type: 动作类型
            content: 动作内容
            confidence: 置信度
        """
        self.habits[pattern] = {
            "action_type": action_type, "content": content,
            "confidence": confidence,
        }
        logger.info(f"基底节添加习惯: '{pattern}' → {action_type}")
        self._save_habits()

    def _state_to_key(self, state: dict[str, Any]) -> str:
        """将状态字典转为匹配键"""
        return json.dumps(state, ensure_ascii=False, sort_keys=True).lower()

    def _load_habits(self) -> None:
        try:
            with open(self._habits_path, "r", encoding="utf-8") as f:
                self.habits = json.load(f)
            logger.info(f"加载 {len(self.habits)} 条习惯")
        except Exception as e:
            logger.warning(f"习惯加载失败: {e}")

    def _save_habits(self) -> None:
        if self._habits_path:
            self._habits_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._habits_path, "w", encoding="utf-8") as f:
                json.dump(self.habits, f, ensure_ascii=False, indent=2)
