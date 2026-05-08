"""
NeuroCortex AI — BasalGanglia (基底节)
======================================
快速通路：习惯与条件反射存储。
维护“刺激-反应”习惯库。对于高频重复的模式，系统会形成肌肉记忆直接响应，绕过慢速皮层。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from loguru import logger

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HABITS_FILE = _DATA_DIR / "habits.json"


class BasalGanglia:
    """基底节 — 习惯反射匹配
    
    职责:
      - 维护独立的习惯库 (habits.json)
      - 提供毫秒级的模式匹配
      - 若匹配成功，输出固定的动作指令，直接短路慢速系统
    """

    def __init__(self, habits_path: Optional[Path | str] = None) -> None:
        self.habits: list[dict[str, Any]] = []
        if habits_path:
            self._habits_file = Path(habits_path)
        else:
            self._habits_file = _DATA_DIR / "habits.json"
        self._load_habits()

    def _load_habits(self) -> None:
        """加载习惯库"""
        self._habits_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._habits_file.exists():
            # 预置一些默认习惯测试
            default_habits = [
                {
                    "id": "habit_01",
                    "trigger_pattern": r"^(你好|hello|hi)[!！]?$",
                    "response": "你好！我是 NeuroCortex AI，准备就绪。",
                    "description": "基础问候反射"
                },
                {
                    "id": "habit_02",
                    "trigger_pattern": r"(退下|闭嘴|安静)",
                    "response": "好的，我进入静默模式。",
                    "description": "紧急静默反射"
                }
            ]
            with open(self._habits_file, "w", encoding="utf-8") as f:
                json.dump(default_habits, f, indent=2, ensure_ascii=False)
            self.habits = default_habits
            logger.info("初始化了默认习惯库")
        else:
            try:
                with open(self._habits_file, "r", encoding="utf-8") as f:
                    self.habits = json.load(f)
                logger.info(f"成功加载 {len(self.habits)} 个习惯规则")
            except Exception as e:
                logger.error(f"加载习惯库失败: {e}")
                self.habits = []

    def save_habits(self) -> None:
        """保存习惯库"""
        try:
            with open(self._habits_file, "w", encoding="utf-8") as f:
                json.dump(self.habits, f, indent=2, ensure_ascii=False)
            logger.debug("习惯库已同步至磁盘")
        except Exception as e:
            logger.error(f"保存习惯库失败: {e}")

    def add_habit(self, pattern: str, response: str, desc: str = "") -> None:
        """添加新习惯（由睡眠巩固提取出时调用）"""
        habit_id = f"habit_{len(self.habits) + 1:03d}"
        new_habit = {
            "id": habit_id,
            "trigger_pattern": pattern,
            "response": response,
            "description": desc
        }
        self.habits.append(new_habit)
        self.save_habits()
        logger.info(f"基底节习得了新习惯: {desc} -> {response}")

    def lookup(self, packet: Any) -> Optional[str]:
        """查表匹配习惯反射
        
        Args:
            packet: SensoryPacket (包含 raw_data 和 modality)
            
        Returns:
            若匹配，返回包含 'response' 和 'habit_id' 的字典；否则返回 None
        """
        if packet.modality != "text":
            return None  # 目前只针对文本做正则反射
            
        text = str(packet.raw_data).strip()
        logger.info(f"▶ BasalGanglia 正在查表: input='{text}'")
        
        for habit in self.habits:
            pattern = habit.get("trigger_pattern", "")
            if not pattern:
                continue
                
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.success(f"⚡ 基底节习惯触发! 匹配规则: {habit.get('id')} - {habit.get('description')}")
                    return {"habit_id": habit.get("id"), "response": habit.get("response")}
            except re.error as e:
                logger.error(f"习惯正则表达式错误 ({pattern}): {e}")
                
        return None
