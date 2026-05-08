"""
Tests — BasalGanglia (基底节)
==============================
测试习惯匹配、CRUD 操作和持久化。
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurocortex.core.thalamus import SensoryPacket


class TestBasalGangliaLookup:
    """习惯匹配逻辑测试"""

    def test_exact_match(self, basal_ganglia):
        """精确正则匹配应触发反射"""
        basal_ganglia.habits.clear()
        basal_ganglia.add_habit(r"^(你好|hello)$", "你好！测试通过。", desc="test_greeting")

        packet = SensoryPacket(
            modality="text", raw_data="你好",
            embedding=[0.1], salience=0.5, routing=["frontal"],
        )
        result = basal_ganglia.lookup(packet)
        assert result is not None
        assert result["response"] == "你好！测试通过。"

    def test_no_match(self, basal_ganglia):
        """不匹配的输入应返回 None"""
        basal_ganglia.add_habit(r"^退下$", "好的", desc="test_dismiss")

        packet = SensoryPacket(
            modality="text", raw_data="今天天气真好",
            embedding=[0.1], salience=0.5, routing=["frontal"],
        )
        result = basal_ganglia.lookup(packet)
        assert result is None

    def test_non_text_skipped(self, basal_ganglia):
        """非文本模态应跳过匹配"""
        basal_ganglia.add_habit(r".*", "catch_all", desc="test_all")

        packet = SensoryPacket(
            modality="visual", raw_data="image.png",
            embedding=[0.1], salience=0.5, routing=["occipital"],
        )
        result = basal_ganglia.lookup(packet)
        assert result is None


class TestBasalGangliaCRUD:
    """习惯增删查改测试"""

    def test_add_habit(self, basal_ganglia):
        basal_ganglia.habits.clear()
        basal_ganglia.add_habit(r"^test$", "test_response", desc="test")
        assert len(basal_ganglia.habits) == 1

    def test_persistence(self, basal_ganglia, tmp_data_dir):
        """习惯应持久化到文件"""
        basal_ganglia.habits.clear()
        basal_ganglia.add_habit(r"^persist$", "persisted!", desc="persist_test")

        # 重新加载
        from neurocortex.pathways.fast.basal_ganglia import BasalGanglia
        bg2 = BasalGanglia(habits_path=tmp_data_dir / "test_habits.json")
        assert len(bg2.habits) == 1
        assert bg2.habits[0]["response"] == "persisted!"

    def test_get_habits_list(self, basal_ganglia):
        basal_ganglia.habits.clear()
        basal_ganglia.add_habit(r"^a$", "A", desc="a")
        basal_ganglia.add_habit(r"^b$", "B", desc="b")
        assert len(basal_ganglia.habits) == 2
