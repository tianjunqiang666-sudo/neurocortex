"""
Tests — FrontalLobe (额叶)
============================
测试负反馈检测和纠错逻辑。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeedbackDetection:
    """负反馈自然语言检测"""

    def test_negative_feedback_chinese(self, frontal_lobe):
        assert frontal_lobe.analyze_feedback("你记错了") > 0.5

    def test_negative_feedback_error(self, frontal_lobe):
        assert frontal_lobe.analyze_feedback("这是错误的") > 0.5

    def test_negative_feedback_english(self, frontal_lobe):
        assert frontal_lobe.analyze_feedback("that is wrong") > 0.5

    def test_neutral_input(self, frontal_lobe):
        assert frontal_lobe.analyze_feedback("今天天气真好") == 0.0

    def test_positive_input(self, frontal_lobe):
        assert frontal_lobe.analyze_feedback("说得好，谢谢") == 0.0


class TestCorrection:
    """纠错逻辑测试"""

    def test_correction_with_rules(self, frontal_lobe, knowledge_base, basal_ganglia):
        """有规则时应降低最近规则置信度"""
        knowledge_base.add_rule("some wrong rule", confidence=0.8)
        result = frontal_lobe.trigger_correction(None, knowledge_base, basal_ganglia)
        assert "降低" in result or "改进" in result

    def test_correction_without_rules(self, frontal_lobe, knowledge_base, basal_ganglia):
        """无规则时应返回兜底消息"""
        result = frontal_lobe.trigger_correction(None, knowledge_base, basal_ganglia)
        assert "警惕" in result or "问题" in result


class TestWorkingMemory:
    """工作记忆测试"""

    def test_add_turn(self, frontal_lobe):
        frontal_lobe.working_memory.add_turn("user", "test message")
        assert len(frontal_lobe.working_memory.conversation_history) == 1

    def test_max_history(self, frontal_lobe):
        for i in range(25):
            frontal_lobe.working_memory.add_turn("user", f"msg {i}")
        assert len(frontal_lobe.working_memory.conversation_history) <= 20
