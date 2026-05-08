"""
NeuroCortex AI — MemoryConsolidation (记忆巩固)
=================================================
模拟睡眠期知识蒸馏。
严格遵循知识蒸馏流程，严禁任何形式的 LLM 参数微调。

巩固流程:
  1. 从海马体采样 importance_score > 0.8 的事件
  2. extract_abstraction() 提取 if-then 规则
  3. 更新知识图谱
  4. 存入长期语义向量库
  5. 可选转化习惯到基底节
  6. 修剪低重要性记忆
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter
from neurocortex.memory.hippocampus import Hippocampus
from neurocortex.memory.knowledge_base import KnowledgeBase
from neurocortex.pathways.fast.basal_ganglia import BasalGanglia
from neurocortex.memory.association_engine import AssociationEngine


class MemoryConsolidation:
    """记忆巩固 — 知识蒸馏 (严禁微调)

    ⚠ 本模块不得出现 model.train()、LoRA、PEFT 等参数更新代码。

    流程:
      海马体重播 → LLM 提取规则 → 更新知识图谱 → 存入向量库

    Attributes:
        router: ModelRouter (用于调用 LLM 提取规则)
        hippocampus: 海马体实例
        knowledge_base: 知识库实例
        importance_threshold: 巩固所需的最低重要性
    """

    # extract_abstraction 的 prompt 模板
    EXTRACTION_PROMPT = """你是一个知识蒸馏系统。分析以下对话/事件记录，从中提取一条实用的 if-then 业务规则或经验法则。

事件记录：
{event_text}

要求：
1. 提取出一条简洁的 if-then 规则
2. 规则格式：IF <条件> THEN <建议行动>
3. 只输出规则本身，不要额外解释
4. 如果无法提取有意义的规则，输出：NO_RULE

示例输出：
IF 用户情绪=愤怒 AND 提及"退款失败" THEN 建议触发安抚流程并转接人工客服"""

    # 提取习惯的 prompt
    HABIT_EXTRACTION_PROMPT = """分析以下对话记录，提取出其中最简单的“命令-响应”固定模式。
这种模式应该是非常基础且通用的，例如简单的问候、开关指令、或特定的短语回复。

对话记录：
{event_text}

要求：
1. 提取出一个简单的正则表达式模式（用于匹配输入）和一个固定的回复字符串。
2. 规则格式：PATTERN: <正则模式> | RESPONSE: <固定回复>
3. 如果无法提取出这种简单的反射规则，输出：NO_HABIT
4. 只输出这一行，不要额外解释

示例输出：
PATTERN: ^(关灯|关闭灯光)$ | RESPONSE: 好的，灯已关。"""

    def __init__(self, router: ModelRouter, hippocampus: Hippocampus,
                 knowledge_base: KnowledgeBase,
                 basal_ganglia: BasalGanglia,
                 importance_threshold: float = 0.8) -> None:
        self.router = router
        self.hippocampus = hippocampus
        self.knowledge_base = knowledge_base
        self.basal_ganglia = basal_ganglia
        self.importance_threshold = importance_threshold
        self.association_engine = AssociationEngine(router, knowledge_base)

        logger.info(f"MemoryConsolidation 初始化 (阈值={importance_threshold})")

    async def run_consolidation_cycle(self) -> dict[str, Any]:
        """执行一次完整的巩固周期 (异步)

        Returns:
            巩固结果摘要
        """
        logger.info("════════ 睡眠巩固开始 ════════")
        results = {
            "sampled_memories": 0,
            "rules_extracted": 0,
            "rules_added": [],
            "memories_pruned": 0,
            "errors": [],
        }

        # 1. 采样高重要性记忆
        important_memories = self.hippocampus.get_important_memories(
            threshold=self.importance_threshold
        )
        results["sampled_memories"] = len(important_memories)
        logger.info(f"步骤1: 采样到 {len(important_memories)} 条高重要性记忆")

        if not important_memories:
            logger.info("无高重要性记忆，跳过巩固")
            logger.info("════════ 睡眠巩固结束 ════════")
            return results

        # 2. 逐条提取抽象规则
        for memory in important_memories:
            try:
                event_text = memory.get("document", "")
                if not event_text:
                    continue

                import asyncio
                rule = await asyncio.to_thread(self.extract_abstraction, event_text)
                if rule and rule != "NO_RULE":
                    # 3. 写入知识图谱
                    rule_id = self.knowledge_base.add_rule(
                        rule_text=rule,
                        source_episode_id=memory.get("id", ""),
                        rule_type="if-then",
                        confidence=0.7,
                    )
                    results["rules_extracted"] += 1
                    results["rules_added"].append({"id": rule_id, "rule": rule[:100]})
                    logger.info(f"步骤2-3: 提取并存储规则: {rule[:80]}")

                    # 3.1 尝试提取习惯并存入基底节
                    habit_data = await asyncio.to_thread(self.extract_habit, event_text)
                    if habit_data and habit_data != "NO_HABIT":
                        # 解析 PATTERN: <正则> | RESPONSE: <回复>
                        try:
                            parts = habit_data.split("|")
                            pattern = parts[0].replace("PATTERN:", "").strip()
                            response = parts[1].replace("RESPONSE:", "").strip()
                            self.basal_ganglia.add_habit(pattern, response, desc="自动蒸馏习惯")
                        except Exception as e:
                            logger.error(f"解析习惯数据失败: {habit_data}, err: {e}")

            except Exception as e:
                error_msg = f"记忆 {memory.get('id', '?')[:8]} 处理失败: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        # 4. 修剪低重要性记忆
        self.hippocampus.prune_low_importance(threshold=0.3)
        results["memories_pruned"] = 1 # 简化逻辑
        logger.info("步骤4: 修剪低重要性记忆完成")

        # 5. 记忆联想 (Phase 4 新增)
        await self.association_engine.run_association_cycle()

        logger.info("════════ 睡眠巩固结束 ════════")
        return results

    def extract_abstraction(self, event_text: str) -> str:
        """从事件中提取抽象规则

        伪代码:
          接收一个事件 JSON/文本，使用 heuristics (如基于模板的规则提取)
          或调用一个已配置的 LLM，提示词为"从以下事件中提取一条 if-then 规则..."，
          输出纯文本规则。

        Args:
            event_text: 事件的文本描述

        Returns:
            提取的规则文本，或 "NO_RULE"
        """
        if not event_text.strip():
            return "NO_RULE"

        try:
            client = self.router.get_client("hippocampus-consolidation")
            prompt = self.EXTRACTION_PROMPT.format(event_text=event_text)

            response = client.chat(
                prompt=prompt,
                system_prompt="你是一个精确的规则提取系统。只输出规则，不要解释。",
                temperature=0.3,
                max_tokens=200,
            )

            # 清理 deepseek-r1 的 <think> 标签
            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

            # 验证输出
            if not response or len(response) < 5:
                return "NO_RULE"
            if "NO_RULE" in response.upper():
                return "NO_RULE"

            return response.strip()

        except Exception as e:
            logger.error(f"规则提取失败: {e}")
            return "NO_RULE"

    def extract_habit(self, event_text: str) -> str:
        """从事件中提取简单的习惯反射"""
        if not event_text.strip():
            return "NO_HABIT"

        try:
            client = self.router.get_client("hippocampus-consolidation")
            prompt = self.HABIT_EXTRACTION_PROMPT.format(event_text=event_text)

            response = client.chat(
                prompt=prompt,
                system_prompt="你是一个精准的习惯提取系统。只输出 PATTERN|RESPONSE 格式，不要解释。",
                temperature=0.1,  # 习惯提取需要极高的一致性
                max_tokens=150,
            )

            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

            if "NO_HABIT" in response.upper() or "|" not in response:
                return "NO_HABIT"

            return response.strip()

        except Exception as e:
            logger.error(f"习惯提取失败: {e}")
            return "NO_HABIT"
