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

    def __init__(self, router: ModelRouter, hippocampus: Hippocampus,
                 knowledge_base: KnowledgeBase,
                 importance_threshold: float = 0.8) -> None:
        self.router = router
        self.hippocampus = hippocampus
        self.knowledge_base = knowledge_base
        self.importance_threshold = importance_threshold

        logger.info(f"MemoryConsolidation 初始化 (阈值={importance_threshold})")

    def run_consolidation_cycle(self) -> dict[str, Any]:
        """执行一次完整的巩固周期

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

                rule = self.extract_abstraction(event_text)
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

            except Exception as e:
                error_msg = f"记忆 {memory.get('id', '?')[:8]} 处理失败: {e}"
                results["errors"].append(error_msg)
                logger.error(error_msg)

        # 4. 修剪低重要性记忆
        pruned = self.hippocampus.prune(importance_threshold=0.3)
        results["memories_pruned"] = pruned

        logger.info(
            f"════════ 睡眠巩固结束: "
            f"采样={results['sampled_memories']}, "
            f"规则={results['rules_extracted']}, "
            f"修剪={results['memories_pruned']} ════════"
        )

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
