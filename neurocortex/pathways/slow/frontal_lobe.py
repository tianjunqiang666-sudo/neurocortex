"""
NeuroCortex AI — FrontalLobe (额叶)
======================================
慢速通路：中央执行系统 — 工作记忆、规划、决策、语言输出。
维护任务状态，调用 LLM 进行推理，支持状态快照与回滚。
"""

from __future__ import annotations

import copy
import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter
from neurocortex.core.event_bus import get_event_bus, Event, EventType
from neurocortex.reasoning.cot_engine import CoTEngine
from neurocortex.reasoning.critique_module import CritiqueModule


class WorkingMemory:
    """工作记忆 — 维护当前对话和任务上下文

    Attributes:
        conversation_history: 对话历史
        task_stack: 当前任务栈
        variables: 临时变量存储
    """

    def __init__(self, max_history: int = 20) -> None:
        self.conversation_history: deque[dict[str, str]] = deque(maxlen=max_history)
        self.task_stack: list[dict[str, Any]] = []
        self.variables: dict[str, Any] = {}
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    def add_turn(self, role: str, content: str) -> None:
        """添加一轮对话"""
        self.conversation_history.append({
            "role": role, "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def push_task(self, task: dict[str, Any]) -> None:
        """压入子任务"""
        self.task_stack.append(task)

    def pop_task(self) -> Optional[dict[str, Any]]:
        """弹出当前任务"""
        return self.task_stack.pop() if self.task_stack else None

    def get_context_window(self, n: int = 10) -> list[dict[str, str]]:
        """获取最近 n 轮对话"""
        return list(self.conversation_history)[-n:]

    def serialize(self) -> dict[str, Any]:
        """序列化为可保存格式（用于快照）"""
        return {
            "conversation_history": list(self.conversation_history),
            "task_stack": copy.deepcopy(self.task_stack),
            "variables": copy.deepcopy(self.variables),
            "created_at": self.created_at,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "WorkingMemory":
        """从序列化数据恢复"""
        wm = cls()
        for turn in data.get("conversation_history", []):
            wm.conversation_history.append(turn)
        wm.task_stack = data.get("task_stack", [])
        wm.variables = data.get("variables", {})
        wm.created_at = data.get("created_at", wm.created_at)
        return wm


class FrontalLobe:
    """额叶 — 中央执行系统

    职责:
      - 工作记忆管理
      - 规划与任务分解 (调用 LLM)
      - 生成最终回复
      - 支持快照回滚 (由 Brainstem 中断触发)
      - 抽象层误差监控

    Attributes:
        router: ModelRouter 实例
        working_memory: 当前工作记忆
        snapshots: 工作记忆快照栈
    """

    required_capabilities = {
        "reasoning_depth": "high",
        "output_schema": {"type": "string"},
    }

    SYSTEM_PROMPT = """你是 NeuroCortex AI 的额叶 (Frontal Lobe) 认知核心。

你的职责是：
1. 基于当前对话上下文和检索到的历史记忆，给出有深度、有帮助的回复。
2. 在回复中体现对用户意图的理解。
3. 如果检索到了相关历史记忆或规则，将其自然地融入回答。
4. 保持对话连贯性和友好性。

请用中文回复。"""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router
        self.working_memory = WorkingMemory()
        self.snapshots: list[dict[str, Any]] = []
        self.cot_engine = CoTEngine(router)
        self.critique_module = CritiqueModule(router)

        logger.info("FrontalLobe 初始化完成")

    # ── 核心推理 ──────────────────────────────────────

    def generate_response(
        self,
        user_input: str,
        episode: Any = None,
        memories: list[dict[str, Any]] | None = None,
        knowledge_rules: list[str] | None = None,
    ) -> str:
        """生成最终回复

        Args:
            user_input: 用户输入文本
            episode: ParietalLobe 输出的 EpisodeTensor
            memories: 海马体检索的相关记忆
            knowledge_rules: 知识库中的相关规则

        Returns:
            LLM 生成的回复文本
        """
        # 记录用户输入
        self.working_memory.add_turn("user", user_input)

        # 构建增强 prompt
        prompt = self._build_prompt(user_input, episode, memories, knowledge_rules)

        try:
            client = self.router.get_client(
                "frontal",
                required_capabilities=self.required_capabilities,
            )
            response = client.chat(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=2048,
            )

            # 清理 deepseek-r1 可能输出的 <think>...</think> 标签
            response = self._clean_response(response)

            # 记录回复
            self.working_memory.add_turn("assistant", response)

            logger.debug(f"FrontalLobe 生成回复: {response[:80]}...")
            return response

        except Exception as e:
            logger.error(f"FrontalLobe 生成回复失败: {e}")
            error_msg = f"抱歉，我的额叶认知核心遇到了暂时的故障: {str(e)[:100]}"
            self.working_memory.add_turn("assistant", error_msg)
            return error_msg

    async def generate_response_stream(
        self,
        user_input: str,
        episode: Any = None,
        memories: list[dict[str, Any]] | None = None,
        knowledge_rules: list[str] | None = None,
    ) -> Any:
        """流式生成回复 (返回 AsyncGenerator)"""
        self.working_memory.add_turn("user", user_input)
        prompt = self._build_prompt(user_input, episode, memories, knowledge_rules)
        event_bus = get_event_bus()

        # 简单判断是否启用 CoT：包含关键词或长度超过一定阈值
        cot_triggers = ["如何", "为什么", "比较", "计算", "分析", "步骤", "how", "why", "compare", "calculate"]
        use_cot = any(trigger in user_input for trigger in cot_triggers) or len(user_input) > 50

        try:
            if use_cot:
                logger.info("▶ FrontalLobe 启用 CoT 推理引擎...")
                event_bus.publish(Event(EventType.REASONING_START, "FrontalLobe"))
                # 由于 CoTEngine 目前不支持流式 yield token，我们在这里执行完后模拟输出，
                # 或者后续重构 CoTEngine 支持 yield。目前先执行并返回。
                context = f"情境描述: {episode.to_text() if episode else '无'}\n相关记忆: {memories}\n相关规则: {knowledge_rules}"
                response = await self.cot_engine.execute(user_input, context)
                
                # CoT 后的反思
                passed, score, verified_response = await self.critique_module.critique(
                    user_input, response, context
                )
                if not passed:
                    logger.warning(f"CoT 反思修正: {response[:30]}... -> {verified_response[:30]}...")
                    response = verified_response

                # 记录并模拟流式输出（因为 CoTEngine 内部已经发过 token 事件了，但 generator 还需要 yield）
                # 这里我们假设 CoTEngine 已经在内部发了中间步骤的 token，最后我们 yield 最终答案
                self.working_memory.add_turn("assistant", response)
                yield response
                event_bus.publish(Event(EventType.RESPONSE_COMPLETE, "FrontalLobe", {
                    "response": response,
                    "critique_score": score
                }))
            else:
                client = self.router.get_client("frontal", required_capabilities=self.required_capabilities)
                event_bus.publish(Event(EventType.REASONING_START, "FrontalLobe"))
                
                full_response = []
                async for chunk in client.chat_stream(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    temperature=0.7,
                    max_tokens=2048,
                ):
                    full_response.append(chunk)
                    event_bus.publish(Event(EventType.TOKEN_GENERATED, "FrontalLobe", {"token": chunk}))
                    yield chunk
                    
                final_text = "".join(full_response)
                final_text = self._clean_response(final_text)

                # 自我反思机制
                context_summary = f"记忆: {memories}\n规则: {knowledge_rules}"
                passed, score, verified_response = await self.critique_module.critique(
                    user_input, final_text, context_summary
                )
                
                if not passed:
                    logger.warning(f"反思修正: {final_text[:30]}... -> {verified_response[:30]}...")
                    final_text = verified_response
                    # 向用户提示发生了修正 (可选)
                    # yield "\n\n[自我反思: 修正了回复中的逻辑偏差]"

                self.working_memory.add_turn("assistant", final_text)
                event_bus.publish(Event(EventType.RESPONSE_COMPLETE, "FrontalLobe", {
                    "response": final_text,
                    "critique_score": score
                }))
            
        except Exception as e:
            logger.error(f"FrontalLobe 流式生成失败: {e}")
            error_msg = f"抱歉，故障: {str(e)[:100]}"
            self.working_memory.add_turn("assistant", error_msg)
            yield error_msg

    def analyze_feedback(self, user_input: str) -> float:
        """分析用户输入是否包含负面反馈 (0.0 ~ 1.0)"""
        # 简单关键字匹配，后续可升级为 LLM 分类
        negative_signals = ["不对", "错误", "记错了", "并不是", "瞎说", "胡扯", "wrong", "false", "not true"]
        matches = [s for s in negative_signals if s in user_input.lower()]
        
        if matches:
            logger.warning(f"检测到潜在负面反馈: {matches}")
            return 0.8 # 高置信度负面反馈
        return 0.0

    def trigger_correction(self, hippocampus: Any, knowledge_base: Any, basal_ganglia: Any) -> str:
        """执行自我纠错逻辑"""
        logger.info("▶ FrontalLobe 执行纠错流程...")
        
        # 1. 撤回最近的海马体记忆 (设置为低重要性)
        # 获取最近 5 轮对话中产生的 Episode
        # 这里简化处理：降低知识库中最近添加的规则的置信度
        rules = knowledge_base.get_all_rules()
        if rules:
            latest_rule = rules[-1] # 假设最后一条是最近的
            knowledge_base.deprecate_rule(latest_rule["id"], penalty=0.4)
            return f"明白了，我已经降低了规则 '{latest_rule.get('rule_text', '')[:30]}...' 的可信度。我会努力改进我的记忆。"
            
        return "抱歉，我还没能确定哪条记忆出了问题，但我会保持警惕。"

    def plan(self, context: str, goal: str) -> list[str]:
        """任务分解与规划

        Args:
            context: 当前情境
            goal: 目标描述

        Returns:
            步骤列表
        """
        prompt = (
            f"当前情境：{context}\n"
            f"目标：{goal}\n\n"
            "请将目标分解为具体的执行步骤（每步一行，用数字编号）："
        )

        try:
            client = self.router.get_client("frontal")
            response = client.chat(prompt=prompt, temperature=0.3)
            response = self._clean_response(response)
            # 解析步骤
            steps = [
                line.strip() for line in response.split("\n")
                if line.strip() and any(c.isdigit() for c in line[:3])
            ]
            return steps if steps else [response]
        except Exception as e:
            logger.error(f"规划失败: {e}")
            return [f"无法生成规划: {e}"]

    # ── 快照与回滚 ────────────────────────────────────

    def rollback_state(self) -> dict[str, Any]:
        """保存当前工作记忆快照并返回

        由 Brainstem 紧急中断时调用。
        """
        snapshot = self.working_memory.serialize()
        self.snapshots.append(snapshot)
        logger.warning(f"FrontalLobe 快照已保存 (共 {len(self.snapshots)} 个)")
        return snapshot

    def restore_state(self, snapshot: dict[str, Any] | None = None) -> None:
        """从快照恢复工作记忆

        Args:
            snapshot: 指定快照，None 则使用最近的
        """
        if snapshot is None:
            if not self.snapshots:
                logger.warning("无可用快照，无法恢复")
                return
            snapshot = self.snapshots.pop()

        self.working_memory = WorkingMemory.deserialize(snapshot)
        logger.info("FrontalLobe 已从快照恢复")

    # ── 内部方法 ──────────────────────────────────────

    def _build_prompt(self, user_input: str, episode: Any,
                      memories: list[dict] | None,
                      rules: list[str] | None) -> str:
        """构建增强的 prompt"""
        parts = []

        # 对话上下文
        context = self.working_memory.get_context_window(6)
        if len(context) > 1:  # 排除刚添加的当前轮
            parts.append("【近期对话上下文】")
            for turn in context[:-1]:
                parts.append(f"  {turn['role']}: {turn['content'][:200]}")

        # 相关记忆
        if memories:
            parts.append("\n【海马体检索到的相关记忆】")
            for i, mem in enumerate(memories[:3], 1):
                doc = mem.get("document", mem.get("documents", ""))
                if isinstance(doc, list):
                    doc = doc[0] if doc else ""
                parts.append(f"  记忆{i}: {str(doc)[:200]}")

        # 知识规则
        if rules:
            parts.append("\n【知识库相关规则】")
            for rule in rules[:3]:
                parts.append(f"  • {rule}")

        # 情景信息
        if episode and hasattr(episode, 'to_text'):
            parts.append(f"\n【当前情景】{episode.to_text()[:200]}")
            if episode.prediction_error > 0.6:
                parts.append(f"  ⚠ 预测误差较高 ({episode.prediction_error:.2f})，可能是出乎意料的情况")

        # 当前输入
        parts.append(f"\n【用户输入】{user_input}")

        return "\n".join(parts)

    def _clean_response(self, response: str) -> str:
        """清理 LLM 输出中的特殊标签"""
        import re
        # 移除 deepseek-r1 的 <think>...</think> 标签
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def get_status(self) -> dict[str, Any]:
        """获取额叶状态"""
        return {
            "conversation_turns": len(self.working_memory.conversation_history),
            "active_tasks": len(self.working_memory.task_stack),
            "snapshots_count": len(self.snapshots),
            "variables": list(self.working_memory.variables.keys()),
        }
