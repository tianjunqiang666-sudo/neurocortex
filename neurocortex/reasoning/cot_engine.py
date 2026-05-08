"""
NeuroCortex AI — Chain-of-Thought (CoT) Engine
================================================
思维链推理引擎，当遇到复杂问题时，将任务分解、逐步执行并融合工具调用。
"""

import json
from typing import Any, List, Dict
from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter, LLMClient
from neurocortex.core.event_bus import get_event_bus, Event, EventType
from neurocortex.tools.registry import registry
import neurocortex.tools.basic_tools  # noqa 触发工具注册


class CoTEngine:
    """思维链引擎"""

    def __init__(self, router: ModelRouter):
        self.router = router

    def _get_tools_schema_text(self) -> str:
        schemas = registry.get_all_schemas()
        return json.dumps(schemas, ensure_ascii=False, indent=2)

    async def execute(self, user_input: str, context: str) -> str:
        """执行完整推理链
        
        1. Decompose (如果足够复杂则分解)
        2. Execute Steps (逐步调用并尝试触发外部工具)
        3. Synthesize (总结归纳)
        """
        event_bus = get_event_bus()
        event_bus.publish(Event(EventType.REASONING_START, "CoTEngine", {"task": "decompose"}))
        
        # 1. 任务分解
        steps = await self._decompose_task(user_input, context)
        if not steps or len(steps) == 1:
            # 不用分解，直接当单步执行
            logger.info("CoTEngine: 任务无需复杂分解，执行单步推理。")
            steps = [user_input]
        else:
            logger.info(f"CoTEngine: 任务分解为 {len(steps)} 步: {steps}")
            
        # 2. 逐步执行
        step_results: List[Dict[str, str]] = []
        for i, step in enumerate(steps, 1):
            event_bus.publish(Event(EventType.REASONING_START, "CoTEngine", {"task": f"step_{i}", "desc": step}))
            
            result = await self._execute_step(step, step_results, context)
            step_results.append({
                "step": step,
                "result": result
            })
            
            # 广播中间步骤，供 UI 或日志展示
            event_bus.publish(Event(EventType.TOKEN_GENERATED, "CoTEngine", {"token": f"\n\n[Step {i}] {step}\n-> {result}\n"}))
            
        # 3. 最终汇总
        if len(steps) == 1:
            return step_results[0]["result"]
            
        event_bus.publish(Event(EventType.REASONING_START, "CoTEngine", {"task": "synthesize"}))
        final_answer = await self._synthesize(user_input, step_results)
        return final_answer

    async def _decompose_task(self, user_input: str, context: str) -> List[str]:
        """将复杂任务分解为子步骤"""
        client = self.router.get_client("frontal")
        prompt = (
            f"上下文信息:\n{context}\n\n"
            f"目标任务: {user_input}\n\n"
            "请判断该任务是否需要分解为多个子步骤。如果非常简单，直接返回一个步骤。如果较复杂，请分解为连续的子步骤。\n"
            "请以 JSON 格式输出，格式如下:\n"
            "{\n  \"steps\": [\"第一步描述\", \"第二步描述\", ...]\n}"
        )
        try:
            import asyncio
            response = await asyncio.to_thread(
                client.chat, prompt=prompt, temperature=0.1, json_mode=True
            )
            data = json.loads(response)
            return data.get("steps", [user_input])
        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            return [user_input]

    async def _execute_step(self, step: str, previous_results: List[Dict[str, str]], context: str) -> str:
        """执行单一子步骤，支持工具调用检查"""
        client = self.router.get_client("frontal")
        
        history_text = "\n".join([f"步骤 {i+1}: {r['step']}\n结果: {r['result']}" for i, r in enumerate(previous_results)])
        
        system_prompt = (
            "你正在执行一个子任务。你可以使用外部工具来辅助完成任务。\n"
            f"可用的外部工具 Schema:\n{self._get_tools_schema_text()}\n\n"
            "【如何使用工具】:\n"
            "如果你需要调用工具，你的回答必须 EXACTLY 按照这个 JSON 格式，不要包含其他解释文本:\n"
            "{\n  \"tool_call\": \"工具名\",\n  \"arguments\": {\"参数1\": \"值1\"}\n}\n"
            "如果不需要工具，直接输出自然语言结果。"
        )
        
        prompt = (
            f"当前情境:\n{context}\n\n"
            f"之前的步骤执行结果:\n{history_text}\n\n"
            f"当前需要执行的步骤: {step}\n\n"
            "请给出执行结果，或调用相关工具。"
        )

        try:
            import asyncio
            response = await asyncio.to_thread(
                client.chat, prompt=prompt, system_prompt=system_prompt, temperature=0.2
            )
            
            # 尝试解析是否为工具调用
            try:
                # 为了防止 LLM 在 JSON 外面包 ```json
                clean_resp = response.strip()
                if clean_resp.startswith("```json"):
                    clean_resp = clean_resp[7:]
                if clean_resp.endswith("```"):
                    clean_resp = clean_resp[:-3]
                
                tool_data = json.loads(clean_resp)
                if "tool_call" in tool_data and "arguments" in tool_data:
                    tool_name = tool_data["tool_call"]
                    tool_args = tool_data["arguments"]
                    logger.info(f"LLM 决定调用工具: {tool_name} with {tool_args}")
                    
                    # 执行外部工具
                    tool_result = registry.execute(tool_name, **tool_args)
                    
                    # 将工具结果喂给 LLM，让其产生本步骤的最终解释
                    followup_prompt = (
                        f"你调用了工具 {tool_name}，参数为 {tool_args}。\n"
                        f"工具返回结果为:\n{tool_result}\n\n"
                        f"请根据这个结果，简要总结【当前步骤: {step}】的结论。"
                    )
                    final_response = await asyncio.to_thread(
                        client.chat, prompt=followup_prompt, temperature=0.2
                    )
                    return final_response
                    
            except json.JSONDecodeError:
                # 不是合法的 JSON 工具调用，就是普通文本回复
                pass
                
            return response
            
        except Exception as e:
            logger.error(f"执行步骤失败: {e}")
            return f"执行失败: {e}"

    async def _synthesize(self, task: str, step_results: List[Dict[str, str]]) -> str:
        """汇总所有步骤的结果并输出最终回复"""
        client = self.router.get_client("frontal")
        
        history_text = "\n".join([f"步骤 {i+1}: {r['step']}\n结论: {r['result']}" for i, r in enumerate(step_results)])
        
        prompt = (
            f"用户原始目标: {task}\n\n"
            f"我们经过了以下几个思考步骤:\n{history_text}\n\n"
            "请根据上述思考过程，直接向用户提供一个连贯、友好的最终回答。不要重复复述每一步，提取精要即可。"
        )
        
        try:
            import asyncio
            response = await asyncio.to_thread(
                client.chat, prompt=prompt, temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"最终汇总失败: {e}")
            return f"推导过程已完成，但在总结时遇到异常: {e}"
