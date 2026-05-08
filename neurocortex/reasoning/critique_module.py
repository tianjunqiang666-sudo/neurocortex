"""
NeuroCortex AI — Critique Module (自我反思)
===========================================
负责对额叶生成的候选回复进行自我审查，检查事实错误、逻辑矛盾或安全隐患。
"""

import json
from typing import Any, Dict, Tuple
from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter


class CritiqueModule:
    """反思模块"""

    def __init__(self, router: ModelRouter):
        self.router = router

    async def critique(self, user_input: str, response_candidate: str, context: str) -> Tuple[bool, str, str]:
        """
        对回复进行反思。
        返回: (is_passed, critique_score, feedback_or_final_response)
        """
        client = self.router.get_client("frontal")
        
        prompt = (
            f"作为 NeuroCortex AI 的自我反思系统，请审查以下对话片段：\n\n"
            f"【用户输入】: {user_input}\n"
            f"【上下文信息】: {context}\n"
            f"【AI 候选回复】: {response_candidate}\n\n"
            "请根据以下标准进行批判性审查：\n"
            "1. 事实一致性：回复是否与提供的上下文矛盾？\n"
            "2. 逻辑连贯性：回复是否自圆其说？\n"
            "3. 回答质量：是否真正解决了用户的问题？\n\n"
            "请以 JSON 格式输出审查结果:\n"
            "{\n"
            "  \"score\": 0.0到1.0的分数,\n"
            "  \"passed\": true/false,\n"
            "  \"critique\": \"简短的评价\",\n"
            "  \"suggestion\": \"如果未通过，请提供改进后的回复内容\"\n"
            "}"
        )

        try:
            import asyncio
            # 反思通常不需要太高随机性
            res_text = await asyncio.to_thread(
                client.chat, prompt=prompt, temperature=0.1, json_mode=True
            )
            
            # 清理 JSON
            res_text = res_text.strip()
            if res_text.startswith("```json"):
                res_text = res_text[7:]
            if res_text.endswith("```"):
                res_text = res_text[:-3]
                
            data = json.loads(res_text)
            passed = data.get("passed", True)
            score = data.get("score", 1.0)
            
            if not passed and score < 0.6:
                logger.warning(f"⚠️ 自我反思未通过! 分数: {score}, 评价: {data.get('critique')}")
                return False, str(score), data.get("suggestion", response_candidate)
            
            return True, str(score), response_candidate
            
        except Exception as e:
            logger.error(f"自我反思模块执行异常: {e}")
            # 异常时默认放行，避免阻塞
            return True, "1.0", response_candidate
