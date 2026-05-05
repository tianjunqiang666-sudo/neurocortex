"""
NeuroCortex AI — OccipitalLobe (枕叶)
========================================
慢速通路：视觉处理。
接收图像输入，调用多模态视觉大模型提取结构化语义特征。
"""

from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

class OccipitalLobe:
    """枕叶 — 视觉处理

    阶段二引入视觉识别能力，调用如 llava 的本地视觉模型，提取图中物体和场景信息。
    """

    required_capabilities = {
        "reasoning_depth": "low",
        "output_schema": {
            "type": "object",
            "properties": {
                "objects": {"type": "array"},
                "scene": {"type": "string"}
            }
        },
    }

    def __init__(self, model_router: Any) -> None:
        self.model_router = model_router
        logger.info("OccipitalLobe 初始化")

    def process(self, sensory_packet: Any) -> dict[str, Any]:
        """处理视觉输入，输出结构化语义摘要

        Args:
            sensory_packet: SensoryPacket (应包含 modality="visual" 和 raw_data=image_path)

        Returns:
            结构化描述:
            {"objects": ["...", "..."], "scene": "...", "text": "..."}
        """
        raw_data = sensory_packet.raw_data if hasattr(sensory_packet, 'raw_data') else str(sensory_packet)
        modality = sensory_packet.modality if hasattr(sensory_packet, 'modality') else "visual"

        if modality != "visual":
            logger.warning(f"OccipitalLobe 收到非视觉信号: {modality}")
            return {"objects": [], "scene": "unknown"}

        return self._process_image(str(raw_data))

    def _process_image(self, image_path: str) -> dict[str, Any]:
        """调用视觉大模型处理图像"""
        if not Path(image_path).exists():
            logger.error(f"视觉处理失败，图像文件不存在: {image_path}")
            return {"objects": [], "scene": "file_not_found"}

        try:
            # Base64 编码图像
            with open(image_path, "rb") as image_file:
                b64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # 获取视觉模型客户端
            client = self.model_router.get_client("occipital", required_capabilities=self.required_capabilities)

            prompt = (
                "You are the visual cortex of an AI. Analyze the provided image.\n"
                "Extract the main objects and the overall scene description.\n"
                "Return the result strictly as a JSON object with 'objects' (list of strings) and 'scene' (string) fields."
            )

            logger.info(f"▶ OccipitalLobe 正在调用视觉模型 {client.model_name} 分析图像...")
            response_text = client.chat(
                prompt=prompt,
                json_mode=True,
                images=[b64_image]
            )

            try:
                result = json.loads(response_text)
                return {
                    "objects": result.get("objects", []),
                    "scene": result.get("scene", "unknown"),
                    "text": result.get("text", "") # Optional text found in image
                }
            except json.JSONDecodeError:
                logger.error(f"视觉模型输出非JSON格式: {response_text}")
                return {"objects": [], "scene": "parsing_error"}

        except Exception as e:
            logger.error(f"视觉处理异常: {e}")
            return {"objects": [], "scene": "processing_error"}
