"""
NeuroCortex AI — Tool Registry
================================
提供基础的工具注册与调用机制，支持自动生成 JSON Schema 描述供 LLM 参考。
"""

import inspect
from typing import Any, Callable, Dict, List
from loguru import logger


class ToolRegistry:
    """工具注册表，管理所有对 LLM 开放的工具"""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}
        
    def register(self, func: Callable) -> Callable:
        """注册工具函数的装饰器"""
        name = func.__name__
        self._tools[name] = func
        self._schemas[name] = self._generate_schema(func)
        logger.debug(f"已注册外部工具: {name}")
        return func
        
    def _generate_schema(self, func: Callable) -> dict:
        """基于函数签名和 docstring 自动生成 JSON Schema 格式的描述"""
        sig = inspect.signature(func)
        schema = {
            "name": func.__name__,
            "description": func.__doc__ or "未提供描述",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
                
            schema["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }
            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)
                
        return schema
        
    def get_all_schemas(self) -> List[dict]:
        """获取所有工具的 Schema 描述，传递给 LLM 系统提示词"""
        return list(self._schemas.values())
        
    def execute(self, name: str, **kwargs) -> str:
        """执行指定工具并返回结果文本"""
        if name not in self._tools:
            return f"Error: 未知工具 {name}"
            
        try:
            logger.info(f"🛠️ 执行外部工具: {name}({kwargs})")
            result = self._tools[name](**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"工具执行异常 {name}: {e}")
            return f"Error: 工具执行失败 ({e})"


# 全局单例注册表
registry = ToolRegistry()
