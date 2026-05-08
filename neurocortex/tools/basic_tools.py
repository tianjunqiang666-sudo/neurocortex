"""
NeuroCortex AI — Basic Tools
==============================
"""

import math
from datetime import datetime
from pathlib import Path
from loguru import logger

from neurocortex.tools.registry import registry

@registry.register
def calculate(expression: str) -> str:
    """计算数学表达式的值，如 123 * 456 或 math.sin(0.5)。只能处理安全的数学运算。"""
    try:
        # 只允许使用 math 模块中安全的方法
        safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        safe_dict["abs"] = abs
        safe_dict["round"] = round
        safe_dict["min"] = min
        safe_dict["max"] = max
        
        # pylint: disable=eval-used
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: 计算 '{expression}' 失败: {e}"

@registry.register
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前系统日期和时间。可选参数 format 指定日期格式。"""
    try:
        return datetime.now().strftime(format)
    except Exception as e:
        return f"Error: 获取时间失败: {e}"

@registry.register
def read_local_file(filepath: str) -> str:
    """读取本地文本文件的内容。filepath 必须是绝对路径或相对项目的路径。"""
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: 文件不存在 '{filepath}'"
        if not path.is_file():
            return f"Error: 路径不是一个文件 '{filepath}'"
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 限制读取内容长度，防止撑爆上下文
        if len(content) > 5000:
            return content[:5000] + "\n...[内容已截断]"
        return content
    except Exception as e:
        return f"Error: 读取文件失败: {e}"
