"""
NeuroCortex AI — Test Fixtures
================================
pytest 共享 fixtures：mock LLM、mock 模块实例。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保项目路径
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def mock_router():
    """模拟 ModelRouter，避免真实 LLM 调用"""
    router = MagicMock()
    mock_client = MagicMock()
    mock_client.chat.return_value = "这是一个模拟的 LLM 回复。"
    router.get_client.return_value = mock_client
    router.config_loader.get_system_settings.return_value = {
        "embedding_model": "all-MiniLM-L6-v2",
        "prediction_error_threshold": 0.6,
        "memory_importance_threshold": 0.8,
    }
    return router


@pytest.fixture
def tmp_data_dir(tmp_path):
    """创建临时数据目录"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def knowledge_base(tmp_data_dir):
    """使用临时目录的 KnowledgeBase"""
    from neurocortex.memory.knowledge_base import KnowledgeBase
    return KnowledgeBase(storage_path=tmp_data_dir / "test_kg.json")


@pytest.fixture
def basal_ganglia(tmp_data_dir):
    """使用临时目录的 BasalGanglia"""
    from neurocortex.pathways.fast.basal_ganglia import BasalGanglia
    return BasalGanglia(habits_path=tmp_data_dir / "test_habits.json")


@pytest.fixture
def frontal_lobe(mock_router):
    """使用模拟 Router 的 FrontalLobe"""
    from neurocortex.pathways.slow.frontal_lobe import FrontalLobe
    return FrontalLobe(mock_router)
