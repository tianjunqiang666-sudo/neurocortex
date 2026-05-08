"""
Tests — Thalamus (丘脑)
========================
测试模态识别、Embedding 生成和 SensoryPacket 结构。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neurocortex.core.thalamus import Thalamus, SensoryPacket


class TestSensoryPacket:
    """SensoryPacket 数据结构测试"""

    def test_packet_creation(self):
        pkt = SensoryPacket(
            modality="text", raw_data="hello",
            embedding=[0.1, 0.2], salience=0.5, routing=["frontal"],
        )
        assert pkt.modality == "text"
        assert pkt.raw_data == "hello"
        assert len(pkt.embedding) == 2
        assert pkt.salience == 0.5
        assert "frontal" in pkt.routing
        assert pkt.id is not None

    def test_packet_to_dict(self):
        pkt = SensoryPacket(
            modality="visual", raw_data="image.png",
            embedding=[0.0], salience=0.9, routing=["occipital"],
        )
        d = pkt.to_dict()
        assert d["modality"] == "visual"
        assert "timestamp" in d


class TestThalamus:
    """Thalamus 模态检测测试"""

    def test_text_modality(self):
        th = Thalamus(embedding_model_name="all-MiniLM-L6-v2")
        packet = th.process_input("你好世界")
        assert packet.modality == "text"
        assert packet.raw_data == "你好世界"

    def test_image_modality(self):
        th = Thalamus(embedding_model_name="all-MiniLM-L6-v2")
        packet = th.process_input("D:/photos/cat.png")
        assert packet.modality == "visual"

    def test_audio_modality(self):
        th = Thalamus(embedding_model_name="all-MiniLM-L6-v2")
        packet = th.process_input("D:/audio/speech.mp3")
        assert packet.modality == "auditory"

    def test_salience_range(self):
        th = Thalamus(embedding_model_name="all-MiniLM-L6-v2")
        packet = th.process_input("test input")
        assert 0.0 <= packet.salience <= 1.0

    def test_embedding_generated(self):
        th = Thalamus(embedding_model_name="all-MiniLM-L6-v2")
        packet = th.process_input("test")
        assert isinstance(packet.embedding, list)
        assert len(packet.embedding) > 0
