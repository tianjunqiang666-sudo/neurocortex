"""
NeuroCortex AI — Event Bus (事件总线)
=====================================
轻量级的异步事件总线，用于解耦各脑区模块的通信。
支持发布/订阅模式。
"""

import asyncio
from enum import Enum
from typing import Any, Callable, Coroutine
from loguru import logger


class EventType(Enum):
    """系统核心事件类型"""
    INPUT_RECEIVED = "input_received"
    THALAMUS_ROUTED = "thalamus_routed"
    HABIT_MATCHED = "habit_matched"
    THREAT_DETECTED = "threat_detected"
    MEMORY_ENCODED = "memory_encoded"
    EPISODE_CREATED = "episode_created"
    FEEDBACK_RECEIVED = "feedback_received"
    CRITIQUE_COMPLETED = "critique_completed"
    REASONING_START = "reasoning_start"
    TOKEN_GENERATED = "token_generated"
    RESPONSE_COMPLETE = "response_complete"
    SYSTEM_ERROR = "system_error"


class Event:
    """事件对象"""
    def __init__(self, event_type: EventType, source: str, payload: Any = None):
        self.type = event_type
        self.source = source
        self.payload = payload or {}

    def __str__(self):
        return f"Event({self.type.value}, source={self.source})"


class EventBus:
    """异步事件总线"""
    _instance: 'EventBus | None' = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._subscribers: dict[EventType, list[Callable[[Event], Coroutine[Any, Any, None]]]] = {
            e: [] for e in EventType
        }
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._worker_task: asyncio.Task | None = None

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """订阅特定类型的事件 (回调必须是异步函数)"""
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """取消订阅"""
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish(self, event: Event):
        """发布事件到队列"""
        try:
            self._queue.put_nowait(event)
        except Exception as e:
            logger.error(f"事件入队失败: {e}")

    async def start(self):
        """启动事件分发循环"""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._dispatch_loop())
        logger.info("⚡ 事件总线已启动")

    async def stop(self):
        """停止事件分发"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 事件总线已停止")

    async def _dispatch_loop(self):
        while self._running:
            try:
                event = await self._queue.get()
                subscribers = self._subscribers.get(event.type, [])
                
                # 并发执行所有订阅者的回调
                if subscribers:
                    tasks = [asyncio.create_task(sub(event)) for sub in subscribers]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"事件分发异常: {e}")


# 全局单例
_event_bus = EventBus()

def get_event_bus() -> EventBus:
    return _event_bus
