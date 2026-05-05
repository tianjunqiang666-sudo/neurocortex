"""
NeuroCortex AI — Brainstem (脑干)
==================================
系统心跳、分级中断、睡眠/清醒调度。
维护系统时钟，控制"清醒/睡眠"循环，接收任意模块的 alert 并实施分级中断。
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

from loguru import logger


class SystemState(str, Enum):
    """系统状态枚举"""
    AWAKE = "awake"
    SLEEPING = "sleeping"
    INTERRUPTED = "interrupted"


class InterruptLevel(str, Enum):
    """中断级别"""
    LOG_ONLY = "log_only"          # level < 0.3
    SLOW_DOWN = "slow_down"        # 0.3 <= level < 0.7
    EMERGENCY = "emergency"        # level >= 0.7


class AlertEvent:
    """警报事件"""
    def __init__(self, level: float, source: str, message: str) -> None:
        self.level = max(0.0, min(1.0, level))
        self.source = source
        self.message = message
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.interrupt_level = self._classify()

    def _classify(self) -> InterruptLevel:
        if self.level < 0.3:
            return InterruptLevel.LOG_ONLY
        elif self.level < 0.7:
            return InterruptLevel.SLOW_DOWN
        else:
            return InterruptLevel.EMERGENCY

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level, "source": self.source,
            "message": self.message, "timestamp": self.timestamp,
            "interrupt_level": self.interrupt_level.value,
        }


class Brainstem:
    """脑干 — 系统调度核心

    职责:
      - 维护系统时钟，控制清醒/睡眠循环
      - 接收分级中断 alert
      - 广播 wake/sleep/interrupt 信号到事件总线

    Attributes:
        state: 当前系统状态
        wake_duration: 清醒期时长(秒)
        sleep_duration: 睡眠期时长(秒)
        alert_history: 警报历史记录
    """

    def __init__(
        self,
        wake_duration_seconds: int = 3600,
        sleep_duration_seconds: int = 600,
    ) -> None:
        self.state: SystemState = SystemState.AWAKE
        self.wake_duration = wake_duration_seconds
        self.sleep_duration = sleep_duration_seconds
        self.alert_history: deque[AlertEvent] = deque(maxlen=1000)
        self.cycle_count: int = 0
        self.started_at: str = datetime.now(timezone.utc).isoformat()

        # 事件回调注册表
        self._on_sleep_callbacks: list[Callable[[], Coroutine | None]] = []
        self._on_wake_callbacks: list[Callable[[], Coroutine | None]] = []
        self._on_interrupt_callbacks: list[Callable[[AlertEvent], Coroutine | None]] = []
        self._on_slowdown_callbacks: list[Callable[[AlertEvent], Coroutine | None]] = []

        # 内部控制
        self._running: bool = False
        self._cycle_task: asyncio.Task | None = None

        logger.info(f"Brainstem 初始化: 清醒{wake_duration_seconds}s / 睡眠{sleep_duration_seconds}s")

    # ── 事件注册 ──────────────────────────────────────

    def on_sleep(self, callback: Callable) -> None:
        """注册睡眠信号回调"""
        self._on_sleep_callbacks.append(callback)

    def on_wake(self, callback: Callable) -> None:
        """注册唤醒信号回调"""
        self._on_wake_callbacks.append(callback)

    def on_interrupt(self, callback: Callable) -> None:
        """注册紧急中断回调 (level >= 0.7)"""
        self._on_interrupt_callbacks.append(callback)

    def on_slowdown(self, callback: Callable) -> None:
        """注册减速信号回调 (0.3 <= level < 0.7)"""
        self._on_slowdown_callbacks.append(callback)

    # ── 核心接口 ──────────────────────────────────────

    def alert(self, level: float, source: str, message: str) -> AlertEvent:
        """接收警报并执行分级中断

        Args:
            level: 警报级别 (0.0 ~ 1.0)
            source: 来源模块名
            message: 警报描述

        Returns:
            创建的 AlertEvent
        """
        event = AlertEvent(level, source, message)
        self.alert_history.append(event)

        if event.interrupt_level == InterruptLevel.LOG_ONLY:
            logger.debug(f"[Alert LOG] {source}: {message} (level={level:.2f})")

        elif event.interrupt_level == InterruptLevel.SLOW_DOWN:
            logger.warning(f"[Alert SLOWDOWN] {source}: {message} (level={level:.2f})")
            self._fire_callbacks_sync(self._on_slowdown_callbacks, event)

        elif event.interrupt_level == InterruptLevel.EMERGENCY:
            logger.error(f"[Alert EMERGENCY] {source}: {message} (level={level:.2f})")
            self.state = SystemState.INTERRUPTED
            self._fire_callbacks_sync(self._on_interrupt_callbacks, event)

        return event

    async def alert_async(self, level: float, source: str, message: str) -> AlertEvent:
        """异步版警报接收"""
        event = AlertEvent(level, source, message)
        self.alert_history.append(event)

        if event.interrupt_level == InterruptLevel.LOG_ONLY:
            logger.debug(f"[Alert LOG] {source}: {message}")
        elif event.interrupt_level == InterruptLevel.SLOW_DOWN:
            logger.warning(f"[Alert SLOWDOWN] {source}: {message}")
            await self._fire_callbacks_async(self._on_slowdown_callbacks, event)
        elif event.interrupt_level == InterruptLevel.EMERGENCY:
            logger.error(f"[Alert EMERGENCY] {source}: {message}")
            self.state = SystemState.INTERRUPTED
            await self._fire_callbacks_async(self._on_interrupt_callbacks, event)

        return event

    def trigger_sleep(self) -> None:
        """手动触发睡眠周期（用于 CLI 的 /sleep 命令）"""
        logger.info("═══ 手动触发睡眠周期 ═══")
        self.state = SystemState.SLEEPING
        self._fire_callbacks_sync(self._on_sleep_callbacks)

    async def trigger_sleep_async(self) -> None:
        """异步手动触发睡眠"""
        logger.info("═══ 手动触发睡眠周期 (async) ═══")
        self.state = SystemState.SLEEPING
        await self._fire_callbacks_async(self._on_sleep_callbacks)
        self.state = SystemState.AWAKE
        self.cycle_count += 1
        logger.info("═══ 唤醒，恢复在线模式 ═══")

    async def start_cycle(self) -> None:
        """启动自动清醒/睡眠循环"""
        self._running = True
        logger.info("Brainstem 自动循环已启动")
        while self._running:
            # 清醒期
            self.state = SystemState.AWAKE
            await self._fire_callbacks_async(self._on_wake_callbacks)
            logger.info(f"清醒期开始，持续 {self.wake_duration}s")
            await asyncio.sleep(self.wake_duration)

            # 睡眠期
            self.state = SystemState.SLEEPING
            await self._fire_callbacks_async(self._on_sleep_callbacks)
            logger.info(f"睡眠期开始，持续 {self.sleep_duration}s")
            await asyncio.sleep(self.sleep_duration)

            self.cycle_count += 1
            logger.info(f"已完成 {self.cycle_count} 个睡眠-清醒周期")

    def stop_cycle(self) -> None:
        """停止自动循环"""
        self._running = False
        if self._cycle_task and not self._cycle_task.done():
            self._cycle_task.cancel()
        logger.info("Brainstem 循环已停止")

    def get_status(self) -> dict[str, Any]:
        """获取系统状态摘要"""
        return {
            "state": self.state.value,
            "cycle_count": self.cycle_count,
            "started_at": self.started_at,
            "total_alerts": len(self.alert_history),
            "recent_alerts": [a.to_dict() for a in list(self.alert_history)[-5:]],
        }

    # ── 内部方法 ──────────────────────────────────────

    def _fire_callbacks_sync(self, callbacks: list, *args) -> None:
        for cb in callbacks:
            try:
                result = cb(*args)
                if asyncio.iscoroutine(result):
                    logger.warning(f"同步上下文中调用了异步回调 {cb.__name__}，跳过")
            except Exception as e:
                logger.error(f"回调执行失败: {e}")

    async def _fire_callbacks_async(self, callbacks: list, *args) -> None:
        for cb in callbacks:
            try:
                result = cb(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"异步回调执行失败: {e}")
