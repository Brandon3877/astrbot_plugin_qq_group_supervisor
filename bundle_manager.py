# bundle_manager.py

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .config_loader import LoadedConfig, normalize_id
from .data_models import CollectedMessage, MessageBundle


@dataclass
class GroupMessageBuffer:
    """
    Runtime message buffer for one QQ group.

    This object is not sent to the LLM directly.
    When a trigger condition is met, it is converted into a MessageBundle.
    """

    group_id: str
    messages: list[CollectedMessage] = field(default_factory=list)
    first_message_received_at: float = 0.0
    last_message_received_at: float = 0.0

    def append(self, message: CollectedMessage, now: float) -> None:
        if not self.messages:
            self.first_message_received_at = now

        self.messages.append(message)
        self.last_message_received_at = now

    def clear(self) -> None:
        self.messages.clear()
        self.first_message_received_at = 0.0
        self.last_message_received_at = 0.0

    @property
    def count(self) -> int:
        return len(self.messages)

    def age_seconds(self, now: float) -> float:
        if not self.messages or self.first_message_received_at <= 0:
            return 0.0

        return max(0.0, now - self.first_message_received_at)


@dataclass
class BundleCreationResult:
    """
    Result returned when a bundle is created.

    trigger_reason examples:
    - "message_limit"
    - "time_limit"
    - "manual_flush"
    """

    bundle: MessageBundle
    trigger_reason: str


@dataclass
class BufferStatus:
    """
    Small debug/status object.

    Useful later for logs or a command like:
        /supervisor_status
    """

    group_id: str
    message_count: int
    first_message_received_at: float
    last_message_received_at: float
    age_seconds: float


class BundleManager:
    """
    Manages message buffers for all monitored groups.

    This class does not call the LLM.
    It only decides when a MessageBundle should be created.
    """

    def __init__(self, loaded_config: LoadedConfig):
        self.loaded_config = loaded_config
        self.buffers: dict[str, GroupMessageBuffer] = {}

    def update_config(self, loaded_config: LoadedConfig) -> None:
        """
        Replace the runtime config.

        Later, if AstrBot reloads plugin config, we can call this without
        losing already collected messages.
        """
        self.loaded_config = loaded_config

    def add_message(
        self,
        message: CollectedMessage,
        now: float | None = None,
    ) -> BundleCreationResult | None:
        """
        Add one collected message.

        Returns BundleCreationResult if this message triggers a bundle.
        Otherwise returns None.
        """
        now = self._now(now)
        group_id = normalize_id(message.group_id)

        if not group_id:
            return None

        buffer = self._get_or_create_buffer(group_id)
        buffer.append(message, now)

        if self._should_trigger_by_message_limit(buffer):
            return self._create_bundle_and_clear(
                group_id=group_id,
                trigger_reason="message_limit",
                created_at=now,
            )

        if self._should_trigger_by_time_limit(buffer, now):
            return self._create_bundle_and_clear(
                group_id=group_id,
                trigger_reason="time_limit",
                created_at=now,
            )

        return None

    def check_expired(
        self,
        now: float | None = None,
    ) -> list[BundleCreationResult]:
        """
        Check all group buffers for time-limit expiration.

        This will be called periodically by the plugin.

        Without this method, a bundle can be ignored for too long,
        because add_message() only triggers when new messages come in.
        """
        now = self._now(now)
        results: list[BundleCreationResult] = []

        for group_id in list(self.buffers.keys()):
            buffer = self.buffers.get(group_id)

            if buffer is None or not buffer.messages:
                continue

            if self._should_trigger_by_time_limit(buffer, now):
                result = self._create_bundle_and_clear(
                    group_id=group_id,
                    trigger_reason="time_limit",
                    created_at=now,
                )
                if result is not None:
                    results.append(result)

        return results

    def force_flush_group(
        self,
        group_id: str,
        reason: str = "manual_flush",
        ignore_min_messages: bool = False,
        now: float | None = None,
    ) -> BundleCreationResult | None:
        """
        Manually create a bundle for one group.

        By default, min_messages_to_analyze still applies.
        Set ignore_min_messages=True for shutdown cleanup.
        """
        now = self._now(now)
        group_id = normalize_id(group_id)

        buffer = self.buffers.get(group_id)

        if buffer is None or not buffer.messages:
            return None

        if not ignore_min_messages and not self._has_min_messages(buffer):
            return None

        return self._create_bundle_and_clear(
            group_id=group_id,
            trigger_reason=reason,
            created_at=now,
        )

    def force_flush_all(
        self,
        reason: str = "manual_flush_all",
        ignore_min_messages: bool = False,
        now: float | None = None,
    ) -> list[BundleCreationResult]:
        """
        Manually create bundles for all non-empty groups.
        """
        now = self._now(now)
        results: list[BundleCreationResult] = []

        for group_id in list(self.buffers.keys()):
            result = self.force_flush_group(
                group_id=group_id,
                reason=reason,
                ignore_min_messages=ignore_min_messages,
                now=now,
            )

            if result is not None:
                results.append(result)

        return results

    def clear_group(self, group_id: str) -> None:
        group_id = normalize_id(group_id)

        if group_id in self.buffers:
            self.buffers[group_id].clear()
            del self.buffers[group_id]

    def clear_all(self) -> None:
        self.buffers.clear()

    def get_buffer_statuses(
        self,
        now: float | None = None,
    ) -> list[BufferStatus]:
        now = self._now(now)
        statuses: list[BufferStatus] = []

        for group_id, buffer in self.buffers.items():
            if not buffer.messages:
                continue

            statuses.append(
                BufferStatus(
                    group_id=group_id,
                    message_count=buffer.count,
                    first_message_received_at=buffer.first_message_received_at,
                    last_message_received_at=buffer.last_message_received_at,
                    age_seconds=buffer.age_seconds(now),
                )
            )

        return statuses

    def get_group_message_count(self, group_id: str) -> int:
        group_id = normalize_id(group_id)
        buffer = self.buffers.get(group_id)

        if buffer is None:
            return 0

        return buffer.count

    def _get_or_create_buffer(self, group_id: str) -> GroupMessageBuffer:
        buffer = self.buffers.get(group_id)

        if buffer is None:
            buffer = GroupMessageBuffer(group_id=group_id)
            self.buffers[group_id] = buffer

        return buffer

    def _should_trigger_by_message_limit(
        self,
        buffer: GroupMessageBuffer,
    ) -> bool:
        runtime = self.loaded_config.runtime

        message_limit = runtime.bundle_message_limit

        if message_limit <= 0:
            return False

        if buffer.count < message_limit:
            return False

        return self._has_min_messages(buffer)

    def _should_trigger_by_time_limit(
        self,
        buffer: GroupMessageBuffer,
        now: float,
    ) -> bool:
        runtime = self.loaded_config.runtime

        time_limit = runtime.bundle_time_limit_seconds

        if time_limit <= 0:
            return False

        if not self._has_min_messages(buffer):
            return False

        return buffer.age_seconds(now) >= time_limit

    def _has_min_messages(self, buffer: GroupMessageBuffer) -> bool:
        runtime = self.loaded_config.runtime
        return buffer.count >= runtime.min_messages_to_analyze

    def _create_bundle_and_clear(
        self,
        group_id: str,
        trigger_reason: str,
        created_at: float,
    ) -> BundleCreationResult | None:
        buffer = self.buffers.get(group_id)

        if buffer is None or not buffer.messages:
            return None

        messages = list(buffer.messages)

        bundle = MessageBundle(
            bundle_id=self._make_bundle_id(group_id, created_at),
            group_id=group_id,
            created_at=created_at,
            messages=messages,
            group_rule=self.loaded_config.get_group_rule(group_id),
            available_punishments=list(self.loaded_config.punishments),
        )

        buffer.clear()
        del self.buffers[group_id]

        return BundleCreationResult(
            bundle=bundle,
            trigger_reason=trigger_reason,
        )

    @staticmethod
    def _make_bundle_id(group_id: str, created_at: float) -> str:
        created_ms = int(created_at * 1000)
        short_uuid = uuid.uuid4().hex[:8]
        return f"group_{group_id}_{created_ms}_{short_uuid}"

    @staticmethod
    def _now(now: float | None) -> float:
        if now is not None:
            return now

        return time.time()


__all__ = [
    "GroupMessageBuffer",
    "BundleCreationResult",
    "BufferStatus",
    "BundleManager",
]