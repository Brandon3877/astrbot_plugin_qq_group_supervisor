# message_filter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config_loader import LoadedConfig, normalize_id
from .data_models import CollectedMessage, RuntimeConfig, SenderRole


@dataclass
class MessageFilterResult:
    """
    Result of trying to collect one incoming group message.

    If should_collect is True, message will contain a CollectedMessage.
    If should_collect is False, internal reason explains why it was rejected.
    """

    should_collect: bool
    reason: str
    message: CollectedMessage | None = None

    @classmethod
    def accept(cls, message: CollectedMessage) -> "MessageFilterResult":
        return cls(
            should_collect=True,
            reason="accepted",
            message=message,
        )

    @classmethod
    def reject(cls, reason: str) -> "MessageFilterResult":
        return cls(
            should_collect=False,
            reason=reason,
            message=None,
        )


def filter_event_to_message(
    event: Any,
    loaded_config: LoadedConfig,
) -> MessageFilterResult:
    """
    Decide whether an AstrBot message event should be collected.

    This function should be called by the event listener.
    It does not mutate plugin state.
    It only reads the event and config, then returns a decision.
    """

    runtime = loaded_config.runtime

    if not runtime.enabled:
        return MessageFilterResult.reject("plugin_disabled")

    message_obj = safe_get(event, "message_obj")

    if message_obj is None:
        return MessageFilterResult.reject("missing_message_obj")

    group_id = extract_group_id(event)

    if not group_id:
        return MessageFilterResult.reject("not_group_message")

    if not loaded_config.is_group_monitored(group_id):
        return MessageFilterResult.reject("group_not_monitored")

    sender_id = extract_sender_id(event)
    self_id = extract_self_id(event)

    if runtime.ignore_self_messages and sender_id and self_id and sender_id == self_id:
        return MessageFilterResult.reject("self_message")

    text = extract_message_text(event)

    if runtime.ignore_empty_messages and not text.strip():
        return MessageFilterResult.reject("empty_message")

    if is_command_message(text, runtime):
        return MessageFilterResult.reject("command_message")

    role = extract_sender_role(event)

    if not is_role_allowed(role, runtime):
        return MessageFilterResult.reject(f"role_not_allowed:{role}")

    group_level = extract_group_level(event)

    if is_group_level_exempt(group_level, runtime):
        return MessageFilterResult.reject(
            f"group_level_exempt:{group_level}"
        )

    message_id = extract_message_id(event)

    if not message_id:
        return MessageFilterResult.reject("missing_message_id")

    if not sender_id:
        return MessageFilterResult.reject("missing_sender_id")

    nickname = extract_sender_name(event)

    collected = CollectedMessage(
        group_id=group_id,
        message_id=message_id,
        user_id=sender_id,
        nickname=nickname,
        role=role,
        group_level=group_level,
        timestamp=extract_timestamp(event),
        text=text,
    )

    return MessageFilterResult.accept(collected)


def is_command_message(text: str, runtime: RuntimeConfig) -> bool:
    if not runtime.ignore_command_messages:
        return False

    stripped = text.strip()

    if not stripped:
        return False

    return stripped.startswith(runtime.command_prefixes)


def is_role_allowed(role: SenderRole, runtime: RuntimeConfig) -> bool:
    if runtime.record_role_mode == "members_only":
        return role == "member"

    if runtime.record_role_mode == "members_and_admins":
        return role in {"member", "admin"}

    if runtime.record_role_mode == "members_admins_owner":
        return role in {"member", "admin", "owner"}

    return False


def is_group_level_exempt(group_level: int, runtime: RuntimeConfig) -> bool:
    """
    Config meaning:

    max_record_group_level = 0:
        no level filtering.

    max_record_group_level = 30:
        level >= 30 is exempt, so messages are not recorded.
    """

    threshold = runtime.max_record_group_level

    if threshold <= 0:
        return False

    return group_level >= threshold


def extract_group_id(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")
    return normalize_id(
        safe_get(message_obj, "group_id")
        or safe_call(event, "get_group_id")
        or ""
    )


def extract_message_id(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")
    return normalize_id(
        safe_get(message_obj, "message_id")
        or safe_get(event, "message_id")
        or ""
    )


def extract_sender_id(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")
    sender = safe_get(message_obj, "sender")

    return normalize_id(
        safe_call(event, "get_sender_id")
        or safe_get(sender, "user_id")
        or safe_get(sender, "id")
        or safe_get(sender, "uin")
        or safe_get(event, "sender_id")
        or ""
    )


def extract_self_id(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")

    return normalize_id(
        safe_get(message_obj, "self_id")
        or safe_get(event, "self_id")
        or safe_call(event, "get_self_id")
        or ""
    )


def extract_sender_name(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")
    sender = safe_get(message_obj, "sender")

    name = (
        safe_call(event, "get_sender_name")
        or safe_get(sender, "card")
        or safe_get(sender, "nickname")
        or safe_get(sender, "name")
        or extract_sender_id(event)
        or "未知昵称"
    )

    return str(name).strip() or "未知昵称"


def extract_sender_role(event: Any) -> SenderRole:
    """
    Common OneBot role values:
        member
        admin
        owner

    If the adapter gives Chinese values, plugin also normalizes them.
    """

    message_obj = safe_get(event, "message_obj")
    sender = safe_get(message_obj, "sender")
    raw_message = safe_get(message_obj, "raw_message")

    raw_sender = safe_get(raw_message, "sender")

    role_raw = (
        safe_get(sender, "role")
        or safe_get(raw_sender, "role")
        or safe_get(event, "role")
        or ""
    )

    role_text = str(role_raw).strip().lower()

    if role_text in {"member", "normal", "普通群员", "群员"}:
        return "member"

    if role_text in {"admin", "administrator", "管理员"}:
        return "admin"

    if role_text in {"owner", "creator", "群主"}:
        return "owner"

    return "unknown"


def extract_group_level(event: Any) -> int:
    """
    QQ group member level may come from:
    - message_obj.sender.level
    - message_obj.raw_message.sender.level
    - adapter-specific dict fields

    If unknown, use 0.
    """

    message_obj = safe_get(event, "message_obj")
    sender = safe_get(message_obj, "sender")
    raw_message = safe_get(message_obj, "raw_message")
    raw_sender = safe_get(raw_message, "sender")

    value = (
        safe_get(sender, "level")
        or safe_get(sender, "group_level")
        or safe_get(raw_sender, "level")
        or safe_get(raw_sender, "group_level")
        or 0
    )

    return as_int(value, default=0, minimum=0)


def extract_message_text(event: Any) -> str:
    message_obj = safe_get(event, "message_obj")

    text = (
        safe_get(event, "message_str")
        or safe_get(message_obj, "message_str")
        or ""
    )

    return str(text)


def extract_timestamp(event: Any) -> float:
    message_obj = safe_get(event, "message_obj")

    value = (
        safe_get(message_obj, "timestamp")
        or safe_get(event, "timestamp")
        or 0
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Get obj.key or obj[key] safely.

    This makes the filter tolerant of AstrBot objects, OneBot raw dicts,
    and adapter-specific message objects.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def safe_call(obj: Any, method_name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    method = getattr(obj, method_name, None)

    if not callable(method):
        return default

    try:
        return method()
    except Exception:
        return default


def as_int(value: Any, default: int = 0, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default

    if minimum is not None and result < minimum:
        return minimum

    return result


__all__ = [
    "MessageFilterResult",
    "filter_event_to_message",
    "extract_group_id",
    "extract_message_id",
    "extract_sender_id",
    "extract_self_id",
    "extract_sender_name",
    "extract_sender_role",
    "extract_group_level",
    "extract_message_text",
    "extract_timestamp",
]