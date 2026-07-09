# operation_handler.py

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal

from .config_loader import normalize_id
from .executor import ModerationOperationHandler
from .permission_checker import GroupMemberRole, normalize_group_member_role


MentionMode = Literal[
    "cq_code",
    "plain_text",
    "none",
]


@dataclass
class OneBotCallResult:
    """
    Normalized result of one protocol API call.
    """

    action_name: str
    payload: dict[str, Any]
    raw_result: Any

    def to_text(self) -> str:
        return (
            f"已调用 {self.action_name}，"
            f"payload={self.payload}，"
            f"返回={self.raw_result}"
        )


class OperationHandlerError(Exception):
    """
    Raised when the QQ operation handler cannot call the protocol-side API.
    """

    pass


class OneBotV11OperationHandler(ModerationOperationHandler):
    """
    Real operation handler for AstrBot AIOCQHTTP / OneBot v11-like platforms.

    The executor only knows the abstract methods:
    - send_group_warning
    - recall_message
    - mute_user
    - kick_user

    This class translates those abstract methods into OneBot API calls:
    - send_group_msg
    - delete_msg
    - set_group_ban
    - set_group_kick

    It also provides send_private_message(),
    which is useful for admin review messages.
    """

    def __init__(
        self,
        *,
        client: Any,
        mention_mode: MentionMode = "cq_code",
    ):
        if client is None:
            raise OperationHandlerError("client is None.")

        self.client = client
        self.mention_mode = mention_mode

    async def get_login_user_id(self) -> str:
        """
        Return the bot's own QQ number.
        """
        result = await self.call_action("get_login_info")
        data = extract_onebot_data(result.raw_result)

        return normalize_id(
            data.get("user_id")
            or data.get("uin")
            or data.get("id")
            or ""
        )

    async def get_group_member_role(
        self,
        *,
        group_id: str,
        user_id: str,
    ) -> GroupMemberRole:
        """
        Return this member's role in the group.

        Expected OneBot/NapCat role values:
            owner / admin / member
        """
        result = await self.call_action(
            "get_group_member_info",
            group_id=to_onebot_id(group_id),
            user_id=to_onebot_id(user_id),
            no_cache=True,
        )

        data = extract_onebot_data(result.raw_result)

        return normalize_group_member_role(data.get("role"))
        
    async def send_group_warning(
        self,
        *,
        group_id: str,
        target_user_id: str,
        warning_text: str,
        quote_message_id: str | None = None,
    ) -> str:
        message = build_warning_message(
            target_user_id=target_user_id,
            warning_text=warning_text,
            mention_mode=self.mention_mode,
            quote_message_id=quote_message_id,
        )

        result = await self.call_action(
            "send_group_msg",
            group_id=to_onebot_id(group_id),
            message=message,
        )

        return result.to_text()

    async def recall_message(
        self,
        *,
        group_id: str,
        message_id: str,
    ) -> str:
        # group_id is currently not needed by OneBot v11 delete_msg,
        # but it can be kept in the method signature because executor.py is
        # platform-independent and other platforms may need it.
        result = await self.call_action(
            "delete_msg",
            message_id=to_onebot_id(message_id),
        )

        return result.to_text()

    async def mute_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        duration_seconds: int,
    ) -> str:
        result = await self.call_action(
            "set_group_ban",
            group_id=to_onebot_id(group_id),
            user_id=to_onebot_id(target_user_id),
            duration=int(duration_seconds),
        )

        return result.to_text()

    async def kick_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        blacklist: bool,
    ) -> str:
        result = await self.call_action(
            "set_group_kick",
            group_id=to_onebot_id(group_id),
            user_id=to_onebot_id(target_user_id),
            reject_add_request=bool(blacklist),
        )

        return result.to_text()

    async def send_private_message(
        self,
        *,
        target_user_id: str,
        text: str,
    ) -> str:
        """
        Useful for admin review notifications.

        This is not part of executor.py's ModerationOperationHandler protocol,
        because executor.py only performs group moderation actions.
        """
        result = await self.call_action(
            "send_private_msg",
            user_id=to_onebot_id(target_user_id),
            message=escape_cq_text(text),
        )

        return result.to_text()

    async def send_group_text(
        self,
        *,
        group_id: str,
        text: str,
    ) -> str:
        """
        General helper for sending group messages.
        """
        result = await self.call_action(
            "send_group_msg",
            group_id=to_onebot_id(group_id),
            message=escape_cq_text(text),
        )

        return result.to_text()

    async def call_action(
        self,
        action_name: str,
        **payload: Any,
    ) -> OneBotCallResult:
        """
        Call protocol-side API in a few compatible ways.

        AstrBot aiocqhttp example:
            await client.api.call_action("delete_msg", **payload)

        Some OneBot SDKs expose:
            await client.call_api(action_name, **payload)

        Some expose:
            await client.call_action(action_name, **payload)
        """

        raw_result = await call_client_api(
            client=self.client,
            action_name=action_name,
            payload=payload,
        )

        return OneBotCallResult(
            action_name=action_name,
            payload=payload,
            raw_result=raw_result,
        )


def create_operation_handler_from_event(
    event: Any,
    *,
    mention_mode: MentionMode = "cq_code",
) -> OneBotV11OperationHandler:
    """
    Create operation handler from an AstrBot aiocqhttp event.

    In main.py, this is done:

        self.operation_handler = create_operation_handler_from_event(event)
        self.executor.update_operation_handler(self.operation_handler)

    This is the most direct path because AstrBot's aiocqhttp event exposes
    event.bot in its protocol API example.
    """

    client = getattr(event, "bot", None)

    if client is None:
        raise OperationHandlerError(
            "Cannot create operation handler from event: event.bot is missing."
        )

    return OneBotV11OperationHandler(
        client=client,
        mention_mode=mention_mode,
    )


async def call_client_api(
    *,
    client: Any,
    action_name: str,
    payload: dict[str, Any],
) -> Any:
    """
    Try several common API-call shapes.

    Supported shapes:
    1. client.api.call_action(action_name, **payload)
    2. client.call_action(action_name, **payload)
    3. client.call_api(action_name, **payload)
    4. client.api.call_api(action_name, **payload)

    The first one is the AstrBot aiocqhttp documented style.
    """

    api = getattr(client, "api", None)

    candidates = [
        getattr(api, "call_action", None) if api is not None else None,
        getattr(client, "call_action", None),
        getattr(client, "call_api", None),
        getattr(api, "call_api", None) if api is not None else None,
    ]

    last_error: Exception | None = None

    for candidate in candidates:
        if not callable(candidate):
            continue

        try:
            result = candidate(action_name, **payload)

            if inspect.isawaitable(result):
                return await result

            return result

        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise OperationHandlerError(
            f"Failed to call API action {action_name}: {last_error}"
        ) from last_error

    raise OperationHandlerError(
        "No supported API caller found on client. "
        "Expected client.api.call_action, client.call_action, "
        "client.call_api, or client.api.call_api."
    )


def build_warning_message(
    *,
    target_user_id: str,
    warning_text: str,
    mention_mode: MentionMode = "cq_code",
    quote_message_id: str | None = None,
) -> str:
    safe_user_id = sanitize_qq_id_for_cq(target_user_id)
    safe_text = escape_cq_text(warning_text)

    parts: list[str] = []

    safe_quote_id = sanitize_qq_id_for_cq(quote_message_id)

    if safe_quote_id:
        parts.append(f"[CQ:reply,id={safe_quote_id}]")

    if mention_mode == "cq_code" and safe_user_id:
        parts.append(f"[CQ:at,qq={safe_user_id}]")
    elif mention_mode == "plain_text" and safe_user_id:
        parts.append(f"@{safe_user_id}")

    parts.append(safe_text)

    return " ".join(parts)


def escape_cq_text(text: str) -> str:
    """
    Escape text for OneBot v11 CQ-code string messages.

    This prevents user-provided or LLM-generated warning text from accidentally
    becoming CQ code.

    CQ escaping:
    - &  -> &amp;
    - [  -> &#91;
    - ]  -> &#93;
    - ,  -> &#44;
    """

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace(",", "&#44;")
    )


def sanitize_qq_id_for_cq(value: Any) -> str:
    """
    CQ at-code should only receive normal numeric QQ IDs.

    If the value is not numeric, return an empty string so the caller can avoid
    generating malformed CQ code.
    """

    text = normalize_id(value)

    if text.isdigit():
        return text

    return ""


def to_onebot_id(value: Any) -> int | str:
    """
    Many OneBot implementations accept ID in either string or int format.
    
    Users and future developers can choose either to convert to int or keep as str.
    If you encounter issues regarding ID format, please change this setting.
    """

    CHOICE_CONVERT_INT = False

    if CHOICE_CONVERT_INT:
        return to_onebot_id_int(value=value)
    
    return to_onebot_id_str(value=value)


def to_onebot_id_int(value: Any) -> int | str:
    """
    Convert numeric-looking IDs into int for OneBot APIs.

    Many OneBot implementations accept either string or int, but the v11-style
    docs usually describe group_id/user_id/message_id as numeric values.
    Keeping non-numeric values as strings makes this helper tolerant of unusual
    adapters.
    """

    text = normalize_id(value)

    if text.isdigit():
        try:
            return int(text)
        except ValueError:
            return text

    return text


def to_onebot_id_str(value: Any) -> str:
    """
    Also offers an option to keep OneBot IDs as strings.

    Newest NapCat recommends using string types for message_id/user_id/group_id.
    This also avoids int64 precision problems and keeps internal ID handling consistent.
    """
    return normalize_id(value)


def extract_onebot_data(raw_result: Any) -> dict[str, Any]:
    """
    Extract OneBot response data.

    Common shapes:
        {"status": "ok", "retcode": 0, "data": {...}}
        {"user_id": "...", "nickname": "..."}
        object with .data
    """

    if raw_result is None:
        return {}

    if isinstance(raw_result, dict):
        data = raw_result.get("data")

        if isinstance(data, dict):
            return data

        return raw_result

    data = getattr(raw_result, "data", None)

    if isinstance(data, dict):
        return data

    return {}


__all__ = [
    "MentionMode",
    "OneBotCallResult",
    "OperationHandlerError",
    "OneBotV11OperationHandler",
    "create_operation_handler_from_event",
    "call_client_api",
    "build_warning_message",
    "escape_cq_text",
    "sanitize_qq_id_for_cq",
    "to_onebot_id",
    "to_onebot_id_int",
    "to_onebot_id_str",
    "extract_onebot_data"
]