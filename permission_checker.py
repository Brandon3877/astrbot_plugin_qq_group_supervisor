# permission_checker.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .config_loader import normalize_id
from .data_models import MessageBundle, SenderRole, ValidatedAction


GroupMemberRole = Literal["owner", "admin", "member", "unknown"]

PermissionDecision = Literal["allowed", "blocked", "unknown"]


@dataclass
class PermissionCheckResult:
    """
    Result of checking whether the bot should attempt a moderation action.

    decision:
        allowed:
            Permission looks OK.

        blocked:
            We know the bot lacks permission, so executor.py should skip.

        unknown:
            Permission check failed or data is incomplete.
            executor.py should still attempt the real API call and catch failure.
    """

    decision: PermissionDecision
    action_type: str
    group_id: str
    self_id: str | None = None
    bot_role: GroupMemberRole = "unknown"
    target_user_id: str | None = None
    target_role: GroupMemberRole = "unknown"
    reason: str = ""

    @property
    def should_attempt(self) -> bool:
        return self.decision != "blocked"


class ModerationPermissionChecker:
    """
    Checks bot and target member roles before real moderation actions.

    It expects role_provider to expose:

        async get_login_user_id() -> str
        async get_group_member_role(group_id: str, user_id: str) -> GroupMemberRole

    Our OneBotV11OperationHandler will implement these methods.
    """

    def __init__(self, role_provider: Any):
        self.role_provider = role_provider
        self._self_id_cache: str | None = None
        self._role_cache: dict[tuple[str, str], GroupMemberRole] = {}

    async def check_action(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> PermissionCheckResult:
        action_type = action.punishment.type
        group_id = normalize_id(bundle.group_id)

        if action_type == "warn":
            return PermissionCheckResult(
                decision="allowed",
                action_type=action_type,
                group_id=group_id,
                target_user_id=action.target_user_id,
                reason="警告只需要发送群消息，不需要群管理员权限预检查。",
            )

        if action_type not in {"recall", "mute", "kick"}:
            return PermissionCheckResult(
                decision="unknown",
                action_type=action_type,
                group_id=group_id,
                target_user_id=action.target_user_id,
                reason=f"未知操作类型：{action_type}，交给真实API调用决定。",
            )

        self_id = await self._get_self_id()

        if not self_id:
            return PermissionCheckResult(
                decision="unknown",
                action_type=action_type,
                group_id=group_id,
                target_user_id=action.target_user_id,
                reason="无法获取机器人自身QQ号，交给真实API调用决定。",
            )

        bot_role = await self._get_member_role(
            group_id=group_id,
            user_id=self_id,
        )

        target_user_id = normalize_id(action.target_user_id)

        target_role = await self._get_target_role(
            bundle=bundle,
            target_user_id=target_user_id,
        )

        base_result = PermissionCheckResult(
            decision="unknown",
            action_type=action_type,
            group_id=group_id,
            self_id=self_id,
            bot_role=bot_role,
            target_user_id=target_user_id or None,
            target_role=target_role,
        )

        if bot_role == "member":
            base_result.decision = "blocked"
            base_result.reason = "机器人是普通群员，不能执行撤回、禁言或踢人。"
            return base_result

        if bot_role == "unknown":
            base_result.decision = "unknown"
            base_result.reason = "无法确认机器人在群内身份，交给真实API调用决定。"
            return base_result

        if not target_user_id:
            base_result.decision = "blocked"
            base_result.reason = f"{action_type} 缺少目标用户ID。"
            return base_result

        if target_user_id == self_id and action_type in {"mute", "kick"}:
            base_result.decision = "blocked"
            base_result.reason = "不会尝试禁言或踢出机器人自己。"
            return base_result

        manage_decision = can_manage_target(
            bot_role=bot_role,
            target_role=target_role,
        )

        if manage_decision == "allowed":
            base_result.decision = "allowed"
            base_result.reason = (
                f"权限检查通过：机器人身份={bot_role}，目标身份={target_role}。"
            )
            return base_result

        if manage_decision == "blocked":
            base_result.decision = "blocked"
            base_result.reason = (
                f"权限不足：机器人身份={bot_role}，目标身份={target_role}，"
                f"不能执行 {action_type}。"
            )
            return base_result

        base_result.decision = "unknown"
        base_result.reason = (
            f"目标身份未知：机器人身份={bot_role}，目标身份={target_role}，"
            "交给真实API调用决定。"
        )
        return base_result

    async def _get_self_id(self) -> str:
        if self._self_id_cache:
            return self._self_id_cache

        method = getattr(self.role_provider, "get_login_user_id", None)

        if not callable(method):
            return ""

        try:
            value = await method()
        except Exception:
            return ""

        self_id = normalize_id(value)

        if self_id:
            self._self_id_cache = self_id

        return self_id

    async def _get_member_role(
        self,
        *,
        group_id: str,
        user_id: str,
    ) -> GroupMemberRole:
        group_id = normalize_id(group_id)
        user_id = normalize_id(user_id)

        if not group_id or not user_id:
            return "unknown"

        cache_key = (group_id, user_id)

        if cache_key in self._role_cache:
            return self._role_cache[cache_key]

        method = getattr(self.role_provider, "get_group_member_role", None)

        if not callable(method):
            return "unknown"

        try:
            role = await method(group_id=group_id, user_id=user_id)
        except Exception:
            role = "unknown"

        normalized_role = normalize_group_member_role(role)
        self._role_cache[cache_key] = normalized_role
        return normalized_role

    async def _get_target_role(
        self,
        *,
        bundle: MessageBundle,
        target_user_id: str,
    ) -> GroupMemberRole:
        """
        Prefer role from collected messages first.
        If unavailable, ask OneBot/NapCat.
        """

        target_user_id = normalize_id(target_user_id)

        if not target_user_id:
            return "unknown"

        for message in bundle.messages:
            if message.user_id == target_user_id:
                return sender_role_to_group_member_role(message.role)

        return await self._get_member_role(
            group_id=bundle.group_id,
            user_id=target_user_id,
        )

    def clear_cache(self) -> None:
        self._self_id_cache = None
        self._role_cache.clear()


def sender_role_to_group_member_role(role: SenderRole | str) -> GroupMemberRole:
    return normalize_group_member_role(role)


def normalize_group_member_role(value: Any) -> GroupMemberRole:
    text = str(value or "").strip().lower()

    if text in {"owner", "creator", "群主"}:
        return "owner"

    if text in {"admin", "administrator", "管理员"}:
        return "admin"

    if text in {"member", "normal", "普通群员", "群员"}:
        return "member"

    return "unknown"


def can_manage_target(
    *,
    bot_role: GroupMemberRole,
    target_role: GroupMemberRole,
) -> PermissionDecision:
    """
    Conservative QQ group management rules:

    - 普通群员 cannot manage anyone.
    - 管理员 can usually manage ordinary members only.
    - 群主 can usually manage admins and ordinary members.
    - Nobody should manage the group owner.
    - Unknown target role means we do not block; we let the real API decide.
    """

    if bot_role == "member":
        return "blocked"

    if bot_role == "unknown":
        return "unknown"

    if target_role == "unknown":
        return "unknown"

    if target_role == "owner":
        return "blocked"

    if bot_role == "admin":
        if target_role == "member":
            return "allowed"

        return "blocked"

    if bot_role == "owner":
        if target_role in {"admin", "member"}:
            return "allowed"

        return "blocked"

    return "unknown"


__all__ = [
    "GroupMemberRole",
    "PermissionDecision",
    "PermissionCheckResult",
    "ModerationPermissionChecker",
    "normalize_group_member_role",
    "can_manage_target",
]