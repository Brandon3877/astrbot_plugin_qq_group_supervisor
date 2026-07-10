# data_models.py

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal


MonitorMode = Literal["whitelist", "blacklist", "all"]

SenderRole = Literal["member", "admin", "owner", "unknown"]

RecordRoleMode = Literal[
    "members_only",
    "members_and_admins",
    "members_admins_owner",
]

PunishmentType = Literal["warn", "recall", "mute", "kick"]

Severity = Literal["none", "low", "medium", "high", "critical"]


@dataclass
class RuntimeConfig:
    """
    This contains normalized runtime config.

    config_loader.py will convert the raw config from WebUI to this cleaned format.
    """

    enabled: bool = False

    monitor_mode: MonitorMode = "whitelist"
    group_id_list: set[str] = field(default_factory=set)

    bundle_message_limit: int = 50
    bundle_time_limit_seconds: int = 600
    min_messages_to_analyze: int = 1

    max_record_group_level: int = 30
    record_role_mode: RecordRoleMode = "members_only"
    ignore_self_messages: bool = True
    ignore_empty_messages: bool = True
    ignore_command_messages: bool = False
    command_prefixes: tuple[str, ...] = ("/", "#", "*", "%")

    provider_id: str = ""
    temperature: float = 0.2
    max_tokens: int = 4096

    enable_actions: bool = False
    max_actions_per_bundle: int = 6
    allow_multi_action_same_target: bool = True

    admin_review_enabled: bool = True
    admin_review_timeout_minutes: int = 30
    notify_admin_when_no_action: bool = True

    dry_run: bool = True
    log_collected_messages: bool = False
    log_llm_raw_response: bool = True


@dataclass
class CollectedMessage:
    """
    A single group message after filtering and normalization.

    This is the object to keep in the group buffer.
    """

    group_id: str
    message_id: str
    user_id: str
    nickname: str
    role: SenderRole
    group_level: int
    timestamp: float
    text: str

    def to_llm_text(self) -> str:
        role_text = {
            "member": "普通群员",
            "admin": "管理员",
            "owner": "群主",
            "unknown": "未知",
        }.get(self.role, "未知")

        return (
            f"昵称: {self.nickname}\n"
            f"用户ID: {self.user_id}\n"
            f"群身份: {role_text}\n"
            f"群等级: {self.group_level}\n"
            f"消息ID: {self.message_id}\n"
            f"消息内容: {self.text}"
        )

    def to_safe_dict(self) -> dict[str, Any]:
        """
        Safe serializable version.

        Useful for logs, LLM prompt building, and admin review storage.
        """
        return asdict(self)


@dataclass
class PunishmentOption:
    """
    One punishment option configured by the AstrBot user.

    The user controls the order in WebUI.
    Plugin generates punishment_id from the order: p1, p2, p3...
    """

    punishment_id: str
    type: PunishmentType
    display_name: str

    warning_text: str | None = None
    quote_trigger_message: bool = True
    rewrite_by_llm: bool = False

    duration_minutes: int | None = None

    blacklist: bool = False

    def to_llm_dict(self) -> dict[str, Any]:
        """
        Give the LLM only the fields it needs.

        The LLM may choose punishment_id, but must not invent action type,
        mute duration, or blacklist option.
        """
        data: dict[str, Any] = {
            "punishment_id": self.punishment_id,
            "type": self.type,
            "display_name": self.display_name,
        }

        if self.type == "warn":
            data["warning_text"] = self.warning_text or ""
            data["quote_trigger_message"] = self.quote_trigger_message
            data["rewrite_by_llm"] = self.rewrite_by_llm

        elif self.type == "mute":
            data["duration_minutes"] = self.duration_minutes

        elif self.type == "kick":
            data["blacklist"] = self.blacklist

        return data

    def to_admin_text(self) -> str:
        if self.type == "warn":
            quote = "引用触发消息" if self.quote_trigger_message else "不引用触发消息"
            mode = "LLM改写" if self.rewrite_by_llm else "直接发送"
            return f"{self.display_name}：{self.warning_text or ''}（{quote}, {mode}）"

        if self.type == "recall":
            return f"{self.display_name}：撤回消息"

        if self.type == "mute":
            return f"{self.display_name}：禁言 {self.duration_minutes} 分钟"

        if self.type == "kick":
            return f"{self.display_name}：踢出群聊，拉黑={self.blacklist}"

        return self.display_name


@dataclass
class GroupRule:
    """
    A rule text selected for a specific group.

    Later config_loader.py will choose:
    - group-specific rule first
    - otherwise general rule
    - otherwise built-in fallback rule
    """

    display_name: str
    group_id: str | None
    rule_text: str


@dataclass
class PluginAdmin:
    """
    Admin selected for receiving review messages.

    Later config_loader.py will choose:
    - group-specific admin first
    - otherwise general admin
    """

    display_name: str
    group_id: str | None
    admin_qq: str

    # Reserves specific group only auto execute / request execute.
    # True / False means override, while None means follow general.
    review_before_execute: bool | None = None


@dataclass
class MessageBundle:
    """
    A complete bundle sent to the LLM.
    """

    bundle_id: str
    group_id: str
    created_at: float
    messages: list[CollectedMessage]
    group_rule: GroupRule
    available_punishments: list[PunishmentOption]

    def message_ids(self) -> set[str]:
        return {m.message_id for m in self.messages}

    def user_ids(self) -> set[str]:
        return {m.user_id for m in self.messages}

    def find_message(self, message_id: str) -> CollectedMessage | None:
        for message in self.messages:
            if message.message_id == message_id:
                return message
        return None

    def find_user_messages(self, user_id: str) -> list[CollectedMessage]:
        return [m for m in self.messages if m.user_id == user_id]

    def punishment_map(self) -> dict[str, PunishmentOption]:
        return {p.punishment_id: p for p in self.available_punishments}

    def to_llm_prompt_context(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "group_id": self.group_id,
            "group_rule": self.group_rule.rule_text,
            "available_punishments": [
                p.to_llm_dict() for p in self.available_punishments
            ],
            "messages": [
                {
                    "nickname": m.nickname,
                    "user_id": m.user_id,
                    "role": m.role,
                    "group_level": m.group_level,
                    "message_id": m.message_id,
                    "text": m.text,
                }
                for m in self.messages
            ],
        }


@dataclass
class MessageDiagnosis:
    """
    Diagnosis for a single message, parsed from the LLM result.
    """

    message_id: str
    user_id: str
    severity: Severity
    rule_violated: str
    summary: str


@dataclass
class UserDiagnosis:
    """
    Diagnosis for a user across the whole bundle.
    """

    user_id: str
    severity: Severity
    summary: str


@dataclass
class SuggestedAction:
    """
    Raw action suggested by the LLM.

    This is not trusted yet.
    It must pass validation before execution.
    """

    punishment_id: str
    target_user_id: str | None = None
    target_message_id: str | None = None
    based_on_message_ids: list[str] = field(default_factory=list)
    reason: str = ""

    # Only used by warn punishments when rewrite_by_llm is true.
    warning_text: str | None = None


@dataclass
class LLMModerationResult:
    """
    Parsed LLM JSON result.

    This is parsed successfully, but not necessarily validated yet.
    """

    normal: bool
    message_bundle_assessment: str
    message_diagnoses: list[MessageDiagnosis] = field(default_factory=list)
    user_diagnoses: list[UserDiagnosis] = field(default_factory=list)
    actions: list[SuggestedAction] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class ValidatedAction:
    """
    A safe action after validation.

    Only this object should be given to executor.py.
    """

    punishment: PunishmentOption
    target_user_id: str | None = None
    target_message_id: str | None = None
    based_on_message_ids: list[str] = field(default_factory=list)
    reason: str = ""
    warning_text: str | None = None

    def to_admin_text(self, bundle: MessageBundle) -> str:
        target_parts: list[str] = []

        if self.target_user_id:
            user_messages = bundle.find_user_messages(self.target_user_id)
            nickname = user_messages[0].nickname if user_messages else "未知昵称"
            target_parts.append(f"目标用户: {nickname} / {self.target_user_id}")

        if self.target_message_id:
            target_parts.append(f"目标消息ID: {self.target_message_id}")

        target_text = "\n".join(target_parts) if target_parts else "目标: 未指定"

        operation_text = self._action_operation_text()

        return (
            f"操作: {operation_text}\n"
            f"{target_text}\n"
            f"依据消息ID: {', '.join(self.based_on_message_ids) if self.based_on_message_ids else '无'}\n"
            f"理由: {self.reason}"
        )

    def _action_operation_text(self) -> str:
        if self.punishment.type == "warn":
            mode = "LLM改写" if self.punishment.rewrite_by_llm else "直接发送"
            quote = "引用触发消息" if self.punishment.quote_trigger_message else "不引用触发消息"
            text = self.warning_text or self.punishment.warning_text or ""
            return f"{self.punishment.display_name}：{text}（{quote}, {mode}）"

        return self.punishment.to_admin_text()


@dataclass
class ValidationResult:
    """
    Result produced by action_validator.py.
    """

    valid_actions: list[ValidatedAction] = field(default_factory=list)
    rejected_actions: list[str] = field(default_factory=list)

    @property
    def has_valid_actions(self) -> bool:
        return len(self.valid_actions) > 0


@dataclass
class PendingReview:
    """
    Stored when admin review is enabled.

    The admin may later reply 执行 or 取消.
    """

    review_id: str
    group_id: str
    admin_qq: str
    bundle: MessageBundle
    llm_result: LLMModerationResult
    validation_result: ValidationResult
    created_at: float