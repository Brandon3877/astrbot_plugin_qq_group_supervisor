# config_loader.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data_models import (
    GroupRule,
    MonitorMode,
    PluginAdmin,
    PunishmentOption,
    RecordRoleMode,
    RuntimeConfig,
)


DEFAULT_GROUP_RULE = """群聊规范：
1. 拉踩引战：禁止通过贬低、对比等方式挑起群体/个人对立，或故意发布争议性言论诱导争吵。
2. 骚扰他人：禁止未经同意频繁私聊、@他人，或以言语、图片等形式对群友进行持续性打扰、纠缠。
3. 人身攻击：禁止辱骂、嘲讽、造谣、人肉搜索，或使用侮辱性称呼恶意针对特定群友。
4. 恶意刷屏：禁止连续发送无意义内容、重复消息、无意义长文本霸屏，干扰正常交流。
5. 违规推广：禁止发布广告、未经同意分享其他群聊、兼职/引流信息。
6. 传播有害信息：禁止发布涉政敏感、色情暴力、谣言诈骗及其他违反法律法规的内容。"""


MONITOR_MODE_MAP: dict[str, MonitorMode] = {
    "只监控以下群聊": "whitelist",
    "不监控以下群聊": "blacklist",
    "监控所有群聊": "all",
}


RECORD_ROLE_MODE_MAP: dict[str, RecordRoleMode] = {
    "仅普通群员": "members_only",
    "普通群员和管理员": "members_and_admins",
    "群员和管理员": "members_and_admins",
    "群员、管理员和群主": "members_admins_owner",
}


@dataclass
class LoadedConfig:
    """
    Fully normalized plugin config.

    This is what the rest of the entire plugin should depend on.
    """

    runtime: RuntimeConfig
    group_rules: list[GroupRule]
    plugin_admins: list[PluginAdmin]
    punishments: list[PunishmentOption]

    def is_group_monitored(self, group_id: str) -> bool:
        group_id = normalize_id(group_id)
        mode = self.runtime.monitor_mode
        group_ids = self.runtime.group_id_list

        if mode == "all":
            return True

        if mode == "whitelist":
            return group_id in group_ids

        if mode == "blacklist":
            return group_id not in group_ids

        return False

    def get_group_rule(self, group_id: str) -> GroupRule:
        """
        Selection order:
        1. Group-specific rule for this group.
        2. General rule.
        3. Built-in fallback rule.

        If the user accidentally configures multiple matching rules,
        the latest one will be used, because it was created last.
        """
        group_id = normalize_id(group_id)

        for rule in reversed(self.group_rules):
            if rule.group_id == group_id:
                return rule

        for rule in reversed(self.group_rules):
            if rule.group_id is None:
                return rule

        return GroupRule(
            display_name="内置默认群规范",
            group_id=None,
            rule_text=DEFAULT_GROUP_RULE,
        )

    def get_plugin_admin(self, group_id: str) -> PluginAdmin | None:
        """
        Selection order:
        1. Group-specific plugin admin for this group.
        2. General plugin admin.
        3. None.

        Empty QQ numbers are ignored.
        """
        group_id = normalize_id(group_id)

        for admin in reversed(self.plugin_admins):
            if admin.group_id == group_id and admin.admin_qq:
                return admin

        for admin in reversed(self.plugin_admins):
            if admin.group_id is None and admin.admin_qq:
                return admin

        return None
    
    def should_review_before_execute(self, group_id: str) -> bool:
        """
        Selection order:
        1. If this group has a group-specific admin with review_before_execute set,
           use that value.
        2. Otherwise use the global configuration admin_review.enabled.
        """
        group_id = normalize_id(group_id)

        admin = self.get_plugin_admin(group_id)

        if admin is not None and admin.group_id == group_id:
            if admin.review_before_execute is not None:
                return admin.review_before_execute

        return self.runtime.admin_review_enabled


def load_config(raw_config: Any) -> LoadedConfig:
    """
    Main entry point.

    raw_config should be the config object passed by AstrBot into plugin __init__.

    Example:
        self.loaded_config = load_config(config)
    """
    runtime = load_runtime_config(raw_config)
    group_rules = load_group_rules(raw_config)
    plugin_admins = load_plugin_admins(raw_config)
    punishments = load_punishments(raw_config)

    return LoadedConfig(
        runtime=runtime,
        group_rules=group_rules,
        plugin_admins=plugin_admins,
        punishments=punishments,
    )


def load_runtime_config(raw_config: Any) -> RuntimeConfig:
    basic = get_section(raw_config, "basic")
    bundle = get_section(raw_config, "bundle")
    record_filter = get_section(raw_config, "record_filter")
    llm = get_section(raw_config, "llm")
    action_control = get_section(raw_config, "action_control")
    admin_review = get_section(raw_config, "admin_review")
    debug = get_section(raw_config, "debug")

    monitor_mode_text = as_str(
        get_value(basic, "monitor_all_groups", "只监控以下群聊")
    )
    monitor_mode = MONITOR_MODE_MAP.get(monitor_mode_text, "whitelist")

    role_mode_text = as_str(
        get_value(record_filter, "record_role_mode", "仅普通群员")
    )
    record_role_mode = RECORD_ROLE_MODE_MAP.get(role_mode_text, "members_only")

    return RuntimeConfig(
        enabled=as_bool(get_value(basic, "enabled", False), False),
        monitor_mode=monitor_mode,
        group_id_list=normalize_id_set(get_value(basic, "enabled_group_ids", [])),

        bundle_message_limit=as_int(
            get_value(bundle, "bundle_message_limit", 50),
            default=50,
            minimum=0,
        ),
        bundle_time_limit_seconds=as_int(
            get_value(bundle, "bundle_time_limit_seconds", 600),
            default=600,
            minimum=0,
        ),
        min_messages_to_analyze=as_int(
            get_value(bundle, "min_messages_to_analyze", 1),
            default=1,
            minimum=1,
        ),

        max_record_group_level=as_int(
            get_value(record_filter, "max_record_group_level", 30),
            default=30,
            minimum=0,
        ),
        record_role_mode=record_role_mode,
        ignore_self_messages=as_bool(
            get_value(record_filter, "ignore_self_messages", True),
            True,
        ),
        ignore_empty_messages=as_bool(
            get_value(record_filter, "ignore_empty_messages", True),
            True,
        ),
        ignore_command_messages=as_bool(
            get_value(record_filter, "ignore_command_messages", False),
            False,
        ),

        provider_id=as_str(get_value(llm, "provider_id", "")).strip(),
        temperature=as_float(
            get_value(llm, "temperature", 0.2),
            default=0.2,
            minimum=0.0,
        ),
        max_tokens=as_int(
            get_value(llm, "max_tokens", 4096),
            default=4096,
            minimum=1,
        ),

        enable_actions=as_bool(
            get_value(action_control, "enable_actions", False),
            False,
        ),
        max_actions_per_bundle=as_int(
            get_value(action_control, "max_actions_per_bundle", 10),
            default=10,
            minimum=0,
        ),
        allow_multi_action_same_target=as_bool(
            get_value(action_control, "allow_multi_action_same_target", True),
            True,
        ),

        admin_review_enabled=as_bool(
            get_value(admin_review, "enabled", True),
            True,
        ),
        admin_review_timeout_minutes=as_int(
            get_value(admin_review, "review_timeout_minutes", 30),
            default=30,
            minimum=1,
        ),
        notify_admin_when_timeout=as_bool(
            get_value(admin_review, "notify_admin_when_timeout", False),
            False,
        ),
        notify_admin_when_no_action=as_bool(
            get_value(admin_review, "notify_admin_when_no_action", True),
            True,
        ),

        dry_run=as_bool(get_value(debug, "dry_run", False), False),
        log_collected_messages=as_bool(
            get_value(debug, "log_collected_messages", False),
            False,
        ),
        log_llm_raw_response=as_bool(
            get_value(debug, "log_llm_raw_response", True),
            True,
        ),
    )


def load_group_rules(raw_config: Any) -> list[GroupRule]:
    llm = get_section(raw_config, "llm")
    raw_rules = get_value(llm, "group_rules", [])

    if not isinstance(raw_rules, list):
        raw_rules = []

    result: list[GroupRule] = []

    for item in raw_rules:
        if not isinstance(item, dict):
            continue

        template_key = as_str(get_value(item, "__template_key", ""))

        group_number = normalize_id(get_value(item, "group_number", ""))
        rule_text = as_str(get_value(item, "group_rule", "")).strip()
        display_name = as_str(get_value(item, "display_name", "")).strip()

        if not rule_text:
            continue

        if template_key == "group_specific" or group_number:
            if not group_number:
                continue

            result.append(
                GroupRule(
                    display_name=display_name or "专属群规范",
                    group_id=group_number,
                    rule_text=rule_text,
                )
            )
        else:
            result.append(
                GroupRule(
                    display_name=display_name or "通用群规范",
                    group_id=None,
                    rule_text=rule_text,
                )
            )

    has_general_rule = any(rule.group_id is None for rule in result)

    if not has_general_rule:
        result.append(
            GroupRule(
                display_name="内置默认群规范",
                group_id=None,
                rule_text=DEFAULT_GROUP_RULE,
            )
        )

    return result


def load_plugin_admins(raw_config: Any) -> list[PluginAdmin]:
    admin_review = get_section(raw_config, "admin_review")
    raw_admins = get_value(admin_review, "admin_qq", [])

    if not isinstance(raw_admins, list):
        raw_admins = []

    result: list[PluginAdmin] = []

    for item in raw_admins:
        if not isinstance(item, dict):
            continue

        template_key = as_str(get_value(item, "__template_key", ""))

        group_number = normalize_id(get_value(item, "group_number", ""))
        admin_qq = normalize_id(get_value(item, "admin_qq_number", ""))
        display_name = as_str(get_value(item, "display_name", "")).strip()

        if not admin_qq:
            continue

        if template_key == "group_specific" or group_number:
            if not group_number:
                continue

            result.append(
                PluginAdmin(
                    display_name=display_name or "专属管理员",
                    group_id=group_number,
                    admin_qq=admin_qq,
                    review_before_execute=as_bool(
                        get_value(item, "review_before_execute", True),
                        True,
                    ),
                )
            )
        else:
            result.append(
                PluginAdmin(
                    display_name=display_name or "通用管理员",
                    group_id=None,
                    admin_qq=admin_qq,
                    review_before_execute=None,
                )
            )

    return result


def load_punishments(raw_config: Any) -> list[PunishmentOption]:
    raw_punishments = get_value(raw_config, "punishments", [])

    if not isinstance(raw_punishments, list):
        raw_punishments = []

    result: list[PunishmentOption] = []

    for item in raw_punishments:
        if not isinstance(item, dict):
            continue

        template_key = as_str(get_value(item, "__template_key", "")).strip()
        if template_key not in {"warn", "recall", "mute", "kick"}:
            continue

        punishment_id = f"p{len(result) + 1}"

        if template_key == "warn":
            display_name = as_str(
                get_value(item, "display_name", "警告")
            ).strip() or "警告"

            warning_text = as_str(
                get_value(
                    item,
                    "warning_text",
                    "请注意你的言辞，避免继续影响群聊秩序。",
                )
            ).strip()

            if not warning_text:
                warning_text = "请注意你的言辞，避免继续影响群聊秩序。"
            
            result.append(
                PunishmentOption(
                    punishment_id=punishment_id,
                    type="warn",
                    display_name=display_name,
                    warning_text=warning_text,
                    quote_trigger_message=as_bool(
                        get_value(item, "quote_trigger_message", True),
                        True,
                    ),
                    rewrite_by_llm=as_bool(
                        get_value(item, "rewrite_by_llm", False),
                        False,
                    ),
                )
            )

        elif template_key == "recall":
            display_name = as_str(
                get_value(item, "display_name", "撤回")
            ).strip() or "撤回"

            result.append(
                PunishmentOption(
                    punishment_id=punishment_id,
                    type="recall",
                    display_name=display_name,
                )
            )

        elif template_key == "mute":
            display_name = as_str(
                get_value(item, "display_name", "禁言")
            ).strip() or "禁言"

            duration_minutes = as_int(
                get_value(item, "duration_minutes", 5),
                default=5,
                minimum=1,
            )

            result.append(
                PunishmentOption(
                    punishment_id=punishment_id,
                    type="mute",
                    display_name=display_name,
                    duration_minutes=duration_minutes,
                )
            )

        elif template_key == "kick":
            display_name = as_str(
                get_value(item, "display_name", "踢走")
            ).strip() or "踢走"

            result.append(
                PunishmentOption(
                    punishment_id=punishment_id,
                    type="kick",
                    display_name=display_name,
                    blacklist=as_bool(
                        get_value(item, "blacklist", True),
                        True,
                    ),
                )
            )

    return result


def get_section(config: Any, key: str) -> Any:
    value = get_value(config, key, {})
    return value if isinstance(value, dict) else {}


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    get_method = getattr(obj, "get", None)
    if callable(get_method):
        try:
            return get_method(key, default)
        except TypeError:
            try:
                value = get_method(key)
                return default if value is None else value
            except Exception:
                return default
        except Exception:
            return default

    return getattr(obj, key, default)


def normalize_id(value: Any) -> str:
    """
    QQ IDs and group IDs are handled as strings for consistency inside plugin.
    """
    if value is None:
        return ""

    return str(value).strip()


def normalize_id_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()

    result: set[str] = set()

    for item in value:
        normalized = normalize_id(item)
        if normalized:
            result.add(normalized)

    return result


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value)


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, int):
        return value != 0

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered in {"true", "yes", "y", "1", "on", "开启", "是"}:
            return True

        if lowered in {"false", "no", "n", "0", "off", "关闭", "否"}:
            return False

    return default


def as_int(value: Any, default: int = 0, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default

    if minimum is not None and result < minimum:
        return minimum

    return result


def as_float(value: Any, default: float = 0.0, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default

    if minimum is not None and result < minimum:
        return minimum

    return result


__all__ = [
    "DEFAULT_GROUP_RULE",
    "LoadedConfig",
    "load_config",
    "load_runtime_config",
    "load_group_rules",
    "load_plugin_admins",
    "load_punishments",
]