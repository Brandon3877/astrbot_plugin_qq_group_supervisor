# action_validator.py

from __future__ import annotations

from dataclasses import dataclass

from .data_models import (
    LLMModerationResult,
    MessageBundle,
    PunishmentOption,
    RuntimeConfig,
    SuggestedAction,
    ValidatedAction,
    ValidationResult,
)


@dataclass
class _CandidateAction:
    """
    Internal wrapper.

    The original LLM action index is kept,
    so rejections are easy to debug.
    """

    index: int
    action: ValidatedAction
    punishment_order: int


def validate_llm_result(
    bundle: MessageBundle,
    llm_result: LLMModerationResult,
    runtime: RuntimeConfig,
) -> ValidationResult:
    """
    Validate parsed LLM output against the original bundle and runtime config.

    This function checks:
    - punishment_id exists in the configured punishment list
    - target_user_id exists in the bundle
    - target_message_id exists in the bundle
    - based_on_message_ids exist in the bundle
    - recall/warn/mute/kick have the required targets
    - warning_text obeys the configured warn punishment
    - max_actions_per_bundle
    - allow_multi_action_same_target

    This function does not execute anything.
    Only ValidatedAction objects should then be given to executor.py.
    """

    result = ValidationResult()

    _validate_diagnosis_ids(bundle, llm_result, result)

    if llm_result.normal:
        if llm_result.actions:
            for index, _action in enumerate(llm_result.actions):
                result.rejected_actions.append(
                    f"actions[{index}]: rejected because normal=true but actions is not empty."
                )
        return result

    punishment_map = bundle.punishment_map()
    punishment_order_map = {
        punishment.punishment_id: index + 1
        for index, punishment in enumerate(bundle.available_punishments)
    }

    candidates: list[_CandidateAction] = []

    for index, suggested_action in enumerate(llm_result.actions):
        candidate, reject_reason = _validate_single_action(
            index=index,
            suggested_action=suggested_action,
            bundle=bundle,
            punishment_map=punishment_map,
            punishment_order_map=punishment_order_map,
        )

        if candidate is None:
            result.rejected_actions.append(f"actions[{index}]: {reject_reason}")
            continue

        candidates.append(candidate)

    candidates = _apply_multi_action_policy(
        candidates=candidates,
        runtime=runtime,
        rejected_actions=result.rejected_actions,
    )

    candidates = _apply_max_action_limit(
        candidates=candidates,
        runtime=runtime,
        rejected_actions=result.rejected_actions,
    )

    result.valid_actions = [candidate.action for candidate in candidates]
    return result


def _validate_single_action(
    *,
    index: int,
    suggested_action: SuggestedAction,
    bundle: MessageBundle,
    punishment_map: dict[str, PunishmentOption],
    punishment_order_map: dict[str, int],
) -> tuple[_CandidateAction | None, str]:
    punishment_id = normalize_id(suggested_action.punishment_id)

    if not punishment_id:
        return None, "missing punishment_id."

    punishment = punishment_map.get(punishment_id)

    if punishment is None:
        return None, f"unknown punishment_id: {punishment_id}."

    punishment_order = punishment_order_map.get(punishment_id, 999999)

    target_user_id = normalize_optional_id(suggested_action.target_user_id)
    target_message_id = normalize_optional_id(suggested_action.target_message_id)
    based_on_message_ids = normalize_id_list(suggested_action.based_on_message_ids)

    if punishment.type == "recall":
        valid, reason, target_user_id, target_message_id = _validate_recall_target(
            bundle=bundle,
            target_user_id=target_user_id,
            target_message_id=target_message_id,
        )

        if not valid:
            return None, reason

    elif punishment.type in {"warn", "mute", "kick"}:
        valid, reason = _validate_user_target(
            bundle=bundle,
            punishment=punishment,
            target_user_id=target_user_id,
            target_message_id=target_message_id,
        )

        if not valid:
            return None, reason

    else:
        return None, f"unsupported punishment type: {punishment.type}."

    if target_message_id and not based_on_message_ids:
        based_on_message_ids = [target_message_id]

    valid, reason = _validate_based_on_messages(
        bundle=bundle,
        target_user_id=target_user_id,
        based_on_message_ids=based_on_message_ids,
    )

    if not valid:
        return None, reason

    warning_text = _resolve_warning_text(
        punishment=punishment,
        suggested_action=suggested_action,
    )

    validated_action = ValidatedAction(
        punishment=punishment,
        target_user_id=target_user_id,
        target_message_id=target_message_id,
        based_on_message_ids=based_on_message_ids,
        reason=suggested_action.reason.strip(),
        warning_text=warning_text,
    )

    return (
        _CandidateAction(
            index=index,
            action=validated_action,
            punishment_order=punishment_order,
        ),
        "",
    )


def _validate_recall_target(
    *,
    bundle: MessageBundle,
    target_user_id: str | None,
    target_message_id: str | None,
) -> tuple[bool, str, str | None, str | None]:
    if not target_message_id:
        return False, "recall requires target_message_id.", target_user_id, target_message_id

    message = bundle.find_message(target_message_id)

    if message is None:
        return (
            False,
            f"target_message_id does not exist in bundle: {target_message_id}.",
            target_user_id,
            target_message_id,
        )

    if target_user_id and target_user_id != message.user_id:
        return (
            False,
            (
                "target_user_id does not match the sender of target_message_id: "
                f"user={target_user_id}, message_sender={message.user_id}."
            ),
            target_user_id,
            target_message_id,
        )

    return True, "", message.user_id, target_message_id


def _validate_user_target(
    *,
    bundle: MessageBundle,
    punishment: PunishmentOption,
    target_user_id: str | None,
    target_message_id: str | None,
) -> tuple[bool, str]:
    if not target_user_id:
        return False, f"{punishment.type} requires target_user_id."

    if target_user_id not in bundle.user_ids():
        return False, f"target_user_id does not exist in bundle: {target_user_id}."

    if punishment.type == "mute":
        if punishment.duration_minutes is None or punishment.duration_minutes <= 0:
            return False, f"mute punishment has invalid duration_minutes: {punishment.duration_minutes}."

    if target_message_id:
        message = bundle.find_message(target_message_id)

        if message is None:
            return False, f"target_message_id does not exist in bundle: {target_message_id}."

        if message.user_id != target_user_id:
            return (
                False,
                (
                    "target_message_id sender does not match target_user_id: "
                    f"user={target_user_id}, message_sender={message.user_id}."
                ),
            )

    return True, ""


def _validate_based_on_messages(
    *,
    bundle: MessageBundle,
    target_user_id: str | None,
    based_on_message_ids: list[str],
) -> tuple[bool, str]:
    if not based_on_message_ids:
        return False, "based_on_message_ids is empty."

    for message_id in based_on_message_ids:
        message = bundle.find_message(message_id)

        if message is None:
            return False, f"based_on_message_id does not exist in bundle: {message_id}."

        if target_user_id and message.user_id != target_user_id:
            return (
                False,
                (
                    "based_on_message_id sender does not match target_user_id: "
                    f"message_id={message_id}, user={target_user_id}, "
                    f"message_sender={message.user_id}."
                ),
            )

    return True, ""


def _resolve_warning_text(
    *,
    punishment: PunishmentOption,
    suggested_action: SuggestedAction,
) -> str | None:
    if punishment.type != "warn":
        return None

    configured_text = (punishment.warning_text or "").strip()

    if not punishment.rewrite_by_llm:
        return configured_text

    llm_text = suggested_action.warning_text.strip() if suggested_action.warning_text else ""

    # Good case: LLM actually generated a rewritten warning.
    if llm_text and not _is_same_warning_text(llm_text, configured_text):
        return llm_text

    # Robust fallback: if LLM forgot to rewrite, create a slightly expanded
    # warning based on the action reason instead of sending only the original text.
    return _build_fallback_rewritten_warning_text(
        configured_text=configured_text,
        reason=suggested_action.reason,
    )


def _is_same_warning_text(text_a: str, text_b: str) -> bool:
    """
    Loose equality check for warning text.

    This avoids treating a tiny whitespace/punctuation difference as a real rewrite.
    """
    normalized_a = _normalize_warning_text(text_a)
    normalized_b = _normalize_warning_text(text_b)

    return normalized_a == normalized_b


def _normalize_warning_text(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("，", ",")
        .replace("。", ".")
        .replace("！", "!")
        .replace("？", "?")
    )


def _build_fallback_rewritten_warning_text(
    *,
    configured_text: str,
    reason: str,
) -> str:
    reason = reason.strip()

    if reason:
        return f"你的发言不符合群聊规范：{reason}。{configured_text}"

    return f"你的发言可能不符合群聊规范。{configured_text}"


def _validate_diagnosis_ids(
    bundle: MessageBundle,
    llm_result: LLMModerationResult,
    result: ValidationResult,
) -> None:
    """
    Diagnosis fields are not executed, so invalid diagnosis IDs do not directly
    block action validation.
    """

    known_message_ids = bundle.message_ids()
    known_user_ids = bundle.user_ids()

    for index, diagnosis in enumerate(llm_result.message_diagnoses):
        if diagnosis.message_id not in known_message_ids:
            result.rejected_actions.append(
                f"message_diagnoses[{index}]: unknown message_id {diagnosis.message_id}."
            )

        if diagnosis.user_id not in known_user_ids:
            result.rejected_actions.append(
                f"message_diagnoses[{index}]: unknown user_id {diagnosis.user_id}."
            )

        message = bundle.find_message(diagnosis.message_id)
        if message is not None and message.user_id != diagnosis.user_id:
            result.rejected_actions.append(
                (
                    f"message_diagnoses[{index}]: user_id does not match message sender. "
                    f"user={diagnosis.user_id}, message_sender={message.user_id}."
                )
            )

    for index, diagnosis in enumerate(llm_result.user_diagnoses):
        if diagnosis.user_id not in known_user_ids:
            result.rejected_actions.append(
                f"user_diagnoses[{index}]: unknown user_id {diagnosis.user_id}."
            )


def _apply_multi_action_policy(
    *,
    candidates: list[_CandidateAction],
    runtime: RuntimeConfig,
    rejected_actions: list[str],
) -> list[_CandidateAction]:
    """
    If allow_multi_action_same_target is true:
    - keep all valid candidates

    If false:
    - prefer recall actions first
    - otherwise prefer the lightest configured punishment
    - allow only one selected action touching the same user or message
    """

    if runtime.allow_multi_action_same_target:
        return candidates

    selected: list[_CandidateAction] = []
    used_conflict_keys: set[str] = set()

    for candidate in sorted(candidates, key=_single_action_priority):
        keys = _conflict_keys(candidate.action)

        if used_conflict_keys.intersection(keys):
            rejected_actions.append(
                (
                    f"actions[{candidate.index}]: rejected by "
                    "allow_multi_action_same_target=false conflict policy."
                )
            )
            continue

        selected.append(candidate)
        used_conflict_keys.update(keys)

    return sorted(selected, key=lambda item: item.index)


def _single_action_priority(candidate: _CandidateAction) -> tuple[int, int, int]:
    punishment = candidate.action.punishment

    if punishment.type == "recall":
        priority_group = 0
    else:
        priority_group = 1

    return (
        priority_group,
        candidate.punishment_order,
        candidate.index,
    )


def _conflict_keys(action: ValidatedAction) -> set[str]:
    keys: set[str] = set()

    if action.target_user_id:
        keys.add(f"user:{action.target_user_id}")

    if action.target_message_id:
        keys.add(f"message:{action.target_message_id}")

    for message_id in action.based_on_message_ids:
        keys.add(f"message:{message_id}")

    return keys


def _apply_max_action_limit(
    *,
    candidates: list[_CandidateAction],
    runtime: RuntimeConfig,
    rejected_actions: list[str],
) -> list[_CandidateAction]:
    max_actions = runtime.max_actions_per_bundle

    if max_actions <= 0:
        return candidates

    if len(candidates) <= max_actions:
        return candidates

    kept = candidates[:max_actions]
    removed = candidates[max_actions:]

    for candidate in removed:
        rejected_actions.append(
            f"actions[{candidate.index}]: rejected by max_actions_per_bundle={max_actions}."
        )

    return kept


def normalize_id(value: object) -> str:
    if value is None:
        return ""

    return str(value).strip()


def normalize_optional_id(value: object) -> str | None:
    text = normalize_id(value)

    if not text:
        return None

    if text.lower() == "null":
        return None

    return text


def normalize_id_list(values: list[str]) -> list[str]:
    """
    Normalize ID list and remove duplicates while preserving order.
    """

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = normalize_id(value)

        if not text:
            continue

        if text.lower() == "null":
            continue

        if text in seen:
            continue

        seen.add(text)
        result.append(text)

    return result


__all__ = [
    "validate_llm_result",
]