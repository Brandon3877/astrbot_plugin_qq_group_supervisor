# llm_parser.py

from __future__ import annotations

import json
from typing import Any, cast

from .data_models import (
    LLMModerationResult,
    MessageDiagnosis,
    Severity,
    SuggestedAction,
    UserDiagnosis,
)


ALLOWED_SEVERITIES: set[str] = {
    "none",
    "low",
    "medium",
    "high",
    "critical",
}


class LLMParseError(Exception):
    """
    Raised when the LLM response cannot be parsed into our expected structure.
    """

    pass


def parse_llm_response(
    raw_response: str,
    *,
    allow_json_extraction: bool = True,
) -> LLMModerationResult:
    """
    Parse raw LLM output into LLMModerationResult.

    The prompt asks the LLM to output JSON only. However, during development,
    models may still add Markdown fences or a short explanation. Therefore:

    1. Try direct JSON parsing first.
    2. If that fails and allow_json_extraction=True, try to extract the first
       valid JSON object from the response.
    3. Convert the JSON object into our dataclasses.
    4. Raise LLMParseError if the shape is too broken.

    It does NOT validate whether user_id/message_id/punishment_id really exist in
    the original bundle. That will be checked against inside action_validator.py.
    """

    if not isinstance(raw_response, str):
        raise LLMParseError("LLM response is not a string.")

    cleaned = raw_response.strip()

    if not cleaned:
        raise LLMParseError("LLM response is empty.")

    data = _loads_json_object(cleaned, allow_json_extraction=allow_json_extraction)

    return _parse_result_object(data, raw_response=raw_response)


def _loads_json_object(
    text: str,
    *,
    allow_json_extraction: bool,
) -> dict[str, Any]:
    """
    Load a JSON object from text.

    Plugin first tries strict json.loads().
    If that fails, we optionally try:
    - Markdown code fence removal
    - JSON object extraction using JSONDecoder.raw_decode()
    """

    try:
        data = json.loads(text)
        return _ensure_dict(data, "Top-level JSON must be an object.")
    except json.JSONDecodeError:
        pass

    unfenced = _strip_markdown_json_fence(text)

    if unfenced != text:
        try:
            data = json.loads(unfenced)
            return _ensure_dict(data, "Top-level JSON inside code fence must be an object.")
        except json.JSONDecodeError:
            pass

    if allow_json_extraction:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            return extracted

    raise LLMParseError("Failed to parse LLM response as JSON object.")


def _strip_markdown_json_fence(text: str) -> str:
    """
    Handles responses like:

        ```json
        {...}
        ```

    or:

        ```
        {...}
        ```
    """

    stripped = text.strip()

    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()

    if len(lines) < 3:
        return text

    first_line = lines[0].strip().lower()
    last_line = lines[-1].strip()

    if not first_line.startswith("```"):
        return text

    if last_line != "```":
        return text

    return "\n".join(lines[1:-1]).strip()


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """
    Tries to find the first valid JSON object inside a longer string.

    This is intentionally only a recovery mechanism.
    In normal operation, the LLM should output JSON only.
    """

    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue

        try:
            data, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            return data

    return None


def _parse_result_object(
    data: dict[str, Any],
    *,
    raw_response: str,
) -> LLMModerationResult:
    _require_keys(
        data,
        [
            "normal",
            "message_bundle_assessment",
            "message_diagnoses",
            "user_diagnoses",
            "actions",
        ],
        location="top-level result",
    )

    normal = _parse_bool(data["normal"], "normal")
    assessment = _parse_str(data["message_bundle_assessment"], "message_bundle_assessment")

    message_diagnoses = _parse_message_diagnoses(data["message_diagnoses"])
    user_diagnoses = _parse_user_diagnoses(data["user_diagnoses"])
    actions = _parse_actions(data["actions"])

    return LLMModerationResult(
        normal=normal,
        message_bundle_assessment=assessment,
        message_diagnoses=message_diagnoses,
        user_diagnoses=user_diagnoses,
        actions=actions,
        raw_response=raw_response,
    )


def _parse_message_diagnoses(value: Any) -> list[MessageDiagnosis]:
    items = _parse_list(value, "message_diagnoses")
    result: list[MessageDiagnosis] = []

    for index, item in enumerate(items):
        location = f"message_diagnoses[{index}]"
        obj = _ensure_dict(item, f"{location} must be an object.")

        _require_keys(
            obj,
            ["message_id", "user_id", "severity", "rule_violated", "summary"],
            location=location,
        )

        result.append(
            MessageDiagnosis(
                message_id=_parse_str(obj["message_id"], f"{location}.message_id"),
                user_id=_parse_str(obj["user_id"], f"{location}.user_id"),
                severity=_parse_severity(obj["severity"], f"{location}.severity"),
                rule_violated=_parse_str(
                    obj["rule_violated"],
                    f"{location}.rule_violated",
                ),
                summary=_parse_str(obj["summary"], f"{location}.summary"),
            )
        )

    return result


def _parse_user_diagnoses(value: Any) -> list[UserDiagnosis]:
    items = _parse_list(value, "user_diagnoses")
    result: list[UserDiagnosis] = []

    for index, item in enumerate(items):
        location = f"user_diagnoses[{index}]"
        obj = _ensure_dict(item, f"{location} must be an object.")

        _require_keys(
            obj,
            ["user_id", "severity", "summary"],
            location=location,
        )

        result.append(
            UserDiagnosis(
                user_id=_parse_str(obj["user_id"], f"{location}.user_id"),
                severity=_parse_severity(obj["severity"], f"{location}.severity"),
                summary=_parse_str(obj["summary"], f"{location}.summary"),
            )
        )

    return result


def _parse_actions(value: Any) -> list[SuggestedAction]:
    items = _parse_list(value, "actions")
    result: list[SuggestedAction] = []

    for index, item in enumerate(items):
        location = f"actions[{index}]"
        obj = _ensure_dict(item, f"{location} must be an object.")

        _require_keys(
            obj,
            [
                "punishment_id",
                "target_user_id",
                "target_message_id",
                "based_on_message_ids",
                "reason",
                "warning_text",
            ],
            location=location,
        )

        based_on_message_ids = _parse_str_list(
            obj["based_on_message_ids"],
            f"{location}.based_on_message_ids",
        )

        result.append(
            SuggestedAction(
                punishment_id=_parse_str(
                    obj["punishment_id"],
                    f"{location}.punishment_id",
                ),
                target_user_id=_parse_optional_str_id(
                    obj["target_user_id"],
                    f"{location}.target_user_id",
                ),
                target_message_id=_parse_optional_str_id(
                    obj["target_message_id"],
                    f"{location}.target_message_id",
                ),
                based_on_message_ids=based_on_message_ids,
                reason=_parse_str(obj["reason"], f"{location}.reason"),
                warning_text=_parse_optional_warning_text(
                    obj["warning_text"],
                    f"{location}.warning_text",
                ),
            )
        )

    return result


def _require_keys(
    obj: dict[str, Any],
    keys: list[str],
    *,
    location: str,
) -> None:
    missing = [key for key in keys if key not in obj]

    if missing:
        raise LLMParseError(
            f"Missing required key(s) in {location}: {', '.join(missing)}"
        )


def _ensure_dict(value: Any, error_message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LLMParseError(error_message)

    return value


def _parse_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise LLMParseError(f"{location} must be a list.")

    return value


def _parse_str(value: Any, location: str) -> str:
    if value is None:
        raise LLMParseError(f"{location} must be a string, got null.")

    if isinstance(value, (dict, list)):
        raise LLMParseError(f"{location} must be a string, got {type(value).__name__}.")

    return str(value).strip()


def _parse_optional_str(value: Any, location: str) -> str | None:
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        raise LLMParseError(f"{location} must be a string or null.")

    text = str(value).strip()

    if not text:
        return None

    return text


def _parse_optional_warning_text(value: Any, location: str) -> str | None:
    """
    Parse warning_text.

    Required ideal format:
        "warning_text": "..."

    Tolerated recovery format:
        "warning_text": ["..."]

    This tolerance is only used on warning_text, because it is non-destructive
    text content. For other fields, this tolerence is not allowed.
    """

    if value is None:
        return None

    if isinstance(value, list):
        parts: list[str] = []

        for index, item in enumerate(value):
            item_location = f"{location}[{index}]"

            if item is None:
                continue

            if isinstance(item, (dict, list)):
                raise LLMParseError(
                    f"{item_location} must be a string if warning_text is a list."
                )

            text = str(item).strip()

            if text:
                parts.append(text)

        if not parts:
            return None

        return "\n".join(parts)

    if isinstance(value, dict):
        raise LLMParseError(f"{location} must be a string, null, or list of strings.")

    text = str(value).strip()

    if not text:
        return None

    if text.lower() == "null":
        return None

    return text


def _parse_optional_str_id(value: Any, location: str) -> str | None:
    """
    Parse optional IDs.

    JSON null, empty string, and "null" are treated as None.
    """

    if value is None:
        return None

    if isinstance(value, (dict, list)):
        raise LLMParseError(f"{location} must be a string/number/null.")

    text = str(value).strip()

    if not text:
        return None

    if text.lower() == "null":
        return None

    return text


def _parse_str_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list):
        raise LLMParseError(f"{location} must be a list of strings.")

    result: list[str] = []

    for index, item in enumerate(value):
        item_location = f"{location}[{index}]"

        if item is None:
            raise LLMParseError(f"{item_location} must not be null.")

        if isinstance(item, (dict, list)):
            raise LLMParseError(
                f"{item_location} must be a string/number, got {type(item).__name__}."
            )

        text = str(item).strip()

        if text:
            result.append(text)

    return result


def _parse_bool(value: Any, location: str) -> bool:
    """
    Prefer real JSON booleans.

    But during development some models may output "true"/"false" as strings,
    so we accept those too.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered == "true":
            return True

        if lowered == "false":
            return False

    raise LLMParseError(f"{location} must be a boolean true/false.")


def _parse_severity(value: Any, location: str) -> Severity:
    text = _parse_str(value, location).lower()

    if text not in ALLOWED_SEVERITIES:
        raise LLMParseError(
            f"{location} must be one of: {', '.join(sorted(ALLOWED_SEVERITIES))}."
        )

    return cast(Severity, text)


__all__ = [
    "LLMParseError",
    "parse_llm_response",
]