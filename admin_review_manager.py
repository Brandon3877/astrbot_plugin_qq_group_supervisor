# admin_review_manager.py

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from .config_loader import LoadedConfig, normalize_id
from .data_models import (
    LLMModerationResult,
    MessageBundle,
    PendingReview,
    PluginAdmin,
    ValidatedAction,
    ValidationResult,
)


AdminPlanKind = Literal[
    "none",
    "notify_only",
    "review_required",
    "auto_execute",
]

AdminDecisionKind = Literal[
    "execute",
    "cancel",
    "unknown",
    "ambiguous",
]


@dataclass
class AdminHandlingPlan:
    """
    Controls what the plugin should do after LLM result validation.

    Meanings explained:

    none:
        Do not notify admin and do not execute.

    notify_only:
        Send admin a diagnostic message only.
        No pending review is created.

    review_required:
        Send admin a review request.
        A PendingReview is created and stored.

    auto_execute:
        No admin approval is required.
        The caller should execute actions_to_execute.
        The caller may still notify admin after execution.
    """

    kind: AdminPlanKind
    reason: str
    admin_qq: str | None = None
    message_text: str = ""
    pending_review: PendingReview | None = None
    actions_to_execute: list[ValidatedAction] | None = None

    @property
    def should_notify_admin(self) -> bool:
        return self.kind in {"notify_only", "review_required"} and bool(self.admin_qq)

    @property
    def requires_review(self) -> bool:
        return self.kind == "review_required" and self.pending_review is not None

    @property
    def should_auto_execute(self) -> bool:
        return self.kind == "auto_execute" and bool(self.actions_to_execute)


@dataclass
class AdminDecision:
    """
    Parsed admin reply.

    This object only tells us what the admin intended.
    It does not execute actions.
    """

    kind: AdminDecisionKind
    message: str
    review_id: str | None = None
    review: PendingReview | None = None


class AdminReviewManager:
    """
    Stores pending admin reviews and parses admin decisions.

    This class is intentionally independent from AstrBot sender APIs.
    Later, main.py or admin_message_sender.py will use this manager,
    then send actual messages through AstrBot / aiocqhttp.
    """

    def __init__(self, review_timeout_seconds: int = 1800):
        self.review_timeout_seconds = review_timeout_seconds
        self.pending_reviews: dict[str, PendingReview] = {}

    def prepare_admin_handling_plan(
        self,
        *,
        loaded_config: LoadedConfig,
        bundle: MessageBundle,
        llm_result: LLMModerationResult,
        validation_result: ValidationResult,
        now: float | None = None,
    ) -> AdminHandlingPlan:
        """
        Decide what should happen after validation.

        This is the main function the later moderation pipeline should call.
        """

        now = self._now(now)
        runtime = loaded_config.runtime
        admin = loaded_config.get_plugin_admin(bundle.group_id)

        if llm_result.normal:
            return AdminHandlingPlan(
                kind="none",
                reason="llm_result_normal",
            )

        valid_actions = validation_result.valid_actions

        if not valid_actions:
            if admin is not None and runtime.notify_admin_when_no_action:
                return AdminHandlingPlan(
                    kind="notify_only",
                    reason="abnormal_but_no_valid_actions",
                    admin_qq=admin.admin_qq,
                    message_text=build_admin_notification_text(
                        bundle=bundle,
                        llm_result=llm_result,
                        validation_result=validation_result,
                        admin=admin,
                        title="群聊监督诊断结果",
                        action_status="LLM 判断存在问题，但没有可执行的有效操作。",
                    ),
                )

            return AdminHandlingPlan(
                kind="none",
                reason="abnormal_but_no_valid_actions_and_no_notification",
            )

        if not runtime.enable_actions:
            if admin is not None:
                return AdminHandlingPlan(
                    kind="notify_only",
                    reason="actions_disabled_notify_only",
                    admin_qq=admin.admin_qq,
                    message_text=build_admin_notification_text(
                        bundle=bundle,
                        llm_result=llm_result,
                        validation_result=validation_result,
                        admin=admin,
                        title="群聊监督诊断结果",
                        action_status=(
                            "已生成建议操作，但“允许机器人执行实际操作”处于关闭状态，"
                            "因此不会询问执行，也不会实际执行。"
                        ),
                    ),
                )

            return AdminHandlingPlan(
                kind="none",
                reason="actions_disabled_and_no_admin",
            )

        if loaded_config.should_review_before_execute(bundle.group_id):
            if admin is None:
                return AdminHandlingPlan(
                    kind="none",
                    reason="admin_review_enabled_but_no_admin_configured",
                )

            pending_review = self.create_pending_review(
                group_id=bundle.group_id,
                admin_qq=admin.admin_qq,
                bundle=bundle,
                llm_result=llm_result,
                validation_result=validation_result,
                now=now,
            )

            return AdminHandlingPlan(
                kind="review_required",
                reason="admin_review_required",
                admin_qq=admin.admin_qq,
                pending_review=pending_review,
                message_text=build_admin_notification_text(
                    bundle=bundle,
                    llm_result=llm_result,
                    validation_result=validation_result,
                    admin=admin,
                    title="群聊分析完成，等待确认执行操作",
                    action_status=(
                        "是否执行建议的操作？\n"
                        f"回复：执行 {pending_review.review_id}\n"
                        f"或：取消 {pending_review.review_id}\n"
                        "如果当前只有一个待审核请求，可以直接回复：执行 / 取消"
                    ),
                    review_id=pending_review.review_id,
                ),
            )

        return AdminHandlingPlan(
            kind="auto_execute",
            reason="auto_execute_without_admin_review",
            admin_qq=admin.admin_qq if admin is not None else None,
            actions_to_execute=list(valid_actions),
            message_text=build_admin_notification_text(
                bundle=bundle,
                llm_result=llm_result,
                validation_result=validation_result,
                admin=admin,
                title="群聊分析完成，将自动执行操作",
                action_status="管理员审核关闭，以下有效操作已被计划自动执行。",
            )
            if admin is not None
            else "",
        )
    
    def update_timeout_seconds(self, review_timeout_seconds: int) -> None:
        self.review_timeout_seconds = max(1, int(review_timeout_seconds))

    def create_pending_review(
        self,
        *,
        group_id: str,
        admin_qq: str,
        bundle: MessageBundle,
        llm_result: LLMModerationResult,
        validation_result: ValidationResult,
        now: float | None = None,
    ) -> PendingReview:
        now = self._now(now)
        review_id = self._make_review_id(group_id, now)

        review = PendingReview(
            review_id=review_id,
            group_id=normalize_id(group_id),
            admin_qq=normalize_id(admin_qq),
            bundle=bundle,
            llm_result=llm_result,
            validation_result=validation_result,
            created_at=now,
        )

        self.pending_reviews[review_id] = review
        return review

    def get_review(self, review_id: str) -> PendingReview | None:
        return self.pending_reviews.get(normalize_id(review_id))

    def finish_review(self, review_id: str) -> PendingReview | None:
        """
        Remove and return a pending review.

        Call this after actions are executed, or after cancellation is accepted.
        """
        return self.pending_reviews.pop(normalize_id(review_id), None)

    def cancel_review(self, review_id: str) -> PendingReview | None:
        return self.finish_review(review_id)

    def list_pending_for_admin(self, admin_qq: str) -> list[PendingReview]:
        admin_qq = normalize_id(admin_qq)

        return [
            review
            for review in self.pending_reviews.values()
            if review.admin_qq == admin_qq
        ]

    def parse_admin_decision(
        self,
        *,
        admin_qq: str,
        text: str,
    ) -> AdminDecision:
        """
        Parse private admin reply.

        Supported examples:
        - 执行
        - 取消
        - 执行 review_123_...
        - 取消 review_123_...
        - cancel
        - stop
        """

        admin_qq = normalize_id(admin_qq)
        clean_text = text.strip()

        if not clean_text:
            return AdminDecision(
                kind="unknown",
                message="空回复，无法判断是执行还是取消。",
            )

        decision_kind = detect_decision_kind(clean_text)

        if decision_kind is None:
            return AdminDecision(
                kind="unknown",
                message="未识别到执行或取消指令。",
            )

        explicit_review_id = extract_review_id(clean_text)

        if explicit_review_id:
            review = self.get_review(explicit_review_id)

            if review is None:
                return AdminDecision(
                    kind="unknown",
                    review_id=explicit_review_id,
                    message=f"未找到待审核请求：{explicit_review_id}",
                )

            if review.admin_qq != admin_qq:
                return AdminDecision(
                    kind="unknown",
                    review_id=explicit_review_id,
                    message="该待审核请求不属于当前管理员。",
                )

            return AdminDecision(
                kind=decision_kind,
                review_id=explicit_review_id,
                review=review,
                message="已识别管理员审核指令。",
            )

        pending_for_admin = self.list_pending_for_admin(admin_qq)

        if len(pending_for_admin) == 0:
            return AdminDecision(
                kind="unknown",
                message="当前没有属于该管理员的待审核请求。",
            )

        if len(pending_for_admin) > 1:
            review_ids = ", ".join(review.review_id for review in pending_for_admin)
            return AdminDecision(
                kind="ambiguous",
                message=(
                    "当前有多个待审核请求，请在回复中带上 review_id。\n"
                    f"待审核请求：{review_ids}"
                ),
            )

        review = pending_for_admin[0]

        return AdminDecision(
            kind=decision_kind,
            review_id=review.review_id,
            review=review,
            message="已识别管理员审核指令。",
        )

    def cleanup_expired_reviews(
        self,
        *,
        now: float | None = None,
    ) -> list[PendingReview]:
        """
        Remove expired pending reviews.

        Later we can call this periodically to prevent memory accumulation.
        """
        now = self._now(now)
        expired: list[PendingReview] = []

        for review_id, review in list(self.pending_reviews.items()):
            age = now - review.created_at

            if age >= self.review_timeout_seconds:
                expired.append(review)
                del self.pending_reviews[review_id]

        return expired

    def clear_all(self) -> None:
        self.pending_reviews.clear()

    @staticmethod
    def _make_review_id(group_id: str, now: float) -> str:
        group_id = normalize_id(group_id) or "unknown_group"
        created_ms = int(now * 1000)
        short_uuid = uuid.uuid4().hex[:8]
        return f"review_{group_id}_{created_ms}_{short_uuid}"

    @staticmethod
    def _now(now: float | None) -> float:
        return time.time() if now is None else now


def build_admin_notification_text(
    *,
    bundle: MessageBundle,
    llm_result: LLMModerationResult,
    validation_result: ValidationResult,
    admin: PluginAdmin | None,
    title: str,
    action_status: str,
    review_id: str | None = None,
) -> str:
    """
    Build plain-text admin notification.

    Later, the QQ adapter, will let this text become one node in
    a forwarded chat-record message. The original offending messages
    can be added as extra forward nodes.
    """

    lines: list[str] = []

    lines.append(f"【{title}】")

    if review_id:
        lines.append(f"审核ID：{review_id}")

    lines.append(f"群号：{bundle.group_id}")
    lines.append(f"消息包ID：{bundle.bundle_id}")
    lines.append(f"消息数量：{len(bundle.messages)}")

    if admin is not None:
        lines.append(f"接收管理员：{admin.admin_qq}")

    lines.append("")
    lines.append("【整体判断】")
    lines.append(llm_result.message_bundle_assessment or "无")

    problem_message_diagnoses = [
        diagnosis
        for diagnosis in llm_result.message_diagnoses
        if diagnosis.severity != "none" or diagnosis.rule_violated.strip()
    ]

    if problem_message_diagnoses:
        lines.append("")
        lines.append("【问题消息】")
        for diagnosis in problem_message_diagnoses:
            message = bundle.find_message(diagnosis.message_id)
            nickname = message.nickname if message is not None else "未知昵称"

            lines.append(
                f"- 消息 {diagnosis.message_id} | "
                f"{nickname} / {diagnosis.user_id} | "
                f"严重程度：{diagnosis.severity} | "
                f"违反：{diagnosis.rule_violated} | "
                f"{diagnosis.summary}"
            )

    problem_user_diagnoses = [
        diagnosis
        for diagnosis in llm_result.user_diagnoses
        if diagnosis.severity != "none"
    ]

    if problem_user_diagnoses:
        lines.append("")
        lines.append("【问题用户】")
        for diagnosis in problem_user_diagnoses:
            nickname = find_nickname_by_user_id(bundle, diagnosis.user_id)
            lines.append(
                f"- {nickname} / {diagnosis.user_id} | "
                f"严重程度：{diagnosis.severity} | "
                f"{diagnosis.summary}"
            )

    lines.append("")
    lines.append("【已验证的建议操作】")

    if validation_result.valid_actions:
        for index, action in enumerate(validation_result.valid_actions, start=1):
            lines.append(f"{index}. {action.to_admin_text(bundle)}")
    else:
        lines.append("无")

    if validation_result.rejected_actions:
        lines.append("")
        lines.append("【被拒绝的 LLM 输出】")
        lines.append("以下内容不会被执行，仅用于调试：")

        for item in validation_result.rejected_actions[:10]:
            lines.append(f"- {item}")

        remaining = len(validation_result.rejected_actions) - 10
        if remaining > 0:
            lines.append(f"- 还有 {remaining} 条未展示")

    lines.append("")
    lines.append("【处理状态】")
    lines.append(action_status)

    return "\n".join(lines)


def build_admin_notification_nodes(
    *,
    bot_user_id: str,
    bot_nickname: str,
    admin_text: str,
    bundle: MessageBundle,
    validation_result: ValidationResult,
    include_trigger_messages: bool = True,
) -> list[dict[str, Any]]:
    from .operation_handler import build_forward_node

    nodes: list[dict[str, Any]] = []

    nodes.append(
        build_forward_node(
            user_id=bot_user_id,
            nickname=bot_nickname,
            content=admin_text,
        )
    )

    if not include_trigger_messages:
        return nodes

    message_ids: list[str] = []
    seen: set[str] = set()

    for action in validation_result.valid_actions:
        if action.target_message_id and action.target_message_id not in seen:
            seen.add(action.target_message_id)
            message_ids.append(action.target_message_id)

        for message_id in action.based_on_message_ids:
            if message_id not in seen:
                seen.add(message_id)
                message_ids.append(message_id)

    for message_id in message_ids:
        message = bundle.find_message(message_id)

        if message is None:
            continue

        nodes.append(
            build_forward_node(
                user_id=message.user_id,
                nickname=message.nickname,
                content=(
                    f"消息ID: {message.message_id}\n"
                    f"用户ID: {message.user_id}\n"
                    f"群身份: {message.role}\n"
                    f"群等级: {message.group_level}\n"
                    f"消息内容:\n{message.text}"
                ),
            )
        )

    return nodes


def build_admin_execution_finished_text(
    *,
    review: PendingReview | None,
    bundle: MessageBundle,
    validation_result: ValidationResult,
    execution_summary: str,
) -> str:
    """
    Build a message after executor.py finishes.

    The actual execution result will be produced by executor.py later.
    """

    lines: list[str] = []

    lines.append("【群聊监督操作结果】")

    if review is not None:
        lines.append(f"审核ID：{review.review_id}")

    lines.append(f"群号：{bundle.group_id}")
    lines.append(f"消息包ID：{bundle.bundle_id}")
    lines.append("")
    lines.append("【已处理操作】")

    if validation_result.valid_actions:
        for index, action in enumerate(validation_result.valid_actions, start=1):
            lines.append(f"{index}. {action.to_admin_text(bundle)}")
    else:
        lines.append("无")

    lines.append("")
    lines.append("【执行结果】")
    lines.append(execution_summary or "无执行结果。")

    return "\n".join(lines)


def build_admin_cancelled_text(review: PendingReview) -> str:
    return (
        "【群聊监督审核已取消】\n"
        f"审核ID：{review.review_id}\n"
        f"群号：{review.group_id}\n"
        f"消息包ID：{review.bundle.bundle_id}\n"
        "建议操作已取消，不会执行。"
    )


def detect_decision_kind(text: str) -> Literal["execute", "cancel"] | None:
    """
    Detect admin decision from reply text.
    """

    normalized = text.strip().lower()

    execute_words = {
        "执行",
        #"确认执行",
        #"同意",
        #"同意执行",
        #"确认",
        #"通过",
        #"yes",
        #"y",
        #"ok",
        #"execute",
        #"run",
    }

    cancel_words = {
        "取消",
        "不执行",
        #"拒绝",
        #"否",
        #"no",
        #"n",
        "cancel",
        #"stop",
    }

    first_token = normalized.split(maxsplit=1)[0]

    if normalized in execute_words or first_token in execute_words:
        return "execute"

    if normalized in cancel_words or first_token in cancel_words:
        return "cancel"

    return None


def extract_review_id(text: str) -> str | None:
    match = re.search(r"\breview_[A-Za-z0-9_]+\b", text)

    if not match:
        return None

    return match.group(0)


def find_nickname_by_user_id(bundle: MessageBundle, user_id: str) -> str:
    user_id = normalize_id(user_id)

    for message in bundle.messages:
        if message.user_id == user_id:
            return message.nickname

    return "未知昵称"


__all__ = [
    "AdminHandlingPlan",
    "AdminDecision",
    "AdminReviewManager",
    "build_admin_notification_text",
    "build_admin_execution_finished_text",
    "build_admin_cancelled_text",
    "detect_decision_kind",
    "extract_review_id",
]