# executor.py

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .data_models import (
    MessageBundle,
    RuntimeConfig,
    ValidatedAction,
)

from .permission_checker import ModerationPermissionChecker


ExecutionStatus = Literal[
    "success",
    "failed",
    "skipped",
    "dry_run",
]


class ModerationOperationHandler(Protocol):
    """
    Adapter boundary for real QQ operations.

    executor.py does not directly depend on AstrBot or OneBot APIs.
    operation_handler.py is an AstrBot/OneBot implementation of this handler.

    With this design, users and future developers can test executor.py
    safely without touching real groups.
    """

    async def send_group_warning(
        self,
        *,
        group_id: str,
        target_user_id: str,
        warning_text: str,
        quote_message_id: str | None = None,
    ) -> str:
        ...

    async def recall_message(
        self,
        *,
        group_id: str,
        message_id: str,
    ) -> str:
        ...

    async def mute_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        duration_seconds: int,
    ) -> str:
        ...

    async def kick_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        blacklist: bool,
    ) -> str:
        ...


@dataclass
class ActionExecutionResult:
    """
    Result of executing one ValidatedAction.
    """

    status: ExecutionStatus
    action: ValidatedAction
    message: str
    exception_text: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"success", "dry_run", "skipped"}


@dataclass
class ExecutionBatchResult:
    """
    Result of executing a group of ValidatedAction objects.
    """

    group_id: str
    bundle_id: str
    results: list[ActionExecutionResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for item in self.results if item.status == "success")

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.results if item.status == "failed")

    @property
    def skipped_count(self) -> int:
        return sum(1 for item in self.results if item.status == "skipped")

    @property
    def dry_run_count(self) -> int:
        return sum(1 for item in self.results if item.status == "dry_run")

    @property
    def total_count(self) -> int:
        return len(self.results)

    def to_summary_text(self, bundle: MessageBundle) -> str:
        lines: list[str] = []

        lines.append(
            f"总数 {self.total_count}，"
            f"成功 {self.success_count}，"
            f"失败 {self.failed_count}，"
            f"跳过 {self.skipped_count}，"
            f"演习 {self.dry_run_count}。"
        )

        if not self.results:
            lines.append("没有需要执行的操作。")
            return "\n".join(lines)

        lines.append("")

        for index, result in enumerate(self.results, start=1):
            action_text = result.action.to_admin_text(bundle)

            lines.append(f"{index}. 状态：{result.status}")
            lines.append(action_text)
            lines.append(f"结果：{result.message}")

            if result.exception_text:
                lines.append(f"异常：{result.exception_text}")

            lines.append("")

        return "\n".join(lines).rstrip()


class ModerationExecutor:
    """
    Executes validated moderation actions.

    Safety rules:
    1. If runtime.enable_actions is False, skip everything.
    2. If runtime.dry_run is True, do not call real QQ operations.
    3. If no operation_handler is provided, fail safely.
    4. Catch exceptions per action, so one failed operation does not stop all.
    """

    def __init__(
        self,
        *,
        runtime: RuntimeConfig,
        operation_handler: ModerationOperationHandler | None = None,
        permission_checker: ModerationPermissionChecker | None = None,
    ):
        self.runtime = runtime
        self.operation_handler = operation_handler

        if permission_checker is not None:
            self.permission_checker = permission_checker
        elif operation_handler is not None:
            self.permission_checker = ModerationPermissionChecker(operation_handler)
        else:
            self.permission_checker = None

    def update_runtime(self, runtime: RuntimeConfig) -> None:
        self.runtime = runtime

    def update_operation_handler(
        self,
        operation_handler: ModerationOperationHandler | None,
    ) -> None:
        self.operation_handler = operation_handler

        if operation_handler is not None:
            self.permission_checker = ModerationPermissionChecker(operation_handler)
        else:
            self.permission_checker = None

    async def _check_permission_before_execute(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult | None:
        """
        Check group role permission before destructive actions.

        If permission is clearly insufficient, skip.
        If permission is unknown, still attempt the real API call.
        """

        if self.permission_checker is None:
            return None

        permission = await self.permission_checker.check_action(
            bundle=bundle,
            action=action,
        )

        if permission.should_attempt:
            return None

        return ActionExecutionResult(
            status="skipped",
            action=action,
            message=(
                "权限预检查未通过，已跳过实际执行。\n"
                f"操作类型：{permission.action_type}\n"
                f"群号：{permission.group_id}\n"
                f"机器人QQ：{permission.self_id or '未知'}\n"
                f"机器人身份：{permission.bot_role}\n"
                f"目标QQ：{permission.target_user_id or '未知'}\n"
                f"目标身份：{permission.target_role}\n"
                f"原因：{permission.reason}"
            ),
        )

    async def execute_actions(
        self,
        *,
        bundle: MessageBundle,
        actions: list[ValidatedAction],
    ) -> ExecutionBatchResult:
        batch_result = ExecutionBatchResult(
            group_id=bundle.group_id,
            bundle_id=bundle.bundle_id,
        )

        if not actions:
            return batch_result

        if not self.runtime.enable_actions:
            for action in actions:
                batch_result.results.append(
                    ActionExecutionResult(
                        status="skipped",
                        action=action,
                        message="enable_actions=False，已跳过实际执行。",
                    )
                )
            return batch_result

        if self.runtime.dry_run:
            for action in actions:
                batch_result.results.append(
                    self._make_dry_run_result(
                        bundle=bundle,
                        action=action,
                    )
                )
            return batch_result

        if self.operation_handler is None:
            for action in actions:
                batch_result.results.append(
                    ActionExecutionResult(
                        status="failed",
                        action=action,
                        message="未配置 operation_handler，无法执行真实QQ操作。",
                    )
                )
            return batch_result

        for action in actions:
            result = await self._execute_one_action(
                bundle=bundle,
                action=action,
            )
            batch_result.results.append(result)

        return batch_result

    async def _execute_one_action(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        permission_skip = await self._check_permission_before_execute(
            bundle=bundle,
            action=action,
        )

        if permission_skip is not None:
            return permission_skip

        punishment_type = action.punishment.type

        try:
            if punishment_type == "warn":
                return await self._execute_warn(bundle=bundle, action=action)

            if punishment_type == "recall":
                return await self._execute_recall(bundle=bundle, action=action)

            if punishment_type == "mute":
                return await self._execute_mute(bundle=bundle, action=action)

            if punishment_type == "kick":
                return await self._execute_kick(bundle=bundle, action=action)

            return ActionExecutionResult(
                status="failed",
                action=action,
                message=f"未知处罚类型：{punishment_type}",
            )

        except Exception as exc:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message=f"执行时发生异常：{exc}",
                exception_text=traceback.format_exc(),
            )

    async def _execute_warn(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        if not action.target_user_id:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message="警告操作缺少 target_user_id。",
            )

        warning_text = action.warning_text or action.punishment.warning_text or ""

        if not warning_text.strip():
            return ActionExecutionResult(
                status="failed",
                action=action,
                message="警告文本为空。",
            )

        assert self.operation_handler is not None

        quote_message_id = self._get_warning_quote_message_id(action)

        result_message = await self.operation_handler.send_group_warning(
            group_id=bundle.group_id,
            target_user_id=action.target_user_id,
            warning_text=warning_text,
            quote_message_id=quote_message_id
        )

        return ActionExecutionResult(
            status="success",
            action=action,
            message=result_message or "已发送警告。",
        )
    
    def _get_warning_quote_message_id(
        self,
        action: ValidatedAction,
    ) -> str | None:
        """
        Decide which message should be quoted for a warn action.

        Priority:
        1. Do not quote if the punishment config disables quote_trigger_message.
        2. Prefer action.target_message_id.
        3. Otherwise use the first based_on_message_id.
        """

        if action.punishment.type != "warn":
            return None

        if not action.punishment.quote_trigger_message:
            return None

        if action.target_message_id:
            return action.target_message_id

        if action.based_on_message_ids:
            return action.based_on_message_ids[0]

        return None

    async def _execute_recall(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        if not action.target_message_id:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message="撤回操作缺少 target_message_id。",
            )

        assert self.operation_handler is not None

        result_message = await self.operation_handler.recall_message(
            group_id=bundle.group_id,
            message_id=action.target_message_id,
        )

        return ActionExecutionResult(
            status="success",
            action=action,
            message=result_message or "已撤回消息。",
        )

    async def _execute_mute(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        if not action.target_user_id:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message="禁言操作缺少 target_user_id。",
            )

        duration_minutes = action.punishment.duration_minutes

        if duration_minutes is None or duration_minutes <= 0:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message=f"禁言时长无效：{duration_minutes}",
            )

        duration_seconds = duration_minutes * 60

        assert self.operation_handler is not None

        result_message = await self.operation_handler.mute_user(
            group_id=bundle.group_id,
            target_user_id=action.target_user_id,
            duration_seconds=duration_seconds,
        )

        return ActionExecutionResult(
            status="success",
            action=action,
            message=result_message or f"已禁言 {duration_minutes} 分钟。",
        )

    async def _execute_kick(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        if not action.target_user_id:
            return ActionExecutionResult(
                status="failed",
                action=action,
                message="踢走操作缺少 target_user_id。",
            )

        assert self.operation_handler is not None

        result_message = await self.operation_handler.kick_user(
            group_id=bundle.group_id,
            target_user_id=action.target_user_id,
            blacklist=action.punishment.blacklist,
        )

        return ActionExecutionResult(
            status="success",
            action=action,
            message=result_message or "已踢出用户。",
        )

    def _make_dry_run_result(
        self,
        *,
        bundle: MessageBundle,
        action: ValidatedAction,
    ) -> ActionExecutionResult:
        punishment_type = action.punishment.type

        if punishment_type == "warn":
            target = action.target_user_id or "未知用户"
            warning_text = action.warning_text or action.punishment.warning_text or ""
            quote_message_id = self._get_warning_quote_message_id(action)
            if quote_message_id:
                message = (
                    f"演习模式：将引用消息 {quote_message_id}，"
                    f"并向 {target} 发送警告：{warning_text}"
                )
            message = f"演习模式：将向 {target} 发送警告：{warning_text}"

        elif punishment_type == "recall":
            message = f"演习模式：将撤回消息 {action.target_message_id}"

        elif punishment_type == "mute":
            message = (
                f"演习模式：将禁言用户 {action.target_user_id} "
                f"{action.punishment.duration_minutes} 分钟"
            )

        elif punishment_type == "kick":
            message = (
                f"演习模式：将踢出用户 {action.target_user_id}，"
                f"拉黑={action.punishment.blacklist}"
            )

        else:
            message = f"演习模式：未知操作 {punishment_type}"

        return ActionExecutionResult(
            status="dry_run",
            action=action,
            message=message,
        )


class DryRunOperationHandler:
    """
    Optional fake handler.

    Usually we do not need this, because runtime.dry_run=True already prevents
    real execution. But this class is useful for unit tests.
    """

    async def send_group_warning(
        self,
        *,
        group_id: str,
        target_user_id: str,
        warning_text: str,
        quote_message_id: str | None = None,
    ) -> str:
        return (
            f"[DryRunHandler] send_group_warning "
            f"group={group_id}, user={target_user_id}, "
            f"quote={quote_message_id}, text={warning_text}"
        )

    async def recall_message(
        self,
        *,
        group_id: str,
        message_id: str,
    ) -> str:
        return f"[DryRunHandler] recall_message group={group_id}, message={message_id}"

    async def mute_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        duration_seconds: int,
    ) -> str:
        return f"[DryRunHandler] mute_user group={group_id}, user={target_user_id}, seconds={duration_seconds}"

    async def kick_user(
        self,
        *,
        group_id: str,
        target_user_id: str,
        blacklist: bool,
    ) -> str:
        return f"[DryRunHandler] kick_user group={group_id}, user={target_user_id}, blacklist={blacklist}"


__all__ = [
    "ExecutionStatus",
    "ModerationOperationHandler",
    "ActionExecutionResult",
    "ExecutionBatchResult",
    "ModerationExecutor",
    "DryRunOperationHandler",
]