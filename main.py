from __future__ import annotations

import asyncio
import traceback
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .action_validator import validate_llm_result
from .admin_review_manager import (
    AdminHandlingPlan,
    AdminReviewManager,
    build_admin_cancelled_text,
    build_admin_execution_finished_text,
    detect_decision_kind,
)
from .bundle_manager import BundleCreationResult, BundleManager
from .config_loader import LoadedConfig, load_config, normalize_id
from .data_models import MessageBundle, ValidationResult
from .executor import ModerationExecutor
from .llm_parser import LLMParseError, parse_llm_response
from .message_filter import (
    extract_group_id,
    extract_message_text,
    extract_sender_id,
    filter_event_to_message,
)
from .operation_handler import (
    OneBotV11OperationHandler,
    OperationHandlerError,
    create_operation_handler_from_event,
)
from .prompt_builder import build_moderation_prompt


class EnhancedQQGroupSupervisor(Star):
    """
    Enhanced QQ Group Supervisor.

    Target platform:
        AstrBot aiocqhttp / OneBot v11 / NapCat QQ.

    Safety default:
        Whether real moderation happens still depends on:
        - enable_actions
        - dry_run
        - admin_review
        - action validation
        - permission pre-check
    """

    CHECK_INTERVAL_SECONDS = 30

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)

        self.config = config
        self.loaded_config: LoadedConfig = load_config(config)

        self.bundle_manager = BundleManager(self.loaded_config)
        self.admin_review_manager = AdminReviewManager(review_timeout_seconds=1800)

        self.executor = ModerationExecutor(
            runtime=self.loaded_config.runtime,
            operation_handler=None,
        )

        # Cached operation handlers.
        # The handler is created from real aiocqhttp events, then reused for
        # delayed bundle flushing and admin execution replies.
        self._last_operation_handler: OneBotV11OperationHandler | None = None
        self._group_operation_handlers: dict[str, OneBotV11OperationHandler] = {}

        # If provider_id is empty in WebUI, we try to learn the provider ID from
        # the current group conversation and cache it.
        self._group_provider_ids: dict[str, str] = {}

        self._timer_task: asyncio.Task | None = None
        self._stopping = False

        logger.info("Enhanced QQ Group Supervisor initialized.")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        """
        当AstrBot完成加载时，启动一个后台计时器。这用于周期性检查需要监督的群聊中，已经收集到的群消息合集的状态。
        """
        self._start_background_timer()

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """
        对于需要监管的群聊，按照设置筛选收集群消息，并在满足条件时触发LLM分析。
        """
        operation_handler = self._update_operation_handler_from_event(event)

        group_id = extract_group_id(event)
        if group_id and operation_handler is not None:
            self._group_operation_handlers[group_id] = operation_handler

        filter_result = filter_event_to_message(event, self.loaded_config)

        if not filter_result.should_collect:
            if self.loaded_config.runtime.log_collected_messages:
                logger.debug(f"[QQ群监督员] 已忽略消息: {filter_result.reason}")
            return

        assert filter_result.message is not None

        if self.loaded_config.runtime.log_collected_messages:
            logger.info(
                "[QQ群监督员] 已收集消息: "
                f"group={filter_result.message.group_id}, "
                f"user={filter_result.message.user_id}, "
                f"message={filter_result.message.message_id}"
            )

        bundle_result = self.bundle_manager.add_message(filter_result.message)

        if bundle_result is None:
            return

        await self._handle_bundle_result(
            bundle_result=bundle_result,
            event=event,
            operation_handler=operation_handler,
        )

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """
        接受并寻找来自插件管理员的消息，无关消息会被跳过。
        """
        text = extract_message_text(event).strip()

        if not text:
            return

        # The message does not look anything like an execute / cancel
        # command, so ignore the message without doing anything.
        if detect_decision_kind(text) is None and "review_" not in text:
            return

        operation_handler = self._update_operation_handler_from_event(event)

        if operation_handler is None:
            logger.warning("[QQ群监督员] 未能处理回复: 行动处理程序不可用。")
            return

        sender_id = extract_sender_id(event)

        if not sender_id:
            logger.warning("[QQ群监督员] 未能处理回复: 缺失 sender_id")
            return
        
        # The message, although looks like an execute / cancel command,
        # comes from a someone who is not configured as a plugin
        # admin for any qq group. Do nothing to his message.
        if not self._is_known_plugin_admin(sender_id):
            return

        decision = self.admin_review_manager.parse_admin_decision(
            admin_qq=sender_id,
            text=text,
        )

        if decision.kind in {"unknown", "ambiguous"}:
            await operation_handler.send_private_message(
                target_user_id=sender_id,
                text=f"【QQ群监督员】{decision.message}",
            )
            return

        if decision.review is None or decision.review_id is None:
            await operation_handler.send_private_message(
                target_user_id=sender_id,
                text="【QQ群监督员】未找到对应的待审核请求。",
            )
            return

        if decision.kind == "cancel":
            review = self.admin_review_manager.cancel_review(decision.review_id)

            if review is None:
                await operation_handler.send_private_message(
                    target_user_id=sender_id,
                    text="【QQ群监督员】该待审核请求已经不存在，可能已被处理或过期。",
                )
                return

            await self._send_admin_text_as_forward_node(
                handler=operation_handler,
                admin_qq=sender_id,
                text=build_admin_cancelled_text(review),
                node_title="QQ群监督员操作已取消",
            )
            return

        if decision.kind == "execute":
            review = decision.review

            self.executor.update_operation_handler(operation_handler)
            self.executor.update_runtime(self.loaded_config.runtime)

            execution_result = await self.executor.execute_actions(
                bundle=review.bundle,
                actions=review.validation_result.valid_actions,
            )

            finished_review = self.admin_review_manager.finish_review(review.review_id)

            execution_text = build_admin_execution_finished_text(
                review=finished_review,
                bundle=review.bundle,
                validation_result=review.validation_result,
                execution_summary=execution_result.to_summary_text(review.bundle),
            )

            await self._send_admin_text_as_forward_node(
                handler=operation_handler,
                admin_qq=sender_id,
                text=execution_text,
                node_title="QQ群监督员操作结果",
            )
            return

    def _is_known_plugin_admin(self, user_id: str) -> bool:
        user_id = normalize_id(user_id)

        if not user_id:
            return False

        for admin in self.loaded_config.plugin_admins:
            if normalize_id(admin.admin_qq) == user_id:
                return True

        if self.admin_review_manager.list_pending_for_admin(user_id):
            return True

        return False

    async def _handle_bundle_result(
        self,
        *,
        bundle_result: BundleCreationResult,
        event: AstrMessageEvent | None,
        operation_handler: OneBotV11OperationHandler | None,
    ) -> None:
        bundle = bundle_result.bundle

        logger.info(
            "[QQ群监督员] handling bundle: "
            f"bundle_id={bundle.bundle_id}, "
            f"group={bundle.group_id}, "
            f"count={len(bundle.messages)}, "
            f"trigger={bundle_result.trigger_reason}"
        )

        try:
            await self._handle_bundle(
                bundle=bundle,
                event=event,
                operation_handler=operation_handler,
            )
        except Exception as exc:
            logger.error(
                "[QQ群监督员] bundle handling failed: "
                f"bundle_id={bundle.bundle_id}, error={exc}\n"
                f"{traceback.format_exc()}"
            )
            await self._notify_admin_error(
                bundle=bundle,
                operation_handler=operation_handler,
                title="QQ群监督员处理异常",
                error_text=f"{exc}\n\n{traceback.format_exc()}",
            )

    async def _handle_bundle(
        self,
        *,
        bundle: MessageBundle,
        event: AstrMessageEvent | None,
        operation_handler: OneBotV11OperationHandler | None,
    ) -> None:
        provider_id = await self._resolve_provider_id(
            event=event,
            group_id=bundle.group_id,
        )

        if not provider_id:
            raise RuntimeError(
                "No LLM provider_id available. Please configure llm.provider_id "
                "in WebUI, or let the plugin learn it from a live group event first."
            )

        prompt = build_moderation_prompt(bundle)

        llm_raw_response = await self._call_llm(
            provider_id=provider_id,
            prompt=prompt,
        )

        if self.loaded_config.runtime.log_llm_raw_response:
            logger.info(
                "[QQ群监督员] raw LLM response: "
                f"bundle_id={bundle.bundle_id}\n{llm_raw_response}"
            )

        try:
            llm_result = parse_llm_response(llm_raw_response)
        except LLMParseError as exc:
            await self._notify_admin_error(
                bundle=bundle,
                operation_handler=operation_handler,
                title="QQ群监督员 LLM 输出解析失败",
                error_text=(
                    f"{exc}\n\n"
                    "【LLM 原始输出】\n"
                    f"{llm_raw_response}"
                ),
            )
            return

        validation_result = validate_llm_result(
            bundle=bundle,
            llm_result=llm_result,
            runtime=self.loaded_config.runtime,
        )

        plan = self.admin_review_manager.prepare_admin_handling_plan(
            loaded_config=self.loaded_config,
            bundle=bundle,
            llm_result=llm_result,
            validation_result=validation_result,
        )

        await self._handle_admin_plan(
            plan=plan,
            bundle=bundle,
            validation_result=validation_result,
            operation_handler=operation_handler,
        )

    async def _send_admin_review_message(
        self,
        *,
        handler: OneBotV11OperationHandler,
        admin_qq: str,
        text: str,
        bundle: MessageBundle,
        validation_result: ValidationResult,
    ) -> None:
        from .admin_review_manager import build_admin_notification_nodes

        bot_user_id = ""
        try:
            bot_user_id = await handler.get_login_user_id()
        except Exception:
            bot_user_id = "10000"

        nodes = build_admin_notification_nodes(
            bot_user_id=bot_user_id,
            bot_nickname="QQ群监督员",
            admin_text=text,
            bundle=bundle,
            validation_result=validation_result,
            include_trigger_messages=True,
        )

        try:
            await handler.send_private_forward_message(
                target_user_id=admin_qq,
                nodes=nodes,
            )
        except Exception as exc:
            logger.warning(
                f"[QQ群监督员] 未能发送私聊转发消息，改用纯文本形式发送: {exc}"
            )
            await handler.send_private_message(
                target_user_id=admin_qq,
                text=text,
            )

    async def _send_admin_text_as_forward_node(
        self,
        *,
        handler: OneBotV11OperationHandler,
        admin_qq: str,
        text: str,
        node_title: str = "QQ群监督员",
    ) -> None:
        """
        Send one text block to plugin admin as a private forwarded-message node.

        If forward sending fails, fall back to normal private text message.
        """

        from .operation_handler import build_forward_node

        try:
            bot_user_id = await handler.get_login_user_id()
        except Exception:
            bot_user_id = "10000"

        nodes = [
            build_forward_node(
                user_id=bot_user_id,
                nickname=node_title,
                content=text,
            )
        ]

        try:
            await handler.send_private_forward_message(
                target_user_id=admin_qq,
                nodes=nodes,
            )
        except Exception as exc:
            logger.warning(
                "[EQGS] failed to send private forward node, "
                f"fallback to plain private message: {exc}"
            )

            await handler.send_private_message(
                target_user_id=admin_qq,
                text=text,
            )

    async def _handle_admin_plan(
        self,
        *,
        plan: AdminHandlingPlan,
        bundle: MessageBundle,
        validation_result: ValidationResult,
        operation_handler: OneBotV11OperationHandler | None,
    ) -> None:
        if plan.kind == "none":
            logger.info(f"[QQ群监督员] 管理员计划: none, 原因={plan.reason}")
            return

        handler = operation_handler or self._get_operation_handler_for_group(bundle.group_id)

        if plan.kind == "notify_only":
            if plan.admin_qq and handler is not None:
                await self._send_admin_review_message(
                    handler=handler,
                    admin_qq=plan.admin_qq,
                    text=plan.message_text,
                    bundle=bundle,
                    validation_result=validation_result,
                )
            else:
                logger.warning(
                    "[QQ群监督员] 仅通知模式已触发，但是管理员或行动处理程序缺失，执行失败。"
                )
            return

        if plan.kind == "review_required":
            if plan.admin_qq and handler is not None:
                await self._send_admin_review_message(
                    handler=handler,
                    admin_qq=plan.admin_qq,
                    text=plan.message_text,
                    bundle=bundle,
                    validation_result=validation_result,
                )
            else:
                logger.warning(
                    "[QQ群监督员] 请求执行模式已触发，但是管理员或行动处理程序缺失，执行失败。"
                )
            return

        if plan.kind == "auto_execute":
            actions = plan.actions_to_execute or []

            handler = handler or self._last_operation_handler

            if handler is None:
                logger.warning(
                    "[EQGS] auto_execute requested but no operation handler is available."
                )
                return

            # 1. Send the built plan before execution.
            if plan.admin_qq and plan.message_text:
                await self._send_admin_review_message(
                    handler=handler,
                    admin_qq=plan.admin_qq,
                    text=plan.message_text,
                    bundle=bundle,
                    validation_result=validation_result,
                )

            # 2. Execute validated actions.
            self.executor.update_runtime(self.loaded_config.runtime)
            self.executor.update_operation_handler(handler)

            execution_result = await self.executor.execute_actions(
                bundle=bundle,
                actions=actions,
            )

            # 3. Send execution result after execution.
            if plan.admin_qq:
                execution_text = build_admin_execution_finished_text(
                    review=None,
                    bundle=bundle,
                    validation_result=validation_result,
                    execution_summary=execution_result.to_summary_text(bundle),
                )

                await self._send_admin_text_as_forward_node(
                    handler=handler,
                    admin_qq=plan.admin_qq,
                    text=execution_text,
                    node_title="QQ群监督员操作结果",
                )

            return

    async def _call_llm(
        self,
        *,
        provider_id: str,
        prompt: str,
    ) -> str:
        """
        Call AstrBot's LLM interface.

        We only pass chat_provider_id and prompt in this version,
        because these are the documented stable parameters.
        """
        llm_resp = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
        )

        completion_text = getattr(llm_resp, "completion_text", None)

        if completion_text is None:
            return str(llm_resp)

        return str(completion_text)

    async def _resolve_provider_id(
        self,
        *,
        event: AstrMessageEvent | None,
        group_id: str,
    ) -> str:
        configured_provider_id = self.loaded_config.runtime.provider_id.strip()

        if configured_provider_id:
            return configured_provider_id

        group_id = normalize_id(group_id)

        if group_id in self._group_provider_ids:
            return self._group_provider_ids[group_id]

        if event is None:
            return ""

        umo = getattr(event, "unified_msg_origin", "")

        if not umo:
            return ""

        try:
            provider_id = await self.context.get_current_chat_provider_id(umo=umo)
        except TypeError:
            provider_id = await self.context.get_current_chat_provider_id(umo)

        provider_id = str(provider_id or "").strip()

        if provider_id and group_id:
            self._group_provider_ids[group_id] = provider_id

        return provider_id

    def _update_operation_handler_from_event(
        self,
        event: AstrMessageEvent,
    ) -> OneBotV11OperationHandler | None:
        try:
            handler = create_operation_handler_from_event(event)
        except OperationHandlerError as exc:
            logger.warning(f"[QQ群监督员] 未能创建行动处理程序: {exc}")
            return None
        except Exception as exc:
            logger.warning(
                "[QQ群监督员] 创建行动处理程序时，遇到了意料之外的错误: "
                f"{exc}\n{traceback.format_exc()}"
            )
            return None

        self._last_operation_handler = handler
        self.executor.update_operation_handler(handler)
        return handler

    def _get_operation_handler_for_group(
        self,
        group_id: str,
    ) -> OneBotV11OperationHandler | None:
        group_id = normalize_id(group_id)

        if group_id in self._group_operation_handlers:
            return self._group_operation_handlers[group_id]

        return self._last_operation_handler

    async def _notify_admin_error(
        self,
        *,
        bundle: MessageBundle,
        operation_handler: OneBotV11OperationHandler | None,
        title: str,
        error_text: str,
    ) -> None:
        admin = self.loaded_config.get_plugin_admin(bundle.group_id)

        if admin is None:
            logger.warning(
                "[QQ群监督员] 未能将错误通知给管理员: 未配置任何管理员。\n"
                f"{title}\n{error_text}"
            )
            return

        handler = operation_handler or self._get_operation_handler_for_group(bundle.group_id)

        if handler is None:
            logger.warning(
                "[QQ群监督员] 未能将错误通知给管理员: 没有可用的行动处理程序。\n"
                f"{title}\n{error_text}"
            )
            return

        text = (
            f"【{title}】\n"
            f"群号：{bundle.group_id}\n"
            f"消息包ID：{bundle.bundle_id}\n"
            f"消息数量：{len(bundle.messages)}\n\n"
            f"{error_text}"
        )

        await self._send_admin_text_as_forward_node(
            handler=handler,
            admin_qq=admin.admin_qq,
            text=text,
            node_title="QQ群监督员异常报告",
        )

    def _start_background_timer(self) -> None:
        if self._timer_task is not None and not self._timer_task.done():
            return

        self._stopping = False
        self._timer_task = asyncio.create_task(self._background_timer_loop())
        logger.info("[QQ群监督员] 后台计时器已开始运行。")

    async def _background_timer_loop(self) -> None:
        while not self._stopping:
            try:
                await asyncio.sleep(self.CHECK_INTERVAL_SECONDS)
                await self._check_expired_bundles()
                await self._cleanup_expired_reviews()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "[QQ群监督员] 后台计时器错误: "
                    f"{exc}\n{traceback.format_exc()}"
                )

    async def _check_expired_bundles(self) -> None:
        expired_results = self.bundle_manager.check_expired()

        for bundle_result in expired_results:
            handler = self._get_operation_handler_for_group(bundle_result.bundle.group_id)

            await self._handle_bundle_result(
                bundle_result=bundle_result,
                event=None,
                operation_handler=handler,
            )

    async def _cleanup_expired_reviews(self) -> None:
        expired_reviews = self.admin_review_manager.cleanup_expired_reviews()

        for review in expired_reviews:
            handler = self._get_operation_handler_for_group(review.group_id)

            if handler is None:
                logger.warning(
                    "[QQ群监督员] 等待管理员审核超时，但是没有可用的行动处理程序: "
                    f"{review.review_id}"
                )
                continue

            await self._send_admin_text_as_forward_node(
                handler=handler,
                admin_qq=review.admin_qq,
                text=(
                    "【QQ群监督员审核已过期】\n"
                    f"审核ID：{review.review_id}\n"
                    f"群号：{review.group_id}\n"
                    f"消息包ID：{review.bundle.bundle_id}\n"
                    "该审核请求已超时，不会执行建议操作。"
                ),
                node_title="QQ群监督员审核已过期",
            )

    async def terminate(self):
        """
        Called when plugin is unloaded/disabled.
        """
        self._stopping = True

        if self._timer_task is not None:
            self._timer_task.cancel()

            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

            self._timer_task = None

        self.bundle_manager.clear_all()
        self.admin_review_manager.clear_all()

        logger.info("Enhanced QQ Group Supervisor terminated.")