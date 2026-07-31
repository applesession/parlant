# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any, cast

from pytest import raises

from parlant.core.engines.alpha.engine_context import EngineContext
from parlant.core.engines.alpha.hooks import EngineHookResult, EngineHooks
from parlant.core.tools import ToolContext, ToolError


async def test_that_empty_message_batch_is_passed_to_hook() -> None:
    hooks = EngineHooks()
    received: list[object] = []

    async def handler(
        context: EngineContext, payload: object, exc: Exception | None
    ) -> EngineHookResult:
        received.append(payload)
        return EngineHookResult.CALL_NEXT

    hooks.on_message_batch_generated.append(handler)

    assert await hooks.call_on_message_batch_generated(cast(Any, None), [])
    assert received == [[]]


async def test_that_premoderation_blocks_direct_tool_message_emit() -> None:
    emitted: list[str] = []

    async def emit_message(message: str) -> None:
        emitted.append(message)

    context = ToolContext(
        agent_id="agent",
        session_id="session",
        customer_id="customer",
        emit_message=emit_message,
        premoderation_required=True,
    )

    with raises(ToolError):
        await context.emit_message("must not escape")

    assert emitted == []
