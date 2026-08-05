# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from lagom import Container
import pytest
from pytest import raises

from parlant.core.agents import Agent
from parlant.core.customers import CustomerStore
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha import canned_response_generator as canned_module
from parlant.core.engines.alpha.canned_response_generator import (
    CannedResponseGenerator,
    _CannedResponseSelectionResult,
)
from parlant.core.engines.alpha.engine_context import (
    EngineContext,
    Interaction,
    ResponseState,
    is_premoderation_required,
)
from parlant.core.engines.alpha.hooks import EngineHookResult, EngineHooks
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.types import Context
from parlant.core.loggers import Logger
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.sessions import EventKind, EventSource, MessageEventData, SessionStore
from parlant.core.tools import ToolContext, ToolError
from parlant.core.tracer import Tracer

from tests.core.common.utils import create_event_message


def _generation_info() -> GenerationInfo:
    return GenerationInfo(
        schema_name="test",
        model="test",
        duration=0.0,
        usage=UsageInfo(input_tokens=0, output_tokens=0),
    )


async def _engine_context(
    container: Container,
    agent: Agent,
    metadata: dict[str, Any] | None = None,
) -> tuple[EngineContext, EventBuffer]:
    customer = await container[CustomerStore].read_customer(CustomerStore.GUEST_ID)
    session = await container[SessionStore].create_session(customer.id, agent.id)
    interaction = Interaction(
        events=[
            create_event_message(
                offset=0,
                source=EventSource.CUSTOMER,
                message="Hello",
                customer=customer,
                metadata=metadata or {},
            )
        ]
    )
    event_buffer = EventBuffer(agent)
    context = EngineContext(
        info=Context(session_id=session.id, agent_id=agent.id),
        logger=container[Logger],
        tracer=container[Tracer],
        agent=agent,
        customer=customer,
        session=session,
        session_event_emitter=event_buffer,
        response_event_emitter=EventBuffer(agent),
        interaction=interaction,
        state=ResponseState(
            context_variables=[],
            glossary_terms=set(),
            capabilities=[],
            iterations=[],
            ordinary_guideline_matches=[],
            tool_enabled_guideline_matches={},
            journeys=[],
            journey_paths={},
            tool_events=[],
            tool_insights=ToolInsights(),
            prepared_to_respond=False,
            message_events=[],
        ),
    )
    return context, event_buffer


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


@pytest.mark.parametrize(
    ("hook_result", "expected_message_count"),
    [
        (EngineHookResult.BAIL, 0),
        (EngineHookResult.CALL_NEXT, 1),
    ],
)
async def test_that_premoderation_fluid_batch_is_normalized_before_emit(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    hook_result: EngineHookResult,
    expected_message_count: int,
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[MessageGenerator]
    hooks = container[EngineHooks]
    received: list[list[MessageEventData]] = []

    async def handler(
        context: EngineContext,
        payload: list[MessageEventData],
        exc: Exception | None,
    ) -> EngineHookResult:
        assert not [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]
        received.append(payload)
        return hook_result

    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "shots", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generator,
        "_generate_response_message",
        AsyncMock(return_value=(_generation_info(), "Hello from FLUID")),
    )

    result = await generator.generate_response(context)
    message_events = [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]

    assert received == [
        [
            {
                "message": "Hello from FLUID",
                "participant": {"id": agent.id, "display_name": agent.name},
            }
        ]
    ]
    assert len(message_events) == expected_message_count
    assert len(result[0].events) == expected_message_count
    if message_events:
        assert message_events[0].data == received[0][0]


async def test_that_premoderation_fluid_empty_batch_is_reported_once(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[MessageGenerator]
    hooks = container[EngineHooks]
    received: list[list[MessageEventData]] = []

    async def handler(
        context: EngineContext,
        payload: list[MessageEventData],
        exc: Exception | None,
    ) -> EngineHookResult:
        received.append(payload)
        return EngineHookResult.BAIL

    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "shots", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generator,
        "_generate_response_message",
        AsyncMock(return_value=(_generation_info(), None)),
    )

    result = await generator.generate_response(context)

    assert received == [[]]
    assert result[0].events == []
    assert not [event for event in event_buffer.events if event.kind == EventKind.MESSAGE]


@pytest.mark.parametrize(
    ("hook_result", "expected_messages"),
    [
        (EngineHookResult.BAIL, []),
        (EngineHookResult.CALL_NEXT, ["one", "two", "three"]),
    ],
)
async def test_that_premoderation_canned_batch_contains_split_main_and_follow_up(
    container: Container,
    agent: Agent,
    monkeypatch: pytest.MonkeyPatch,
    hook_result: EngineHookResult,
    expected_messages: list[str],
) -> None:
    context, event_buffer = await _engine_context(container, agent)
    generator = container[CannedResponseGenerator]
    hooks = container[EngineHooks]
    received: list[list[MessageEventData]] = []
    main = _CannedResponseSelectionResult(
        message="one\n\ntwo",
        draft=None,
        rendered_canned_responses=[],
        chosen_canned_responses=[],
    )
    follow_up = _CannedResponseSelectionResult(
        message="three",
        draft=None,
        rendered_canned_responses=[],
        chosen_canned_responses=[],
    )
    policy = Mock()
    policy.is_message_splitting_required = AsyncMock(return_value=True)
    policy.get_follow_up_delay = AsyncMock(return_value=0.0)

    async def handler(
        context: EngineContext,
        payload: list[MessageEventData],
        exc: Exception | None,
    ) -> EngineHookResult:
        assert event_buffer.events == []
        received.append(payload)
        return hook_result

    hooks.on_message_batch_generated.append(handler)
    monkeypatch.setattr(generator, "_get_relevant_canned_responses", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generator,
        "_generate_response",
        AsyncMock(return_value=({}, main)),
    )
    monkeypatch.setattr(
        generator,
        "generate_follow_up_response",
        AsyncMock(return_value=({}, follow_up)),
    )
    monkeypatch.setattr(
        generator._perceived_performance_policy_provider,
        "get_policy",
        Mock(return_value=policy),
    )
    monkeypatch.setattr(canned_module.asyncio, "sleep", AsyncMock())

    result = await generator.generate_response(context)
    emitted_messages = [
        cast(dict[str, Any], event.data)["message"]
        for event in event_buffer.events
        if event.kind == EventKind.MESSAGE
    ]

    assert [[message["message"] for message in batch] for batch in received] == [
        ["one", "two", "three"]
    ]
    assert emitted_messages == expected_messages
    assert len(result[0].events) == len(expected_messages)
    if hook_result == EngineHookResult.BAIL:
        assert event_buffer.events == []


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, False),
        (
            {
                "_impact_private": {
                    "impact_cycle_v1": {"premoderation_required": False}
                }
            },
            False,
        ),
        (
            {
                "_impact_private": {
                    "impact_cycle_v1": {"premoderation_required": True}
                }
            },
            True,
        ),
        ({"_impact_private": {"impact_cycle_v1": "malformed"}}, True),
    ],
)
async def test_that_premoderation_cycle_context_is_fail_closed(
    container: Container,
    agent: Agent,
    metadata: dict[str, Any],
    expected: bool,
) -> None:
    context, _ = await _engine_context(container, agent, metadata)

    assert is_premoderation_required(context) is expected


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
