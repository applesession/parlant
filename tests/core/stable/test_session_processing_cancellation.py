import asyncio
from contextlib import suppress
from types import SimpleNamespace

from parlant.core.app_modules.sessions import SessionModule
from parlant.core.sessions import EventId, SessionId


class BackgroundTasks:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task
        self.cancelled_tags: list[str] = []

    async def cancel(self, *, tag: str, reason: str) -> None:
        self.cancelled_tags.append(tag)
        self.task.cancel(reason)


def module_with_task(
    session_id: SessionId,
    trigger_event_id: EventId,
    task: asyncio.Task[None],
) -> tuple[SessionModule, BackgroundTasks]:
    module = SessionModule.__new__(SessionModule)
    background_tasks = BackgroundTasks(task)
    module._session_store = SimpleNamespace(read_session=lambda _: asyncio.sleep(0))
    module._background_task_service = background_tasks
    module._processing_tasks = {session_id: (trigger_event_id, task)}
    module._processing_tasks_lock = asyncio.Lock()
    return module, background_tasks


async def test_matching_trigger_cancels_only_current_processing() -> None:
    session_id = SessionId("session-1")
    trigger_event_id = EventId("event-1")
    task = asyncio.create_task(asyncio.Event().wait())
    module, background_tasks = module_with_task(session_id, trigger_event_id, task)

    assert await module.cancel_processing(session_id, trigger_event_id) == "cancelled"
    await asyncio.sleep(0)
    assert task.cancelled()
    assert background_tasks.cancelled_tags == ["process-session(session-1)"]


async def test_old_trigger_cannot_cancel_new_processing() -> None:
    session_id = SessionId("session-1")
    task = asyncio.create_task(asyncio.Event().wait())
    module, background_tasks = module_with_task(session_id, EventId("event-2"), task)

    assert await module.cancel_processing(session_id, EventId("event-1")) == "not_current"
    assert not task.done()
    assert background_tasks.cancelled_tags == []
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def test_finished_processing_is_reported_without_replay() -> None:
    session_id = SessionId("session-1")
    task = asyncio.create_task(asyncio.sleep(0))
    await task
    module, background_tasks = module_with_task(session_id, EventId("event-1"), task)

    assert await module.cancel_processing(session_id, EventId("event-1")) == "already_finished"
    assert background_tasks.cancelled_tags == []
