"""The safe error boundary around an LLM's provider calls.

Keys are synthetic. The real-SDK tests replace only the bottom httpx
transport, with a fake provider that echoes the received credential in its
error bodies -- the case the boundary exists for.
"""

import asyncio
import contextvars
import json
import logging
from contextlib import contextmanager
from typing import Any, AsyncIterator, List

import httpx
import pytest

from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import (
    CALL_SCOPE_UNAVAILABLE,
    CONTEXT_LENGTH,
    CREDENTIAL_REJECTED,
    INVALID_REQUEST,
    MODEL_NOT_AVAILABLE,
    PROVIDER_ERROR,
    PROVIDER_QUOTA,
    PROVIDER_UNAVAILABLE,
    RATE_LIMITED,
    TIMEOUT,
    VISION_UNAVAILABLE,
    ProviderCallError,
    ProviderContextLengthError,
    UnavailableVisionModel,
    classify_provider_failure,
    guard_llm_calls,
)
from xagent.core.model.chat.error import is_context_length_error
from xagent.core.model.chat.exceptions import (
    LLMContextLengthError,
    LLMToolProtocolError,
)
from xagent.core.model.chat.types import ChunkType, StreamChunk
from xagent.core.model.model import ChatModelConfig

SECRET = "sk-boundary-CALLER-SECRET-0001"
MESSAGES = [{"role": "user", "content": "hi"}]

in_scope: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "boundary_test_scope", default=False
)


class ScopeProbe:
    """A call scope that records how often it was entered and exited."""

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    @contextmanager
    def __call__(self):
        self.entered += 1
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)
            self.exited += 1


class ScopeRecorder(logging.Handler):
    """Captures each record with whether it was created inside the scope."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[logging.LogRecord, bool]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append((record, in_scope.get()))


@pytest.fixture
def recorder():
    handler = ScopeRecorder()
    root = logging.getLogger()
    saved = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        root.setLevel(saved)


class FakeLLM(BaseLLM):
    """A scripted model: ``behaviour`` decides what each call does."""

    def __init__(self, behaviour: Any = "ok", abilities: list[str] | None = None):
        self.behaviour = behaviour
        self._abilities = abilities or ["chat", "tool_calling"]
        self.scope_during_calls: list[bool] = []
        self.closed = False
        self.close_in_scope: bool | None = None
        self.api_key = SECRET
        self.context_window = 12345
        self._model_id = "caller-model-id"

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        self.scope_during_calls.append(in_scope.get())
        await asyncio.sleep(0)
        self.scope_during_calls.append(in_scope.get())
        if isinstance(self.behaviour, BaseException):
            raise self.behaviour
        return {"type": "text", "content": "answer"}

    async def stream_chat(self, messages, **kwargs) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        try:
            for item in self.behaviour:
                self.scope_during_calls.append(in_scope.get())
                await asyncio.sleep(0)
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.closed = True
            self.close_in_scope = in_scope.get()


class HttpError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(Exception):
    pass


def _echo(status: int) -> HttpError:
    return HttpError(f"provider says: bad key {SECRET}", status)


def _assert_safe(exc: BaseException) -> None:
    assert SECRET not in str(exc)
    assert SECRET not in repr(exc)
    assert exc.__cause__ is None
    assert exc.__context__ is None


class TestClassification:
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (_echo(401), CREDENTIAL_REJECTED),
            (_echo(403), CREDENTIAL_REJECTED),
            (AuthenticationError("nope"), CREDENTIAL_REJECTED),
            (_echo(402), PROVIDER_QUOTA),
            (HttpError("You exceeded your current quota", 429), PROVIDER_QUOTA),
            (_echo(429), RATE_LIMITED),
            (_echo(404), MODEL_NOT_AVAILABLE),
            (_echo(408), TIMEOUT),
            (asyncio.TimeoutError(), TIMEOUT),
            (httpx.ReadTimeout("slow"), TIMEOUT),
            (_echo(503), PROVIDER_UNAVAILABLE),
            (httpx.ConnectError("refused"), PROVIDER_UNAVAILABLE),
            (_echo(400), INVALID_REQUEST),
            (RuntimeError("maximum context length is 8192 tokens"), CONTEXT_LENGTH),
            (ValueError(f"odd {SECRET}"), PROVIDER_ERROR),
        ],
    )
    def test_codes(self, error, code):
        assert classify_provider_failure(error) == code

    def test_a_wrapped_cause_is_classified(self):
        try:
            try:
                raise _echo(401)
            except HttpError as inner:
                raise RuntimeError(f"adapter failed: {inner}") from inner
        except RuntimeError as outer:
            assert classify_provider_failure(outer) == CREDENTIAL_REJECTED


class TestChat:
    async def test_success_passes_through_inside_the_scope(self):
        probe = ScopeProbe()
        inner = FakeLLM()
        llm = guard_llm_calls(inner, call_scope=probe)
        assert await llm.chat(MESSAGES) == {"type": "text", "content": "answer"}
        assert inner.scope_during_calls == [True, True]
        assert (probe.entered, probe.exited) == (1, 1)
        assert not in_scope.get()

    @pytest.mark.parametrize("operation", ["chat", "vision_chat"])
    async def test_provider_error_is_replaced_inside_the_scope(
        self, recorder, operation
    ):
        failures: list[str] = []
        inner = FakeLLM(behaviour=_echo(401), abilities=["chat", "vision"])
        llm = guard_llm_calls(
            inner, call_scope=ScopeProbe(), on_failure=failures.append
        )
        with pytest.raises(ProviderCallError) as caught:
            await getattr(llm, operation)(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert failures == [CREDENTIAL_REJECTED]
        boundary_logs = [
            (record, scoped)
            for record, scoped in recorder.records
            if record.name.endswith("call_boundary")
        ]
        assert boundary_logs and all(scoped for _record, scoped in boundary_logs)

    async def test_context_length_keeps_its_runtime_meaning(self):
        llm = guard_llm_calls(
            FakeLLM(behaviour=RuntimeError(f"context_length_exceeded for {SECRET}")),
            call_scope=ScopeProbe(),
        )
        with pytest.raises(ProviderContextLengthError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert isinstance(caught.value, LLMContextLengthError)
        assert is_context_length_error(caught.value)

    async def test_tool_protocol_error_is_copied_without_its_chain(self):
        protocol = LLMToolProtocolError(
            provider="openrouter",
            code="malformed_tool_arguments",
            message="arguments were not JSON",
            details={"tool": "search"},
        )
        protocol.__context__ = _echo(400)
        llm = guard_llm_calls(FakeLLM(behaviour=protocol), call_scope=ScopeProbe())
        with pytest.raises(LLMToolProtocolError) as caught:
            await llm.chat(MESSAGES)
        assert caught.value is not protocol
        assert caught.value.code == "malformed_tool_arguments"
        assert caught.value.details == {"tool": "search"}
        assert caught.value.__context__ is None and caught.value.__cause__ is None

    async def test_cancellation_is_not_a_failure(self):
        failures: list[str] = []

        class Hanging(FakeLLM):
            async def chat(self, messages, **kwargs):  # type: ignore[override]
                await asyncio.Event().wait()

        probe = ScopeProbe()
        llm = guard_llm_calls(Hanging(), call_scope=probe, on_failure=failures.append)
        task = asyncio.ensure_future(llm.chat(MESSAGES))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == []
        assert probe.entered == probe.exited == 1

    async def test_a_scope_that_cannot_open_fails_safely_and_sends_nothing(self):
        inner = FakeLLM()

        @contextmanager
        def broken():
            try:
                json.loads(f"{{{SECRET}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("credential unreadable") from exc
            yield  # pragma: no cover

        failures: list[str] = []
        llm = guard_llm_calls(inner, call_scope=broken, on_failure=failures.append)
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CALL_SCOPE_UNAVAILABLE
        assert inner.scope_during_calls == []
        assert failures == [CALL_SCOPE_UNAVAILABLE]

    async def test_a_raising_failure_callback_does_not_replace_the_error(self):
        def explode(_code: str) -> None:
            raise RuntimeError("callback bug")

        llm = guard_llm_calls(
            FakeLLM(behaviour=_echo(429)), call_scope=ScopeProbe(), on_failure=explode
        )
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        assert caught.value.code == RATE_LIMITED


class TestStream:
    async def test_every_step_and_the_close_run_inside_the_scope(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="he", delta="he"),
            StreamChunk(type=ChunkType.TOKEN, content="hello", delta="llo"),
        ]
        probe = ScopeProbe()
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(inner, call_scope=probe)
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["he", "llo"]
        assert inner.scope_during_calls == [True, True]
        assert inner.closed and inner.close_in_scope is True
        assert probe.entered == probe.exited
        assert not in_scope.get()

    async def test_a_mid_stream_error_ends_the_stream_safely(self):
        chunks: list[Any] = [
            StreamChunk(type=ChunkType.TOKEN, content="partial", delta="partial"),
            HttpError(f"upstream rejected key {SECRET}", 500),
        ]
        failures: list[str] = []
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(
            inner, call_scope=ScopeProbe(), on_failure=failures.append
        )
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        assert [chunk.delta for chunk in received] == ["partial"]
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_UNAVAILABLE
        assert failures == [PROVIDER_UNAVAILABLE]
        assert inner.closed

    async def test_an_error_chunk_is_treated_as_the_error_it_reports(self):
        echoed = _echo(401)
        chunks = [
            StreamChunk(
                type=ChunkType.ERROR,
                content=f"Zhipu streaming API error: {echoed}",
                raw=echoed,
            )
        ]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        assert received == []
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED

    async def test_an_error_chunk_without_an_exception_is_classified_by_text(self):
        chunks = [StreamChunk(type=ChunkType.ERROR, content=f"LLM failed {SECRET}")]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass
        _assert_safe(caught.value)
        assert caught.value.code == PROVIDER_ERROR

    async def test_protocol_error_chunks_pass_through(self):
        chunks = [
            StreamChunk(
                type=ChunkType.PROTOCOL_ERROR,
                protocol_error={"code": "unavailable_tool_call"},
            )
        ]
        llm = guard_llm_calls(FakeLLM(behaviour=chunks), call_scope=ScopeProbe())
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.type for chunk in received] == [ChunkType.PROTOCOL_ERROR]

    async def test_early_close_by_the_consumer_closes_the_provider_stream(self):
        chunks = [
            StreamChunk(type=ChunkType.TOKEN, content="a", delta="a"),
            StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b"),
        ]
        probe = ScopeProbe()
        inner = FakeLLM(behaviour=chunks)
        llm = guard_llm_calls(inner, call_scope=probe)
        stream = llm.stream_chat(messages=MESSAGES)
        first = await anext(stream)
        assert first.delta == "a"
        await stream.aclose()
        assert inner.closed and inner.close_in_scope is True
        assert probe.entered == probe.exited

    async def test_a_failing_close_does_not_replace_the_outcome(self, recorder):
        class BadClose:
            def __init__(self) -> None:
                self.done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.done:
                    raise StopAsyncIteration
                self.done = True
                return StreamChunk(type=ChunkType.TOKEN, content="x", delta="x")

            async def aclose(self):
                raise HttpError(f"close echoed {SECRET}", 500)

        class Inner(FakeLLM):
            def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                return BadClose()

        llm = guard_llm_calls(Inner(), call_scope=ScopeProbe())
        received = [chunk async for chunk in llm.stream_chat(messages=MESSAGES)]
        assert [chunk.delta for chunk in received] == ["x"]
        close_logs = [
            (record, scoped)
            for record, scoped in recorder.records
            if record.getMessage().startswith("Closing a model provider stream failed")
        ]
        assert close_logs and all(scoped for _record, scoped in close_logs)

    async def test_cancellation_mid_stream_propagates_and_closes(self):
        release = asyncio.Event()

        class Slow(FakeLLM):
            async def stream_chat(self, messages, **kwargs):  # type: ignore[override]
                try:
                    yield StreamChunk(type=ChunkType.TOKEN, content="a", delta="a")
                    await release.wait()
                    yield StreamChunk(type=ChunkType.TOKEN, content="ab", delta="b")
                finally:
                    self.closed = True

        failures: list[str] = []
        inner = Slow()
        probe = ScopeProbe()
        llm = guard_llm_calls(inner, call_scope=probe, on_failure=failures.append)

        async def consume() -> None:
            async for _chunk in llm.stream_chat(messages=MESSAGES):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert failures == []
        assert inner.closed
        assert probe.entered == probe.exited


class TestSurface:
    def test_model_facts_are_forwarded(self):
        inner = FakeLLM(abilities=["chat", "tool_calling", "vision"])
        llm = guard_llm_calls(inner, call_scope=ScopeProbe())
        assert llm.abilities == ["chat", "tool_calling", "vision"]
        assert llm.has_ability("vision")
        assert llm.model_name == "fake-model"
        assert llm.model_id == "caller-model-id"
        assert llm.context_window == 12345

    def test_no_attribute_reaches_the_wrapped_model(self):
        llm = guard_llm_calls(FakeLLM(), call_scope=ScopeProbe())
        with pytest.raises(AttributeError):
            llm.api_key  # noqa: B018
        with pytest.raises(AttributeError):
            llm.behaviour  # noqa: B018


class TestUnavailableVisionModel:
    @pytest.mark.parametrize("operation", ["chat", "vision_chat"])
    async def test_every_call_is_refused(self, operation):
        model = UnavailableVisionModel()
        assert not model.has_ability("vision")
        with pytest.raises(ProviderCallError) as caught:
            await getattr(model, operation)(MESSAGES)
        assert caught.value.code == VISION_UNAVAILABLE


class TestRealOpenAIPath:
    """create_base_llm -> RetryWrapper -> OpenAILLM -> openai SDK -> httpx."""

    @pytest.fixture
    def echo_provider(self, monkeypatch):
        state: dict[str, Any] = {"mode": "401", "requests": []}

        async def handle_async_request(self, request):
            await request.aread()
            auth = request.headers.get("authorization", "")
            state["requests"].append(auth)
            key = auth[7:] if auth.lower().startswith("bearer ") else auth
            if state["mode"] == "401":
                response = httpx.Response(
                    401,
                    json={"error": {"message": f"Incorrect API key provided: {key}"}},
                )
            elif state["mode"] == "sse-midway":
                first = {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "he"},
                            "finish_reason": None,
                        }
                    ],
                }
                error = {"error": {"message": f"upstream rejected key {key}"}}
                body = f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n"
                response = httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            else:  # pragma: no cover - test bug
                raise AssertionError(state["mode"])
            response.request = request
            return response

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
        )
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
            monkeypatch.delenv(name, raising=False)
        return state

    def _caller_llm(self) -> BaseLLM:
        return create_base_llm(
            ChatModelConfig(
                id="gpt-4o",
                model_provider="openai",
                model_name="gpt-4o",
                api_key=SECRET,
                max_retries=1,
            )
        )

    async def test_an_echoed_key_never_leaves_a_chat_failure(
        self, echo_provider, recorder
    ):
        llm = guard_llm_calls(self._caller_llm(), call_scope=ScopeProbe())
        with pytest.raises(ProviderCallError) as caught:
            await llm.chat(MESSAGES)
        _assert_safe(caught.value)
        assert caught.value.code == CREDENTIAL_REJECTED
        assert echo_provider["requests"] == [f"Bearer {SECRET}"]
        # Every log record that could carry the echo was made inside the scope.
        leaking = [
            record
            for record, scoped in recorder.records
            if not scoped
            and SECRET in (record.getMessage() + str(record.exc_info or ""))
        ]
        assert leaking == []

    async def test_a_mid_stream_sse_error_never_leaves_the_stream(
        self, echo_provider, recorder
    ):
        echo_provider["mode"] = "sse-midway"
        llm = guard_llm_calls(self._caller_llm(), call_scope=ScopeProbe())
        received: list[StreamChunk] = []
        with pytest.raises(ProviderCallError) as caught:
            async for chunk in llm.stream_chat(messages=MESSAGES):
                received.append(chunk)
        _assert_safe(caught.value)
        leaking = [
            record
            for record, scoped in recorder.records
            if not scoped
            and SECRET in (record.getMessage() + str(record.exc_info or ""))
        ]
        assert leaking == []
