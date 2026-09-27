"""A safe error boundary around every provider call one LLM makes.

Provider and SDK errors carry the provider's response text, and some
providers echo the request's credential back in it. That text travels a long
way once an exception leaves the model: retry warnings, trace events stored
in the database, outbound stream frames, runner logs, and whatever the
calling product renders to its users. A log filter cannot help with most of
those, because they are data, not log records.

:func:`guard_llm_calls` wraps an LLM so that none of that text leaves it.
Every provider-touching operation -- one ``chat``/``vision_chat`` await, each
step of a ``stream_chat`` iteration, and the stream's ``aclose`` -- runs
inside a context manager the caller supplies (a log-redaction scope, say),
and every error raised there is caught inside that context and replaced by a
:class:`ProviderCallError` that carries only a fixed failure code and a fixed
message, with no cause and no context chain. Stream ``ERROR`` chunks, whose
``content``/``raw`` hold the same provider text, are treated as the error
they report. The caller learns each failure code through ``on_failure``.

What stays as it was:

* ``asyncio.CancelledError`` and other ``BaseException`` shutdown signals
  pass through untouched; a cancelled call is not a failed call.
* A context-window rejection becomes :class:`ProviderContextLengthError`,
  a :class:`LLMContextLengthError`, so the agent runtime's compaction
  recovery still recognizes it.
* :class:`LLMToolProtocolError` and ``PROTOCOL_ERROR`` chunks are derived
  from the model's own output, not from a provider error body, and the ReAct
  pattern repairs them from their code and details; they are re-raised as a
  fresh, chain-free copy and passed through respectively.
* Successful responses and ordinary chunks are returned unchanged.

The wrapper exposes the ``BaseLLM`` surface only. There is deliberately no
attribute fallthrough to the wrapped object, so no caller can reach an
unguarded client method by accident.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from typing import Any, List

import httpx

from ..error import is_context_length_error
from ..exceptions import LLMContextLengthError, LLMToolProtocolError
from ..types import ChunkType, StreamChunk
from .base import BaseLLM

logger = logging.getLogger(__name__)

# Fixed failure vocabulary. Callers map these to their own user-facing text.
CONTEXT_LENGTH = "context_length"
CREDENTIAL_REJECTED = "credential_rejected"
PROVIDER_QUOTA = "provider_quota"
RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
PROVIDER_UNAVAILABLE = "provider_unavailable"
MODEL_NOT_AVAILABLE = "model_not_available"
INVALID_REQUEST = "invalid_request"
PROVIDER_ERROR = "provider_error"
# The caller's call scope itself could not be entered (for example the
# credential it needs could not be read); no provider was contacted.
CALL_SCOPE_UNAVAILABLE = "call_scope_unavailable"
# A model that was deliberately configured without vision was asked to see.
VISION_UNAVAILABLE = "vision_unavailable"

PROVIDER_CALL_FAILURE_CODES = frozenset(
    {
        CONTEXT_LENGTH,
        CREDENTIAL_REJECTED,
        PROVIDER_QUOTA,
        RATE_LIMITED,
        TIMEOUT,
        PROVIDER_UNAVAILABLE,
        MODEL_NOT_AVAILABLE,
        INVALID_REQUEST,
        PROVIDER_ERROR,
        CALL_SCOPE_UNAVAILABLE,
        VISION_UNAVAILABLE,
    }
)

_SAFE_MESSAGES = {
    # Keeps a "context length exceeded" marker so is_context_length_error
    # recognizes the message as well as the type.
    CONTEXT_LENGTH: "Model provider call failed: context length exceeded.",
    CREDENTIAL_REJECTED: "Model provider call failed: the credential was rejected.",
    PROVIDER_QUOTA: "Model provider call failed: the provider account has no quota left.",
    RATE_LIMITED: "Model provider call failed: rate limited by the provider.",
    TIMEOUT: "Model provider call failed: the request timed out.",
    PROVIDER_UNAVAILABLE: "Model provider call failed: the provider is unavailable.",
    MODEL_NOT_AVAILABLE: "Model provider call failed: the model is not available.",
    INVALID_REQUEST: "Model provider call failed: the provider rejected the request.",
    PROVIDER_ERROR: "Model provider call failed.",
    CALL_SCOPE_UNAVAILABLE: "Model call could not be prepared.",
    VISION_UNAVAILABLE: "This model is not configured to read images.",
}

_CREDENTIAL_ERROR_NAMES = frozenset({"AuthenticationError", "PermissionDeniedError"})
_RATE_LIMIT_ERROR_NAMES = frozenset({"RateLimitError"})
_NOT_FOUND_ERROR_NAMES = frozenset({"NotFoundError"})
_TIMEOUT_ERROR_NAMES = frozenset({"APITimeoutError", "LLMTimeoutError"})
_UNAVAILABLE_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "InternalServerError",
        "OverloadedError",
        "ServiceUnavailableError",
    }
)
_QUOTA_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient balance",
    "insufficient_balance",
    "exceeded your current quota",
    "billing hard limit",
)


class ProviderCallError(RuntimeError):
    """A provider call failed; only a fixed code and message are kept."""

    def __init__(self, code: str) -> None:
        self.code = code if code in PROVIDER_CALL_FAILURE_CODES else PROVIDER_ERROR
        super().__init__(_SAFE_MESSAGES[self.code])


class ProviderContextLengthError(LLMContextLengthError):
    """A context-window rejection with the provider's text removed."""

    def __init__(self) -> None:
        self.code = CONTEXT_LENGTH
        super().__init__(_SAFE_MESSAGES[CONTEXT_LENGTH])


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < 8 and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _http_status(exc: BaseException) -> int | None:
    candidates = [
        getattr(exc, name, None) for name in ("status_code", "code", "status")
    ]
    candidates.append(getattr(getattr(exc, "response", None), "status_code", None))
    for value in candidates:
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 100 <= value <= 599
        ):
            return value
    return None


def _mentions_quota(exc: BaseException) -> bool:
    # Read, never emitted: only the fixed code leaves this module.
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.lower() in {
        "insufficient_quota",
        "insufficient_balance",
    }:
        return True
    try:
        text = str(exc).lower()
    except Exception:  # noqa: BLE001 - an unprintable error has no text to match
        return False
    return any(marker in text for marker in _QUOTA_MARKERS)


def classify_provider_failure(exc: BaseException) -> str:
    """The fixed failure code for an error raised by a provider call.

    Reads type names, HTTP statuses, and a few well-known quota markers
    anywhere in the exception chain; the provider's text itself is never
    returned. Anything unrecognized is ``provider_error``.
    """
    if is_context_length_error(exc):
        return CONTEXT_LENGTH
    causes = _chain(exc)
    names = {type(cause).__name__ for cause in causes}
    statuses = {
        status for cause in causes if (status := _http_status(cause)) is not None
    }
    if names & _CREDENTIAL_ERROR_NAMES or statuses & {401, 403}:
        return CREDENTIAL_REJECTED
    if 402 in statuses or any(_mentions_quota(cause) for cause in causes):
        return PROVIDER_QUOTA
    if names & _RATE_LIMIT_ERROR_NAMES or 429 in statuses:
        return RATE_LIMITED
    if names & _NOT_FOUND_ERROR_NAMES or 404 in statuses:
        return MODEL_NOT_AVAILABLE
    if (
        names & _TIMEOUT_ERROR_NAMES
        or 408 in statuses
        or any(
            isinstance(
                cause, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)
            )
            for cause in causes
        )
    ):
        return TIMEOUT
    if (
        names & _UNAVAILABLE_ERROR_NAMES
        or any(status >= 500 for status in statuses)
        or any(
            isinstance(cause, (ConnectionError, httpx.TransportError))
            for cause in causes
        )
    ):
        return PROVIDER_UNAVAILABLE
    if any(400 <= status < 500 for status in statuses):
        return INVALID_REQUEST
    return PROVIDER_ERROR


def _safe_error(code: str) -> Exception:
    if code == CONTEXT_LENGTH:
        return ProviderContextLengthError()
    return ProviderCallError(code)


def _detached_protocol_error(exc: LLMToolProtocolError) -> LLMToolProtocolError:
    """A copy of a model-output protocol error without its exception chain."""
    return LLMToolProtocolError(
        provider=exc.provider,
        code=exc.code,
        message=exc.protocol_message,
        details=exc.details,
    )


class BoundaryLLM(BaseLLM):
    """An LLM whose provider calls all run inside one safe error boundary.

    Build it with :func:`guard_llm_calls`.
    """

    def __init__(
        self,
        inner: BaseLLM,
        *,
        call_scope: Callable[[], AbstractContextManager[Any]],
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._call_scope = call_scope
        self._on_failure = on_failure
        # Fixed at construction: the wrapped model is immutable for its run.
        self.context_window = inner.context_window
        self._model_id = inner.model_id or None

    # -- BaseLLM surface ----------------------------------------------------

    @property
    def abilities(self) -> List[str]:
        return list(self._inner.abilities)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def supports_thinking_mode(self) -> bool:
        return self._inner.supports_thinking_mode

    @property
    def supports_json_schema_response_format(self) -> bool:
        return self._inner.supports_json_schema_response_format

    @property
    def supports_json_object_response_format(self) -> bool:
        return self._inner.supports_json_object_response_format

    @property
    def supports_native_video_input(self) -> bool:
        return self._inner.supports_native_video_input

    @property
    def supports_native_video_with_images(self) -> bool:
        return self._inner.supports_native_video_with_images

    @property
    def supports_native_video_time_range(self) -> bool:
        return self._inner.supports_native_video_time_range

    def build_native_video_content(
        self,
        video_url: str,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, Any]:
        return self._inner.build_native_video_content(
            video_url, start_time=start_time, end_time=end_time
        )

    # -- the boundary --------------------------------------------------------

    def _open_scope(self) -> contextlib.ExitStack:
        """Enter the caller's scope, or fail with a safe, chain-free error."""
        stack = contextlib.ExitStack()
        failed_type: str | None = None
        try:
            stack.enter_context(self._call_scope())
        except Exception as exc:  # noqa: BLE001 - replaced below, outside the handler
            failed_type = type(exc).__name__
        if failed_type is not None:
            # No scope is active here, so only the type name is logged.
            logger.warning("Model call scope could not be entered (%s)", failed_type)
            self._report(CALL_SCOPE_UNAVAILABLE)
            raise ProviderCallError(CALL_SCOPE_UNAVAILABLE)
        return stack

    def _report(self, code: str) -> None:
        if self._on_failure is None:
            return
        try:
            self._on_failure(code)
        except Exception as exc:  # noqa: BLE001 - reporting must not replace the failure
            logger.warning("Model failure callback raised (%s)", type(exc).__name__)

    def _record_failure(self, exc: BaseException, operation: str) -> str:
        """Classify and log one provider error. Call inside the caller's scope."""
        code = classify_provider_failure(exc)
        # Inside the caller's scope: a redacting scope scrubs the known secret
        # values from this record's message and traceback.
        logger.warning(
            "Model provider %s failed: %s (%s)",
            operation,
            code,
            type(exc).__name__,
            exc_info=exc,
        )
        self._report(code)
        return code

    async def _guarded_call(self, operation: str, **kwargs: Any) -> Any:
        failure: str | None = None
        protocol_error: LLMToolProtocolError | None = None
        with self._open_scope():
            try:
                return await getattr(self._inner, operation)(**kwargs)
            except LLMToolProtocolError as exc:
                protocol_error = _detached_protocol_error(exc)
            except Exception as exc:  # noqa: BLE001 - replaced below, outside the handler
                failure = self._record_failure(exc, operation)
        # Raised outside the handlers, so neither error carries a context chain.
        if protocol_error is not None:
            raise protocol_error
        raise _safe_error(failure or PROVIDER_ERROR)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        return await self._guarded_call(  # type: ignore[no-any-return]
            "chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        return await self._guarded_call(  # type: ignore[no-any-return]
            "vision_chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            thinking=thinking,
            output_config=output_config,
            **kwargs,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        # The scope is entered for each step rather than held across ``yield``:
        # what the consumer does between chunks is not a provider call, and a
        # context variable set in one step must be reset in the same context.
        failure: str | None = None
        protocol_error: LLMToolProtocolError | None = None
        stream: Any = None
        try:
            with self._open_scope():
                try:
                    stream = aiter(
                        self._inner.stream_chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,
                            tool_choice=tool_choice,
                            response_format=response_format,
                            thinking=thinking,
                            output_config=output_config,
                            **kwargs,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    failure = self._record_failure(exc, "stream_chat")
            while failure is None and protocol_error is None:
                chunk: StreamChunk | None = None
                finished = False
                with self._open_scope():
                    try:
                        chunk = await anext(stream)
                    except StopAsyncIteration:
                        finished = True
                    except LLMToolProtocolError as exc:
                        protocol_error = _detached_protocol_error(exc)
                    except Exception as exc:  # noqa: BLE001
                        failure = self._record_failure(exc, "stream_chat")
                    if chunk is not None and chunk.type == ChunkType.ERROR:
                        failure = self._record_error_chunk(chunk)
                        chunk = None
                if finished or chunk is None:
                    break
                yield chunk
        finally:
            await self._close_stream(stream)
        if protocol_error is not None:
            raise protocol_error
        if failure is not None:
            raise _safe_error(failure)

    def _record_error_chunk(self, chunk: StreamChunk) -> str:
        """Classify an ``ERROR`` chunk as the error it reports. Inside scope."""
        raw = chunk.raw
        if isinstance(raw, BaseException):
            return self._record_failure(raw, "stream_chat")
        code = classify_provider_failure(RuntimeError(chunk.content or ""))
        logger.warning("Model provider stream_chat reported an error chunk: %s", code)
        self._report(code)
        return code

    async def _close_stream(self, stream: Any) -> None:
        aclose = getattr(stream, "aclose", None)
        if not callable(aclose):
            return
        try:
            with self._open_scope():
                try:
                    await aclose()
                except Exception as exc:  # noqa: BLE001 - cleanup must not replace the outcome
                    code = classify_provider_failure(exc)
                    logger.warning(
                        "Closing a model provider stream failed: %s (%s)",
                        code,
                        type(exc).__name__,
                        exc_info=exc,
                    )
        except ProviderCallError:
            # The scope could not be entered for cleanup; the stream is left
            # to garbage collection rather than closed outside the scope.
            pass


def guard_llm_calls(
    llm: BaseLLM,
    *,
    call_scope: Callable[[], AbstractContextManager[Any]],
    on_failure: Callable[[str], None] | None = None,
) -> BaseLLM:
    """Wrap ``llm`` so every provider call runs inside ``call_scope`` and fails safely.

    ``call_scope`` is entered afresh around each provider-touching operation
    and must be cheap and re-entrant. ``on_failure`` receives the fixed
    failure code of each error, inside the scope; it must not raise.
    """
    return BoundaryLLM(llm, call_scope=call_scope, on_failure=on_failure)


class UnavailableVisionModel(BaseLLM):
    """Stands in for a vision model that was deliberately left out.

    Supplying it where a vision model is expected keeps tools from falling
    back to a deployment default vision model; any call is refused with a
    safe ``vision_unavailable`` error and nothing is sent anywhere.
    """

    @property
    def abilities(self) -> List[str]:
        return []

    @property
    def model_name(self) -> str:
        return "vision-unavailable"

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        raise ProviderCallError(VISION_UNAVAILABLE)

    async def vision_chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        output_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        raise ProviderCallError(VISION_UNAVAILABLE)
