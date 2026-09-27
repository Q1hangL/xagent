"""Explicit-credential construction, strict catalog reads, and SDK-thread context.

Hosts that run configurations owned by someone other than the deployment
(a model a user configured with their own key) need three guarantees from
the shared layer:

- the request authenticates with the configuration's own key, never an
  ambient deployment credential (``explicit_credentials_only``);
- a catalog read that failed can be told apart from an empty catalog
  (``raise_on_error``), without changing what existing callers get;
- context variables set by the caller reach the SDK thread an adapter uses.

Only the HTTP transport or the SDK client object is replaced; keys are
synthetic.
"""

import contextvars
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM
from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.model import ChatModelConfig
from xagent.web.services.model_list_service import fetch_models_from_provider

CALLER_KEY = "sk-explicit-CALLER-0001"
PLATFORM_AD_TOKEN = "platform-entra-token-0001"
ENDPOINT = "https://my-team.openai.azure.com"


def _completion(model: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def wire(monkeypatch):
    """Record every request the SDK would send; answer with ``wire.reply``."""
    sent: list[httpx.Request] = []
    state = {"reply": lambda request: httpx.Response(200, json=_completion("m"))}

    async def handle_async_request(self, request):
        sent.append(request)
        response = state["reply"](request)
        response.request = request
        return response

    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
    )
    return SimpleNamespace(sent=sent, state=state)


def _assert_caller_key_only(request: httpx.Request) -> None:
    # Some openai SDK versions also send the api key as a Bearer header; the
    # guarantee is that every credential header carries the configured key.
    assert request.headers.get("api-key") == CALLER_KEY
    assert request.headers.get("authorization") in (None, f"Bearer {CALLER_KEY}")
    assert PLATFORM_AD_TOKEN not in str(request.headers)


class TestAzureAuthentication:
    async def test_api_key_only_never_sends_an_ambient_entra_token(
        self, wire, monkeypatch
    ):
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = AzureOpenAILLM(
            model_name="my-deployment",
            azure_endpoint=ENDPOINT,
            api_key=CALLER_KEY,
            api_key_only=True,
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        # The SDK's per-request copies keep the same authentication.
        llm._ensure_client()
        copied = llm._client.with_options(timeout=5)
        await copied.chat.completions.create(
            model="my-deployment", messages=[{"role": "user", "content": "hi"}]
        )
        assert len(wire.sent) == 2
        for request in wire.sent:
            assert request.url.host == "my-team.openai.azure.com"
            _assert_caller_key_only(request)

    async def test_default_azure_behavior_is_unchanged(self, wire, monkeypatch):
        # Deployment-owned Azure models that rely on an Entra token from the
        # environment keep doing so: the isolation is opt-in.
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = AzureOpenAILLM(
            model_name="my-deployment", azure_endpoint=ENDPOINT, api_key=CALLER_KEY
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        assert (
            wire.sent[0].headers.get("authorization") == f"Bearer {PLATFORM_AD_TOKEN}"
        )

    def test_api_key_only_requires_a_key(self):
        with pytest.raises(ValueError):
            AzureOpenAILLM(
                model_name="d", azure_endpoint=ENDPOINT, api_key="", api_key_only=True
            )

    async def test_the_factory_flag_reaches_the_azure_adapter(self, wire, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", PLATFORM_AD_TOKEN)
        llm = create_base_llm(
            ChatModelConfig(
                id="my-deployment",
                model_name="my-deployment",
                model_provider="azure_openai",
                base_url=ENDPOINT,
                api_key=CALLER_KEY,
                max_retries=1,
                explicit_credentials_only=True,
            )
        )
        await llm.chat([{"role": "user", "content": "hi"}])
        _assert_caller_key_only(wire.sent[0])


class TestExplicitCredentialsFactoryGate:
    @pytest.mark.parametrize("api_key", [None, "", "your-deepseek-key"])
    def test_missing_or_placeholder_keys_are_refused(self, api_key, monkeypatch):
        # DeepSeek would otherwise resolve these from DEEPSEEK_API_KEY.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-platform-deepseek")
        with pytest.raises(ValueError):
            create_base_llm(
                ChatModelConfig(
                    id="deepseek-v4-flash",
                    model_name="deepseek-v4-flash",
                    model_provider="deepseek",
                    api_key=api_key,
                    explicit_credentials_only=True,
                )
            )

    def test_without_the_flag_the_environment_fallback_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-platform-deepseek")
        llm = create_base_llm(
            ChatModelConfig(
                id="deepseek-v4-flash",
                model_name="deepseek-v4-flash",
                model_provider="deepseek",
                api_key="your-deepseek-key",
            )
        )
        assert llm.api_key == "sk-platform-deepseek"


def _status_error(status: int) -> Exception:
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    response = httpx.Response(status, request=request)
    cls = {
        403: openai.PermissionDeniedError,
        429: openai.RateLimitError,
        500: openai.InternalServerError,
    }[status]
    return cls("failed", response=response, body=None)


class TestStrictCatalogReads:
    @pytest.mark.parametrize(
        "error",
        [
            _status_error(403),
            _status_error(429),
            _status_error(500),
            openai.APIConnectionError(
                request=httpx.Request("GET", "https://api.openai.com/v1/models")
            ),
        ],
        ids=["403", "429", "500", "connection"],
    )
    async def test_a_failed_read_raises_only_when_asked(self, error, mocker):
        client = mocker.AsyncMock()
        client.models.list.side_effect = error
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=client
        )
        # Existing callers keep the empty-list answer.
        assert await OpenAILLM.list_available_models("sk-test") == []
        assert await fetch_models_from_provider("openai", "sk-test") == []
        # Strict callers see the failure.
        with pytest.raises(type(error)):
            await OpenAILLM.list_available_models("sk-test", raise_on_error=True)
        with pytest.raises(type(error)):
            await fetch_models_from_provider(
                "openrouter", "sk-test", raise_on_error=True
            )

    async def test_a_genuinely_empty_catalog_is_still_empty(self, mocker):
        client = mocker.AsyncMock()
        client.models.list.return_value = SimpleNamespace(data=[])
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI", return_value=client
        )
        assert (
            await fetch_models_from_provider("openai", "sk-test", raise_on_error=True)
            == []
        )

    async def test_fetchers_without_an_error_mode_ignore_the_option(self):
        # Static catalogs answer locally and have nothing to swallow.
        models = await fetch_models_from_provider(
            "deepseek", "sk-test", raise_on_error=True
        )
        assert {model["id"] for model in models} >= {"deepseek-v4-flash"}

    @pytest.mark.parametrize(
        ("module", "method_path"),
        [
            ("xagent.core.model.chat.basic.claude", "httpx.AsyncClient.get"),
            ("xagent.core.model.chat.basic.zhipu", "httpx.AsyncClient.get"),
        ],
    )
    async def test_other_swallowing_readers_raise_when_asked(
        self, module, method_path, monkeypatch
    ):
        import importlib

        adapter = importlib.import_module(module)
        reader = next(
            getattr(adapter, name).list_available_models
            for name in ("ClaudeLLM", "ZhipuLLM")
            if hasattr(adapter, name)
        )

        async def refuse(*args, **kwargs):
            raise httpx.ReadTimeout("read timed out")

        monkeypatch.setattr(httpx.AsyncClient, "get", refuse)
        with patch(f"{module}.ZhipuAiClient", create=True) as sdk:
            sdk.return_value.models.list.side_effect = AttributeError("no such method")
            assert await reader("sk-test") == []
            with pytest.raises(httpx.ReadTimeout):
                await reader("sk-test", raise_on_error=True)


_CALLER_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "caller_scope", default="unset"
)


def _zhipu_text_response():
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock(content="ok", tool_calls=None)
    return MagicMock(choices=[choice])


class TestZhipuExecutorContext:
    @pytest.fixture
    def zhipu(self):
        client = MagicMock()
        with patch(
            "xagent.core.model.chat.basic.zhipu.ZhipuAiClient", return_value=client
        ):
            llm = ZhipuLLM(api_key="sk-test", abilities=["chat", "vision"])
        llm._client = client
        return llm, client

    async def test_chat_and_vision_run_in_the_callers_context(self, zhipu):
        llm, client = zhipu
        seen: list[str] = []

        def create(**kwargs):
            seen.append(_CALLER_SCOPE.get())
            return _zhipu_text_response()

        client.chat.completions.create.side_effect = create
        token = _CALLER_SCOPE.set("caller-scope")
        try:
            await llm.chat([{"role": "user", "content": "hi"}])
            await llm.vision_chat([{"role": "user", "content": "hi"}])
        finally:
            _CALLER_SCOPE.reset(token)
        assert seen == ["caller-scope", "caller-scope"]

    async def test_the_stream_producer_runs_in_the_callers_context(self, zhipu):
        llm, client = zhipu
        seen: list[str] = []

        def create(**kwargs):
            seen.append(_CALLER_SCOPE.get())
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="ok", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=None,
                )
            ]

        client.chat.completions.create.side_effect = create
        token = _CALLER_SCOPE.set("caller-scope")
        try:
            async for _chunk in llm.stream_chat([{"role": "user", "content": "hi"}]):
                pass
        finally:
            _CALLER_SCOPE.reset(token)
        assert seen == ["caller-scope"]
