"""A smolagents Model that talks to Gemini through Google's own `google-genai`
SDK instead of LiteLLM.

Why this exists: LiteLLM's `vertex_ai/` provider authenticates only via a
service account (google-auth / ADC) and ignores a plain API key. Google's
`google-genai` SDK, by contrast, supports Vertex with an API key
(`Client(vertexai=True, api_key=...)`, "Express mode") as well as the
service-account path. This class gives smolagents a Vertex backend that works
with either credential.

It reuses smolagents' own message/tool normalization (`_prepare_completion_kwargs`
-> OpenAI-format), then converts that to `google-genai` calls, so behavior
matches the other smolagents models.
"""
from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    TokenUsage,
)


def _openai_to_gemini(messages: list[dict], tools: list[dict] | None):
    """Convert smolagents/OpenAI-format messages + tools to google-genai inputs.

    Returns (system_instruction, contents, gemini_tools).
    Assumes message `content` is a plain string (flatten_messages_as_text=True).
    """
    system_bits: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):  # safety if not flattened
            content = "".join(part.get("text", "") for part in content)
        content = content or ""

        if role == "system":
            system_bits.append(content)
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        else:  # user, tool -> present as user turns (tool results as user text)
            text = content if role != "tool" else f"Tool result:\n{content}"
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    gemini_tools = None
    if tools:
        decls = []
        for tool in tools:
            fn = tool.get("function", tool)
            decls.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters") or None,
                )
            )
        gemini_tools = [types.Tool(function_declarations=decls)]

    system_instruction = "\n\n".join(system_bits) if system_bits else None
    return system_instruction, contents, gemini_tools


def _normalize_messages(messages) -> list:
    """Accept both hand-written string content and the agents' list-of-parts.

    smolagents' message cleaner expects `content` to be a list of typed parts;
    a plain string breaks it. We wrap any string content so direct calls like
    `model([ChatMessage(role=USER, content='hi')])` work as well as agent calls.
    """
    norm = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                msg = {**msg, "content": [{"type": "text", "text": content}]}
            norm.append(msg)
        else:  # ChatMessage-like object
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                role = getattr(msg, "role", "user")
                role = getattr(role, "value", role)
                norm.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                norm.append(msg)
    return norm


class VertexAIServerModel(Model):
    """smolagents Model for Gemini via google-genai (Vertex or Gemini API).

    Args:
        model_id: bare Gemini id, e.g. 'gemini-2.5-pro'.
        api_key: if given, authenticates with this API key (Vertex Express when
            use_vertex=True, Gemini API when use_vertex=False). No service account needed.
        project, location: used for the service-account/ADC path when api_key is None.
        use_vertex: route via Vertex AI (True) or the Gemini Developer API (False).
        temperature: sampling temperature.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
        project: str | None = None,
        location: str = "us-central1",
        use_vertex: bool = True,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(flatten_messages_as_text=True, model_id=model_id, **kwargs)
        self.temperature = temperature

        if api_key:
            self._client = genai.Client(vertexai=use_vertex, api_key=api_key)
        elif use_vertex:
            # Service-account / ADC path (GOOGLE_APPLICATION_CREDENTIALS).
            self._client = genai.Client(vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client()  # Gemini API via GOOGLE_API_KEY env

    def generate(
        self,
        messages,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list | None = None,
        **kwargs,
    ) -> ChatMessage:
        ck = self._prepare_completion_kwargs(
            messages=_normalize_messages(messages),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
        )
        system_instruction, contents, gemini_tools = _openai_to_gemini(
            ck["messages"], ck.get("tools")
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", self.temperature),
            stop_sequences=ck.get("stop"),
            tools=gemini_tools,
        )

        response = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )

        # Extract text + any function calls from the first candidate.
        text_parts: list[str] = []
        tool_calls: list[ChatMessageToolCall] = []
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            for i, part in enumerate(parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_calls.append(
                        ChatMessageToolCall(
                            id=getattr(fc, "id", None) or f"call_{i}",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name=fc.name, arguments=dict(fc.args or {})
                            ),
                        )
                    )

        usage = getattr(response, "usage_metadata", None)
        token_usage = (
            TokenUsage(
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            )
            if usage
            else None
        )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            raw=response,
            token_usage=token_usage,
        )
