#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate axllm native tool call support through the OpenAI SDK."
    )
    parser.add_argument("--model", required=True, help="Model name exposed by axllm.")
    parser.add_argument(
        "--api_url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL. Default: http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        default="not-needed",
        help="API key passed to the OpenAI SDK. Default: not-needed",
    )
    parser.add_argument(
        "--city",
        default="Beijing",
        help="City to request from the test tool. Default: Beijing",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="max_tokens for each chat request. Default: 256",
    )
    return parser.parse_args()


def get_weather(location: str) -> dict[str, Any]:
    return {
        "location": location,
        "temperature": 22,
        "unit": "celsius",
        "condition": "sunny",
    }


def tool_result(name: str, arguments: str) -> str:
    try:
        parsed_args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool arguments are not valid JSON: {arguments!r}") from exc

    if name != "get_weather":
        raise ValueError(f"unexpected tool name: {name!r}")

    location = parsed_args.get("location")
    if not isinstance(location, str) or not location.strip():
        raise ValueError(f"missing string argument 'location': {parsed_args!r}")

    return json.dumps(get_weather(location), ensure_ascii=False)


def main() -> int:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.api_url.strip())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, for example Beijing.",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are in a function-calling compliance test. "
                "The only correct first response is a tool call to get_weather. "
                "Do not answer the weather question from memory. "
                "Do not say that you cannot call tools. "
                "Use the available get_weather function whenever the user asks "
                "for weather."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Call get_weather for location={args.city}. "
                "Return no natural-language answer before the tool call."
            ),
        },
    ]

    print("==> Requesting tool call")
    first = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        temperature=0,
        max_tokens=args.max_tokens,
    )

    choice = first.choices[0]
    assistant_message = choice.message
    tool_calls = assistant_message.tool_calls or []

    if not tool_calls:
        print("FAIL: response did not contain tool_calls", file=sys.stderr)
        print(first.model_dump_json(indent=2), file=sys.stderr)
        return 1

    print(f"PASS: received {len(tool_calls)} tool_call(s)")
    messages.append(assistant_message.model_dump(exclude_none=True))

    for call in tool_calls:
        function = call.function
        result = tool_result(function.name, function.arguments)
        print(f"  - {function.name}({function.arguments}) -> {result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
        )

    print("==> Sending tool result back")
    second = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=tools,
        temperature=0,
        max_tokens=args.max_tokens,
    )

    final_message = second.choices[0].message
    content = final_message.content or ""
    if not content.strip():
        print("FAIL: final response is empty", file=sys.stderr)
        print(second.model_dump_json(indent=2), file=sys.stderr)
        return 1

    print("PASS: model accepted tool result and returned final content")
    print("assistant:")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
