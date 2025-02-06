import os
import re
import json
import random
import asyncio
from time import time
from collections import Counter
from typing import Dict, List, Optional, Any, Callable, Type
import threading

import openai

TRACE_FILE = "trace_log.json"


def most_frequent_element(elements: List[Any]) -> Any:
    """Returns the most frequent element in a list. If there's a tie, selects randomly."""
    counts = Counter(elements)
    max_count = max(counts.values())
    most_frequent = [key for key, count in counts.items() if count == max_count]
    return random.choice(most_frequent)


class BaseBackend:
    INSTRUMENTATION = True

    @staticmethod
    def api_call():
        """To be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def generate(
        cls: Type["BaseBackend"],
        prompt: str,
        max_tokens: int,
        repeat: int = 1,
        stop: Optional[str] = None,
        ignore_eos: bool = False,
        regex_constrain: Optional[str] = None,
        func_validate: Optional[Callable] = None,
        trace_label: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates a response based on the given prompt using API calls."""

        def is_valid_response(response: str) -> bool:
            """Validates a response using either a regex constraint or a custom validation function."""
            if func_validate:
                return func_validate(response)
            if regex_constrain:
                return bool(re.fullmatch(regex_constrain, response))
            return True

        responses = []
        start_time = time()

        for _ in range(repeat):
            response = cls.api_call(
                prompt, max_tokens, stop, ignore_eos, regex_constrain, response_format
            )
            if is_valid_response(response):
                responses.append(response)
                break

        duration = time() - start_time

        if not responses:
            print(
                f"Failed to generate a valid response after {repeat} attempts. Returning 'N/A'."
            )
            responses.append("N/A")

        if cls.INSTRUMENTATION:
            trace_entry = {
                "name": cls.__name__,
                "ph": "X",
                "ts": int(start_time * 1e6),
                "dur": int(duration * 1e6),
                "pid": os.getpid(),
                "tid": threading.get_ident(),
                "args": {
                    "prompt": prompt,
                    "config": {
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "regex_constrain": regex_constrain,
                    },
                    "label": trace_label,
                    "response": responses[0],
                },
            }
            cls._log_trace(trace_entry)

        return responses[0]

    @staticmethod
    def _log_trace(trace_entry: Dict[str, Any]) -> None:
        """Logs trace information to a JSON file."""
        if not os.path.isfile(TRACE_FILE):
            with open(TRACE_FILE, "w", encoding="utf-8") as f:
                json.dump([trace_entry], f, ensure_ascii=False, indent=2)
        else:
            try:
                with open(TRACE_FILE, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
            except json.JSONDecodeError:
                trace_data = []
            trace_data.append(trace_entry)
            with open(TRACE_FILE, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2)


class SGLangBackend(BaseBackend):
    @staticmethod
    def api_call(
        messages: List[Dict[str, str]],
        max_tokens: int,
        stop: Optional[str],
        ignore_eos: bool,
        regex_constrain: Optional[str],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Calls SGLang using compatible OpenAI API."""
        client = openai.Client(base_url="http://127.0.0.1:30003/v1", api_key="None")

        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=max_tokens,
            stop=stop,
            response_format=(
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "foo",
                        "schema": response_format.model_json_schema(),
                    },
                }
                if response_format
                else None
            ),
        )

        return response.choices[0].message.content

    @classmethod
    async def generate(
        cls,
        messages: List[Dict[str, str]],
        max_tokens: int,
        repeat: int = 1,
        stop: Optional[str] = None,
        ignore_eos: bool = False,
        regex_constrain: Optional[str] = None,
        func_validate: Optional[Callable] = None,
        trace_label: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Asynchronously calls `generate` from `BaseBackend`."""
        return await asyncio.to_thread(
            super().generate,
            messages,
            max_tokens,
            repeat,
            stop,
            ignore_eos,
            regex_constrain,
            func_validate,
            trace_label,
            response_format,
        )
