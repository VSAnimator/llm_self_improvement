from typing import Optional, Callable
import re
import json
import random
from time import time
from collections import Counter
from typing import Optional, Callable, Type
import asyncio
import os
from typing import Dict, List, Optional, Any, Union

import sglang as sgl
import openai
from sglang import set_default_backend, RuntimeEndpoint


TRACE_FILE = "trace_log.json"


def most_frequent_element(elements):
    # Count the occurrences of each element in the list
    counts = Counter(elements)
    # Find the maximum count
    max_count = max(counts.values())
    # Gather all elements that have the maximum count
    most_frequent = [key for key, count in counts.items() if count == max_count]
    # Randomly pick one if there are ties
    return random.choice(most_frequent)


class BaseBackend:
    INSTRUMENTATION = True

    @staticmethod
    def api_call():
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
        trace_id: Optional[str] = "trace",
        trace_label: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:

        def __func_validate(response: str):
            if func_validate is not None:
                return func_validate(response)
            elif regex_constrain is not None:
                return bool(re.fullmatch(regex_constrain, response))
            else:
                return True

        responses = []
        tic = time()
        for _ in range(repeat):
            response = cls.api_call(
                prompt, max_tokens, stop, ignore_eos, regex_constrain, response_format
            )
            if __func_validate(response):
                responses.append(response)
                break
        duration = time() - tic
        if not responses:
            print(
                f"Failed to generate a valid response after {repeat} attempts, return N/A instead"
            )
            responses.append("N/A")
        if cls.INSTRUMENTATION:
            trace_sample = {
                "prompt": prompt,
                "config": {
                    "max_tokens": max_tokens,
                    "stop": stop,
                    "regex_constrain": regex_constrain,
                },
                "label": trace_label,
                "response": responses[0],
                "start_time": tic,
                "duration": duration,
            }

            # Check if the trace file exists
            if not os.path.isfile(TRACE_FILE):
                # If not, create a new file with this single trace sample in a list
                with open(TRACE_FILE, "w", encoding="utf-8") as f:
                    json.dump([trace_sample], f, ensure_ascii=False, indent=2)
            else:
                # If it does exist, load existing data, append this new entry, and overwrite
                with open(TRACE_FILE, "r", encoding="utf-8") as f:
                    try:
                        trace_data = json.load(f)
                    except json.JSONDecodeError:
                        # If the file is not valid JSON (e.g., empty), start fresh
                        trace_data = []

                trace_data.append(trace_sample)
                # Write updated data back to the file
                with open(TRACE_FILE, "w", encoding="utf-8") as f:
                    json.dump(trace_data, f, ensure_ascii=False, indent=2)
        return responses[0]
        # or we can return the most frequent response out of repeat attempts
        # return most_frequent_element(responses)


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
        # set_default_backend(RuntimeEndpoint("http://localhost:30003"))
        client = openai.Client(base_url="http://127.0.0.1:30003/v1", api_key="None")

        response = client.chat.completions.create(
            **{
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "messages": messages,
                "max_tokens": max_tokens,
                "stop": stop,
                "response_format": (
                    {
                        "type": "json_schema",
                        "json_schema": (
                            {
                                "name": "foo",
                                # convert the pydantic model to json schema
                                "schema": response_format.model_json_schema(),
                            }
                        ),
                    }
                    if response_format
                    else None
                ),
                # "response_format": response_format.dict() if response_format else None,
            }
        )

        # return response
        return response.choices[0].message.content

        response = func_wrapper.run(prompt)
        return response["response"]

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
        trace_id: Optional[str] = "trace",
        trace_label: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await asyncio.to_thread(
            super().generate,
            messages,
            max_tokens,
            repeat,
            stop,
            ignore_eos,
            regex_constrain,
            func_validate,
            trace_id,
            trace_label,
            response_format,
        )
