import multiprocessing.pool
import functools

import openai
from tenacity import (
    retry,
    wait_random
)


client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
    organization="org-vK4evWkGrlhQM0YsKET35H0P"
)


def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


@retry(wait=wait_random(min=10, max=20), reraise=True)
@timeout(120)
def get_completion(messages, max_tokens, model="gpt-4"):
    print(f"Calling API with {model}")
    x = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return x
