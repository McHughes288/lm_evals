import json
import os
import time

import openai

OPENAI_API_MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY", "")


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = 1.4,
    top_p: float = 0.975,
    max_tokens: int = 100,
    verbose: bool = False,
):
    if verbose:
        print(prompt)
    while True:
        try:
            if not model.startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                response = response.choices[0].text.strip()
            else:
                # Use chat completion API
                messages = [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=1,
                    stop=None,
                )
                response = response.choices[0].message.content.strip()
            if verbose:
                print(response)
            return response
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break


def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data
