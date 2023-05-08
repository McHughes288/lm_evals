import json
import os
import time

import openai

OPENAI_API_MODEL = "gpt-3.5-turbo"
openai.api_key = os.getenv("OPENAI_API_KEY", "")

models2cost = {
    "gpt-4": 0.03,
    "gpt-3.5-turbo": 0.0020,
    "text-davinci-003": 0.0200,
    "text-davinci-002": 0.0200,
    "text-davinci-001": 0.0200,
    "text-curie-001": 0.0020,
    "text-babbage-001": 0.0005,
    "text-ada-001": 0.0004,
    "davinci": 0.0200,
    "curie": 0.0020,
    "babbage": 0.0005,
    "ada": 0.0004,
    "text-embedding-ada-002": 0.0004,
}


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = 1.4,
    top_p: float = 0.975,
    max_tokens: int = 100,
    verbose: bool = False,
):
    assert model in models2cost, f"Please update models2cost to contain {model}"
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
                text = response.choices[0].text.strip()
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
                text = response.choices[0].message.content.strip()
            cost = response.usage.total_tokens * models2cost[model] / 1000
            if verbose:
                print(text)
                print(f"${round(cost,2)}")
            return text, cost
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            print(f"Waiting 10 seconds and trying again due to API error. {e}")
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
