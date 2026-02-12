# /// script
# dependencies = [
#     "marimo",
#     "openai==2.16.0",
#     "polars==1.37.1",
#     "python-dotenv==1.2.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from openai import OpenAI
    import polars as pl
    from dotenv import load_dotenv
    import os
    from pathlib import Path

    return OpenAI, load_dotenv, os


@app.cell
def _(OpenAI, load_dotenv, os):
    load_dotenv()

    OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
    client = OpenAI(api_key=OPENAI_API_TOKEN)
    return (client,)


@app.cell
def _(client):
    _response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{
            "role":"user",
            "content":"Quick productivity tip."
        }]
    )

    print(
        _response.choices[0].message.content
    )
    return


@app.cell
def _(client):
    _question="why is earth's core hot and why is it round?"
    _prompt=f"""Summarize this question:{_question} in 4 words"""

    _max_completion_tokens = 50

    _response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{
            "role":"user",
            "content":_prompt,
        }],
        max_completion_tokens=_max_completion_tokens
    )

    print(_response.choices[0].message.content)

    input_tokens = _response.usage.prompt_tokens
    output_tokens = _max_completion_tokens

    # made up pricing
    input_token_price = 0.15 / 1_000_000
    output_token_price = 0.6 / 1_000_000

    cost = (input_tokens * input_token_price + output_tokens * output_token_price)
    print(f"esitmated_cost: ${cost}")

    # try and always estimate costs when deploying features at scale
    return


@app.cell
def _(client):
    _question="why is earth's core hot and why is it round?"
    _number=10
    _prompt=f"""Summarize this question:{_question} in {_number} words"""


    _max_completion_tokens = 50
    _temp = 2.0

    _response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{
            "role":"user",
            "content":_prompt,
        }],
        max_completion_tokens=_max_completion_tokens,
        temperature = _temp
    )

    print(_response.choices[0].message.content)
    return


@app.cell
def _(client):
    # multiple roles for multi-turn conversations

    _response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {"role": "system",
            "content": "You are a Python programming tutor who speaks concisely."},
            {"role": "user",
            "content": "What is the difference between mutable and immutable objects?"}
        ]
    )

    print(_response.choices[0].message.content)
    return


@app.cell
def _():
    # you can use the assistant role for more targeted context
    # you can do chain of thought for clear reasoning.
    # you can ask the ai to play a role -  you can even ask it to play the role of multiple experts, have it do some work and then come to some consensus on what the right solution is.
    return


@app.cell
def _(client):
    mod_response = client.moderations.create(
        input="HI!"
    )
    return (mod_response,)


@app.cell
def _(mod_response):
    mod_response.results[0].categories.self_harm
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
