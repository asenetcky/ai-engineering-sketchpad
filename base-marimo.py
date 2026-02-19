# /// script
# dependencies = [
#     "marimo",
#     "openai==2.21.0",
#     "polars==1.38.1",
#     "python-dotenv==1.2.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from openai import OpenAI
    import json
    from dotenv import load_dotenv
    import os

    return OpenAI, load_dotenv, os


@app.cell
def _(OpenAI, load_dotenv, os):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")

    client = OpenAI(api_key=OPENAI_API_KEY)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
