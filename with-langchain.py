# /// script
# dependencies = [
#     "langchain-core==1.2.15",
#     "langchain-openai==1.1.10",
#     "marimo",
#     "openai==2.23.0",
#     "polars==1.38.1",
#     "pydantic-ai==1.63.0",
#     "python-dotenv==1.2.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from openai import OpenAI
    import json
    from dotenv import load_dotenv
    import os
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    return (
        ChatOpenAI,
        ChatPromptTemplate,
        PromptTemplate,
        StrOutputParser,
        load_dotenv,
        os,
    )


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")

    # client = OpenAI(api_key=OPENAI_API_KEY)
    return (OPENAI_API_KEY,)


@app.cell
def _(ChatOpenAI, OPENAI_API_KEY, PromptTemplate):
    # prompt templtes and chains with LCEL: langchain expression language
    # Create a prompt template
    template = "You are an artificial intelligence assistant, answer the question. {question}"
    prompt_prompt = PromptTemplate.from_template(template=template)

    # Define LLM and create a chain
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    llm_prompt_chain = prompt_prompt | llm
    return llm, llm_prompt_chain


@app.cell
def _(llm_prompt_chain):
    # Invoke the chain
    question = "How does LangChain make LLM application development easier?"
    print(llm_prompt_chain.invoke({"question": question}))
    return


@app.cell
def _(ChatPromptTemplate, llm):
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise, friendly expert that helps with questions.",
            ),
            ("human", "Please provide the best chicken 65 recipe you can find."),
            ("ai", "Here is a highly rated recipe I found at...."),
            ("human", "{question}"),
        ]
    )

    llm_chat_chain = chat_prompt_template | llm

    _question = (
        "Please provide the greatest oatmeal raisin cookie recipe you can find."
    )
    _response = llm_chat_chain.invoke({"question": _question})
    print(_response.content)
    return


@app.cell
def _(PromptTemplate, llm):
    from langchain_core.prompts import FewShotPromptTemplate

    examples = [
        {"question": "What is my favorite programming language?", "answer": "R"},
        {"question": "What is my favorite cookie?", "answer": "oatmeal raisin"},
    ]

    example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

    # few shot prompt

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    _prompt = few_shot_prompt_template.invoke(
        {"input": "What is my favorite programming cookie?"}
    )
    print(_prompt.text)

    _llm_chain = few_shot_prompt_template | llm

    _response = _llm_chain.invoke(
        {"input": "What is my favorite programming cookie?"}
    )
    print(_response.content)
    return


@app.cell
def _(PromptTemplate):
    # building prompts for sequential chains
    # Create a prompt template that takes an input activity
    learning_prompt = PromptTemplate(
        input_variables=["activity"],
        template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?",
    )

    # Create a prompt template that places a time constraint on the output
    time_prompt = PromptTemplate(
        input_variables=["learning_plan"],
        template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan}.",
    )

    # Invoke the learning_prompt with an activity
    print(learning_prompt.invoke({"activity": "program in Rust"}))
    return learning_prompt, time_prompt


@app.cell
def _(StrOutputParser, learning_prompt, llm, time_prompt):
    _seq_chain = (
        {"learning_plan": learning_prompt | llm | StrOutputParser()}
        | time_prompt
        | llm
        | StrOutputParser()
    )

    print(_seq_chain.invoke({"activity": "program in Rust"}))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
