# /// script
# dependencies = [
#     "chromadb==1.5.1",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "openai==2.21.0",
#     "polars==1.38.1",
#     "pysqlite3==0.6.0",
#     "python-dotenv==1.2.1",
#     "scikit-learn==1.8.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md("""
    # Topic Analysis of Clothing Reviews with Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Project Instructions

    - [x] **Create and store the embeddings**
        - Embed the reviews using a suitable text embedding
        algorithm and store them as a list in the variable `embeddings`.

    - [x] **Dimensionality reduction and visualization**
        - Apply an appropriate dimensionality reduction technique
        to reduce the `embeddings` to a 2-dimensional numpy array
        and store this array in the variable `embeddings_2d`.
        - Then, use this variable to plot a 2D visual representation
        of the reviews.
    - [ ] **Feedback categorization**
        - Use your embeddings to identify some reviews that
        discuss topics such as 'quality', 'fit', 'style', 'comfort' etc.
    - [ ] **Similarity search function**
        - Write a function that outputs the closest 3 reviews to a given
        input review, enabling a more personalized customer service response.
        - Apply this function to the first review
        *'Absolutely wonderful - silky and sexy and comfortable'* and
        store the output as a list in the variable `most_similar_reviews`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## **Project Context**

    Welcome to the world of e-commerce, where customer feedback is a goldmine of insights!
    In this project, you'll dive into the Women's Clothing E-Commerce Reviews dataset, focusing on the
    'Review Text' column filled with direct customer opinions.

    Your mission is to use text embeddings and Python to analysze these reviews, uncover underlying themes,
    and understand customer sentiments. This analysis will help improve customer service and
    product offerings.


    ## **The Data**

    You will be working with a dataset specifically focusing on customer reviews.
    Below is the data dictionary for the relevant field:

    ### **womens_clothing_e-commerce_reviews.csv**

    | Column        | Description                           |
    |---------------|---------------------------------------|
    | `'Review Text'` | Textual feedback provided by customers about their shopping experience and product quality. |

    Armed with access to powerful embedding API services, you will process the reviews,
    extract meaningful insights, and present your findings.

    Let's get started!
    """)
    return


@app.cell
def _(reviews):
    reviews.head().collect()
    return


@app.cell
def _(pl, reviews):
    print(reviews.select(pl.col("Division Name").unique()).collect())
    print(reviews.select(pl.col("Department Name").unique()).collect())
    print(reviews.select(pl.col("Class Name").unique()).collect())
    return


@app.cell
def _(reviews):
    reviews.head().collect().to_dicts()
    return


@app.cell
def _(reviews):
    reviews_dict=reviews.collect().to_dicts()
    return (reviews_dict,)


@app.cell
def _(reviews_dict):
    metadatas = []

    for review in reviews_dict:
        meta ={
            "clothing_id": review["Clothing ID"],
            "reviewer_age": review["Age"],
            "rating": review["Rating"],
            "recommened":review["Recommended IND"],
            "pos_feedback_count":review["Positive Feedback Count"],
            "division":review["Division Name"],
            "department":review["Department Name"],
            "class":review["Class Name"]
        }
        metadatas.append(meta)
    metadatas
    return (metadatas,)


@app.cell
def _(reviews_dict):
    review_texts = []

    for document in reviews_dict:
        doco = f"title: {(document.get("Title", " ") or " ")}, text: {document['Review Text']}"
        review_texts.append(doco)
    
    review_texts
    return (review_texts,)


@app.cell
def _(pl, reviews):
    ids= reviews.select(pl.col("Review ID")).collect().to_series().to_list()
    ids = [str(num) for num in ids]
    print(ids)
    return (ids,)


@app.cell
def _(mo):
    mo.md("""
    ## Setup ChromaDB
    """)
    return


@app.cell
def _(chromadb):
    # using a local chromadb client
    chroma_client=chromadb.PersistentClient()
    return (chroma_client,)


@app.cell
def _(OPENAI_API_KEY, OpenAIEmbeddingFunction, chroma_client):
    # create the collection
    collection=chroma_client.create_collection(
        name="clothing_reviews",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
    )
    return (collection,)


@app.cell
def _(collection, ids, metadatas, review_texts):
    collection.upsert(
        ids=ids,
        documents=review_texts,
        metadatas=metadatas,
    )
    return


@app.cell
def _(collection):
    collection.peek()
    return


@app.cell
def _(collection, mo):
    mo.md("create embeddings object")

    embeddings_dict=collection.get(
        include=["embeddings"]
    )
    embeddings=embeddings_dict.get("embeddings")

    embeddings
    return (embeddings,)


@app.cell
def _(TSNE, embeddings):
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)
    embeddings_2d
    return (embeddings_2d,)


@app.cell
def _(embeddings_2d, plt, reviews_dict):
    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])

    topics = [clothing["Class Name"] for clothing in reviews_dict]

    for _i, topic in enumerate(topics):
        plt.annotate(topic, (embeddings_2d[_i, 0], embeddings_2d[_i, 1]))

    plt.show()

    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("""
    ### Dependencies
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from openai import OpenAI
    import json
    from dotenv import load_dotenv
    import os
    import pysqlite3
    import chromadb
    import sys
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt

    return (
        OpenAI,
        OpenAIEmbeddingFunction,
        TSNE,
        chromadb,
        load_dotenv,
        mo,
        os,
        pl,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Setup Environment
    """)
    return


@app.cell
def _(OpenAI, load_dotenv, os):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_TOKEN")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return (OPENAI_API_KEY,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Data
    """)
    return


@app.cell
def _(pl):
    reviews = pl.scan_csv("womens_clothing_e-commerce_reviews.csv")
    return (reviews,)


if __name__ == "__main__":
    app.run()
