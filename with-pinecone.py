# /// script
# dependencies = [
#     "marimo",
#     "pinecone==8.1.0",
#     "python-dotenv==1.2.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pinecone import Pinecone, ServerlessSpec
    import os
    from dotenv import load_dotenv

    return Pinecone, ServerlessSpec, load_dotenv, mo, os


@app.cell
def _(load_dotenv, os):
    load_dotenv()
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    return (PINECONE_API_KEY,)


@app.cell
def _(PINECONE_API_KEY, Pinecone):
    pc=Pinecone(api_key=PINECONE_API_KEY)
    return (pc,)


@app.cell
def _(ServerlessSpec, pc):
    pc.create_index(
        name="datacamp-index",
        dimension=1536,
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    return


@app.cell
def _(pc):
    pc.list_indexes()
    return


@app.cell
def _(pc):
    index=pc.Index("datacamp-index")
    return (index,)


@app.cell
def _(index):
    index.describe_index_stats()
    return


@app.cell
def _(mo):
    mo.md("""
    pinecone heirarchy: organization > projects > indexes > namespaces
    """)
    return


@app.cell
def _(pc):
    pc.list_indexes()
    return


@app.cell
def _(pc):
    pc.delete_index("datacamp-index")
    return


@app.cell
def _(pc):
    pc.list_indexes()
    return


@app.cell
def _(pc):
    pc.delete_index("my-first-index")
    return


@app.cell
def _(vectors):
    # for checking vector length
    vector_dims = [len(vector['values']) == 1536 for vector in vectors]
    all(vector_dims)
    return


@app.cell
def _(index):
    _ids = ['2','5','8']
    _fetched_vectors = index.fetch(ids=_ids)
    _fetched_vectors
    _meta = [_fetched_vectors['vectors'][id]["metadata"]for id in _ids]
    print(_meta)
    return


app._unparsable_cell(
    r"""
    # batching and chunking
    #start with a chunk function
    import itertools

    def chunks(iterable, batch_size=100):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))


    # sequential batching
    batch_index=pc.Index("datacamp-index")

    for chunk in chunks(vector):
        index.upsert(vectors=chunk)

    # can be slow - but rate limit friendly

    # parallel batching
    parallel_pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)

    with parallel_pc.Index('datacamp-index', pool_threads=30) as index:
        async_results = [index.upsert(vectors=chunk, async_req=True)]
            for chunk in chunks(vector, batch_size=100)
        [async_results.get() for async_result in async_results]

    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
