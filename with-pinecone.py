# /// script
# dependencies = [
#     "marimo",
#     "openai==2.21.0",
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


@app.cell
def _(PINECONE_API_KEY, Pinecone, pc, vector):
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
        async_results = [index.upsert(vectors=chunk, async_req=True) for chunk in chunks(vector, batch_size=100)]
        [async_results.get() for async_result in async_results]

    return (index,)


@app.cell
def _(client, index):
    # for semantic search 

    def retrieve(query, top_k, namespace, emb_model):
        query_response = client.embeddings.create(input=query, model=emb_model)
        query_emb = query_response.data[0].embedding
        docs = index.query(vector=query_emb, top_k=top_k, namespace=namespace, include_metadata=True)
        retrieved_docs = [doc['metadata']['text'] for doc in docs['matches']]
        sources = [(doc['metadata']['title'], doc['metadata']['url']) for doc in docs['matches']]
        return retrieved_docs, sources


    return


@app.cell
def _(api, index, os):
    # for rag
    from openai import OpenAI

    OPENAI_API_TOKEN=os.getenv("OPENAI_API_TOKEN")


    client = OpenAI(api)

    def retrieve(query, top_k, namespace, emb_model):

        #encode the input query using OpenAI
        query_response = client.embeddings.create(
            input=query,
            model=emb_model
        )

        query_emb=query_response.data[0].embedding

        docs = index.query(
            vector=query_emb, 
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )

        retrieved_docs = []
        sources = []
        for doc in docs['matches']:
            retrieved_docs.append(
                doc['metadata']['text']
            )
            sources.append(
                (
                    doc['metadata']['title'],
                    doc['metadata']['url']
                )
            )
            return retrieved_docs, sources

    def prompt_with_context_builder(query, docs):
        delim = '\n\n---\n\n'
        prompt_start = 'Answer the question based on the context below.\n\nContext:\n'
        prompt_end = f'\n\nQuestion: {query}\nAnswer:'

        prompt = prompt_start + delim.join(docs) + prompt_end
        return prompt

    def question_answering(prompt, sources, chat_model):
        sys_prompt = "You are a helpful assistant that always answers questions."

        res=client.chat.completions.create(
            model = chat_model,
            messages=[
                {"role":"system", "content":sys_prompt},
                {"role":"user", "content": prompt},
            ],
            temperature=0
        )
        answer = res.choices[0].message.content.strip()
        answer += "\n\nSources:"

        for source in sources:
            answer += "\n" + source[0] + ": " + source[1]

        return answer

    return (client,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
