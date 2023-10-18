import os
from dotenv import load_dotenv

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding

from llama_index import VectorStoreIndex
from llama_index.storage import StorageContext

load_dotenv()


def run_rag_pipeline():
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]

    pinecone.init(api_key=api_key, environment=environment)
    index_name = "test-llamaindex-rag"

    try:
        pinecone.create_index(
            index_name, dimension=1536, metric="euclidean", pod_type="p1"
        )
    except pinecone.exceptions.PineconeException as e:
        print(e)

    pinecone_index = pinecone.Index(index_name=index_name)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    textnode = TextNode(
        text="**Trust less** digital platform where sensitive and non-sensitive data can be exchanged. Give ownership back to the users/creators over their data by creating a **decentralized** platform (through [[Layer 2#DApps|DApps]]) which is governed by these users."
    )
    embed_model = OpenAIEmbedding()
    node_embedding = embed_model.get_text_embedding(
        textnode.get_content(metadata_mode="all")
    )
    textnode.embedding = node_embedding
    pinecone_vector_store.add([textnode])

    ####

    index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store)
    query_engine = index.as_query_engine()

    query = "Can you tell me about the vision of Web3?"
    results = query_engine.query(query)
    print(results)


if __name__ == "__main__":
    run_rag_pipeline()
