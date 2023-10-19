import os
from dotenv import load_dotenv

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex


def run_retrieval(query: str):
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]

    pinecone.init(api_key=api_key, environment=environment)
    index_name = "test-llamaindex-rag"

    pinecone_index = pinecone.Index(index_name=index_name)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store)
    query_engine = index.as_query_engine()

    result = query_engine.query(query)
    return result


if __name__ == "__main__":
    load_dotenv()
    result = run_retrieval("Can you tell me about the vision of Web3?")
    print(result)
