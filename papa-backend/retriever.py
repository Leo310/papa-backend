from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from typing import Any, List
from llama_index.schema import NodeWithScore
from llama_index.vector_stores import VectorStoreQuery


import os
from dotenv import load_dotenv

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding


class PineconeRetriever(BaseRetriever):
    """Retriever over a pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


def run_retrieval(query: str):
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENVIRONMENT"]

    pinecone.init(api_key=api_key, environment=environment)
    index_name = "test-llamaindex-rag"

    pinecone_index = pinecone.Index(index_name=index_name)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    embed_model = OpenAIEmbedding()
    retriever = PineconeRetriever(
        vector_store=pinecone_vector_store,
        embed_model=embed_model,
        query_mode="default",
        similarity_top_k=10,
    )
    retrieved_nodes = retriever.retrieve(query)
    return retrieved_nodes


if __name__ == "__main__":
    load_dotenv()
    retrieved_nodes = run_retrieval("Can you tell me about the vision of Web3?")
    for node in retrieved_nodes:
        print(node.node.id_)
        print(node.node.metadata)
        print(node.node.text)
        print(node.score)
        print("\n")
