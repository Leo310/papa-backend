import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from tqdm import tqdm

import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from llama_index.schema import Document

from llama_index.node_parser import SimpleNodeParser

from markdown_reader import load_document


def load_documents():
    # LamaIndex support own Obsidian loader but we will copy theirs and build on top of it
    docs: List[Document] = []
    for dirpath, dirnames, filenames in os.walk("../knowledge_base"):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.endswith(".md"):
                filepath = os.path.join(dirpath, filename)
                content = load_document(Path(filepath))
                docs.extend(content)
    return docs


def embed_knowledge_base():
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

    docs = load_documents()
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=True)
    embed_model = OpenAIEmbedding()
    for node in tqdm(nodes):
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    pinecone_vector_store.add(nodes)


if __name__ == "__main__":
    load_dotenv()
    embed_knowledge_base()
    # docs = load_documents()
    # print(docs[0].metadata, docs[0])
